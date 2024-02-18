"""
Example template for defining a system
"""
import json
import copy
import time
import inspect
import itertools
from omegaconf import DictConfig
from typing import Any, Optional
from pathlib import Path
import torch
from torch.nn import Module
from typing import Union, Callable
from lightning import LightningModule
from torch.optim.optimizer import Optimizer
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning_utilities.core.rank_zero import rank_zero_only
from torchmetrics import MaxMetric, MeanMetric

from src.models.nn.autobackend import AutoBackend
from src.utils.torch_utils import de_parallel
from src.utils.checks import check_imgsz
from src.utils.ops import Profile
from src.models.nn.tasks import guess_model_task, nn
from src.utils import LOGGER, NORMS, checks, emojis, change_config
from src.models.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel

__all__ = [
    'BaseTrainModule',
]

class BaseTrainModule(LightningModule):
    """
    Sample model to show how to define a tmeplate
    """

    def __init__(
        self, 
        net: Union[DictConfig, Path, str],
        optimizer: DictConfig,
        lr_schedule: DictConfig,
        warmup: DictConfig,
        val: DictConfig,
        pre: DictConfig,
        imgsz: int,
        save_dir: str,
        names: DictConfig,
        close_mosaic: int,              # close mosaic after certain epochs
        hyp: DictConfig,
        task: str,                      # task name
        interval: int = 10,             # print interval
        metrics: dict[Any, int] = None, # validation/training metrics
        plots: dict = {},               
        weight_averager: Module = None, 
        fitness: float = None,          
        loss: float = None,             
        tloss: float = None,            
        plot_idx: list[int] = [0, 1, 2],
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = None
        self.ckpt  = None
        self.ckpt_path = None
        self.overrides = {}

        self.loss = torch.zeros(1, device=self.device)
        self.loss_names = ["Loss"]
        self.save_dir = Path(save_dir)
        # Epoch level metrics
        self.csv = self.save_dir / 'results.csv'
        self.validator = self.get_validator()

        # Load or create new YOLO model
        if isinstance(net, (str, Path)):
            self._load(net, task)
        else:
            self._new(net, task)

        self.set_model_attributes()

        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.example_input_array = torch.randn(1, 3, h, w)

    def forward(self, inputs):
        return self.model(inputs)

    def on_train_start(self):
        """Called when the train begins."""

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32) # grid size (max stride)
        self.hparams.imgsz = check_imgsz(self.hparams.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multi-scale training

        # Initialize metrics
        def _init_metrics(keys):
            return dict(zip(keys, [0] * len(keys)))

        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
        self.metrics = _init_metrics(metric_keys)

        if self.hparams.val.plots:
            self.plot_training_labels()

        if self.hparams.close_mosaic:
            base_idx = (self.trainer.max_epochs - self.hparams.close_mosaic) * self.trainer.num_training_batches
            self.hparams.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

    def on_train_epoch_start(self):
        self.tloss = None
        if self.current_epoch == (self.trainer.max_epochs - self.hparams.close_mosaic) and self.trainer.datamodule.mosaic_enabled:
            self.trainer.datamodule.enable_mosaic(enable=False)
            self.trainer.datamodule.setup('fit')
            # for batch in self.trainer.datamodule.trainDataloader:
            #     for count in range(3):
            #         self.plot_training_samples(batch[count], 1, count, name='close_mosaic')
            #     break

    def training_step(self, batch, batch_idx):
        self.ni = batch_idx + self.trainer.num_training_batches * self.current_epoch
        
        batch = self.preprocess_batch(batch)
        self.loss, self.loss_items = self(batch)
        self.tloss = (self.tloss * batch_idx + self.loss_items) / (batch_idx + 1) if self.tloss is not None \
            else self.loss_items
        
        # Log
        plots = self.hparams.val.plots
        loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
        losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
        loss_messge = {k: round(float(v), 5) for k, v in self.label_loss_items(losses).items()}
        for key, value in loss_messge.items():
            self.log(key, value, batch_size=len(batch["im_file"]), on_step=False, on_epoch=True, prog_bar=True)
        # self.print_log(batch_idx, True, self.losses)
        if plots and self.ni in self.hparams.plot_idx:
            self.plot_training_samples(batch, self.ni)

        return self.loss

    def on_validation_epoch_start(self) -> None:
        self.validator.loss = torch.zeros_like(self.loss_items, device=self.device)
        change_config(self.validator.args, "plots", self.current_epoch == self.trainer.max_epochs - 1, True)
        self.dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        self.validator.init_metrics(de_parallel(self.model))

    def validation_step(self, batch, batch_idx):
        # Preprocess
        with self.dt[0]:
            batch = self.validator.preprocess(batch, self.device)

        # Inference
        with self.dt[1]:
            preds = self.model(batch["img"])

        # Loss
        with self.dt[2]:
            losses = self.model.loss(batch, preds)[1]
            self.validator.loss += losses

        # Postprocess
        with self.dt[3]:
            preds = self.validator.postprocess(preds)

        self.validator.update_metrics(preds, batch)
        if self.validator.args.plots and batch_idx < 3:
            self.validator.plot_val_samples(batch, batch_idx)
            self.validator.plot_predictions(batch, preds, batch_idx)

    def on_validation_epoch_end(self) -> None:
        stats = self.validator.get_stats()
        val_dataset_quantity = len(self.trainer.datamodule.val_dataset)
        self.validator.check_stats(stats)
        self.validator.speed = dict(zip(self.validator.speed.keys(), (x.t / val_dataset_quantity * 1e3 for x in self.dt)))
        self.validator.finalize_metrics()
        # self.validator.print_results()
        results = {**stats, **self.label_loss_items(self.validator.loss.cpu() / val_dataset_quantity, prefix="val")}
        self.metrics = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        keys = [f"val/{x}" for x in self.loss_names]
        for key, value in self.metrics.items():
            if key in keys:
                self.log(key, value, on_step=False, on_epoch=True, prog_bar=True)
        self.fitness = self.metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        self.log("val/fitness", self.fitness, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self) -> None:
        first_three_pairs = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.trainer.optimizers[0].param_groups)}  # for loggers
        self.lr = dict(itertools.islice(first_three_pairs.items(), 3))
        self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})

    def on_test_start(self) -> None:
        # best = self.trainer.checkpoint_callback.best_model_path  # set best metric for test
        change_config(self.validator.args, "plots", self.hparams.val.plots)
        self.model = AutoBackend(
            self.model,
            device=self.device,
            dnn=self.validator.args.dnn,
            fp16=self.validator.args.half
        )
        stride, pt, jit, engine = self.model.stride, self.model.pt, self.model.jit, self.model.engine

        imgsz = check_imgsz(self.hparams.imgsz, stride=stride)
        if engine:
            self.validator.args.device.batchsize_per_gpu = self.model.batch_size
        elif not pt and not jit:
            self.validator.args.device.batchsize_per_gpu = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
        if not pt:
            self.validator.args.rect = False
        self.model.warmup(imgsz=(1 if pt else self.validator.args.device.batchsize_per_gpu, 3, imgsz, imgsz))  # warmup
        self.dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        self.validator.init_metrics(de_parallel(self.model))
        self.jdict = []  # empty before each val

    def test_step(self, batch, batch_idx):
        # Preprocess
        with self.dt[0]:
            batch = self.validator.preprocess(batch, self.device)

        # Inference
        with self.dt[1]:
            preds = self.model(batch["img"], augment=self.hparams.pre.augment)

        # Postprocess
        with self.dt[3]:
            preds = self.validator.postprocess(preds)

        self.validator.update_metrics(preds, batch)
        if self.validator.args.plots and batch_idx < 3:
            self.validator.training = False
            self.validator.plot_val_samples(batch, batch_idx)
            self.validator.plot_predictions(batch, preds, batch_idx)

    def on_test_epoch_end(self) -> None:
        stats = self.validator.get_stats()
        self.validator.check_stats(stats)
        test_dataset_quantity = len(self.trainer.datamodule.test_dataset)
        self.validator.speed = dict(zip(self.validator.speed.keys(), (x.t / test_dataset_quantity * 1e3 for x in self.dt)))
        self.validator.finalize_metrics()
        self.validator.print_results()
        LOGGER.info(
            "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
            % tuple(self.validator.speed.values())
        )
        if self.validator.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.validator.args.plots or self.validator.args.save_json:
            LOGGER.info(f"Results saved to {self.save_dir}")
        self.metrics = stats
        self.metrics.pop("fitness", None)

    def on_test_end(self) -> None:
        if self.validator.args.plots:
            self.plot_metrics()

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """

        optimizer_cfg = self.hparams.optimizer

        optimizer = self._build_optimizer(optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.hparams.lr_schedule)
        name = schedule_cfg.pop("name")     
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.hparams.warmup.steps:
            if self.hparams.warmup.name == "constant":
                k = self.hparams.warmup.ratio
            elif self.hparams.warmup.name == "linear":
                k = 1 - (
                    1 - self.trainer.global_step / self.hparams.warmup.steps
                ) * (1 - self.hparams.warmup.ratio)
            elif self.hparams.warmup.name == "exp":
                k = self.hparams.warmup.ratio ** (
                    1 - self.trainer.global_step / self.hparams.warmup.steps
                )
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    @rank_zero_only
    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.current_epoch + 1] + vals)).rstrip(",") + "\n")

    def print_log(self, batch_idx, is_train, losses):
        flag = batch_idx % self.hparams.interval == 0 or batch_idx == self.trainer.num_training_batches - 1
        _type = 'Train' if is_train else 'Valid'
        all_step = self.trainer.num_training_batches if is_train else self.trainer.num_val_batches
        if not is_train and int(all_step[0]) != batch_idx+1:
            return
        memory = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        loss = list([l for l in losses])
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        log_msg = self.get_log_message(_type, batch_idx, all_step, memory, lr, self.loss_names, loss)

        if flag:
            LOGGER.info(log_msg)

    def get_log_message(self, _type, batch_idx, all_step, memory, lr, all_loss, loss):
        log_msg = "{}|Epoch{}/{}|Iter({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
            _type,
            self.current_epoch + 1,
            self.trainer.max_epochs,
            batch_idx,
            all_step,
            memory,
            lr,
        )
        log_msg += "| ".join(["{}:{:.4f}".format(loss_name, loss_val) for loss_name, loss_val in zip(all_loss, loss)])
        return log_msg

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.hparams.plots[name] = {'data': data, 'timestamp': time.time()}
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def get_validator(self):
        """
        Get validator
        """
        raise NotImplementedError("Not validator")
    
    @rank_zero_only
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.hparams.names

    # TODO: may need to put these following functions into callback
    @rank_zero_only
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass

    @rank_zero_only
    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def _new(self, cfg: DictConfig, task=None, model=None):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
        """
        cfg_dict = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load())(cfg_dict)  # build model
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        weights = checks.check_file(weights)
        self.model, self.ckpt = weights, None
        self.task = task or guess_model_task(weights)
        self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task


    def _check_is_pytorch_model(self):
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, "names") else None
    
    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, "transforms") else None
    
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
    
    def _smart_load(self):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e
        
    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        return {
            "classify": ClassificationModel,
            "detect": DetectionModel,
            "segment": SegmentationModel,
            "pose": PoseModel,
            "obb": OBBModel,
                }
    
    def _build_optimizer(self, config):
        """Build optimizer from config.

        Supports customised parameter-level hyperparameters.
        The config should be like:
        >>> optimizer:
        >>>   name: AdamW
        >>>   lr: 0.001
        >>>   weight_decay: 0.05
        >>>   no_norm_decay: True
        >>>   param_level_cfg:  # parameter-level config
        >>>     backbone:
        >>>       lr_mult: 0.1
        """
        config = copy.deepcopy(config)
        param_dict = {}
        no_norm_decay = config.pop("no_norm_decay", False)
        no_bias_decay = config.pop("no_bias_decay", False)
        param_level_cfg = config.pop("param_level_cfg", {})
        base_lr = config.get("lr", None)
        base_wd = config.get("weight_decay", None)

        name = config.pop("name")
        optim_cls = getattr(torch.optim, name)

        # custom param-wise lr and weight_decay
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            param_dict[p] = {"name": name}

            for key in param_level_cfg:
                if key in name:
                    if "lr_mult" in param_level_cfg[key] and base_lr:
                        param_dict[p].update(
                            {"lr": base_lr * param_level_cfg[key]["lr_mult"]}
                        )
                    if "decay_mult" in param_level_cfg[key] and base_wd:
                        param_dict[p].update(
                            {"weight_decay": base_wd * param_level_cfg[key]["decay_mult"]}
                        )
                    break
        if no_norm_decay:
            # update norms decay
            for name, m in self.model.named_modules():
                if isinstance(m, NORMS):
                    param_dict[m.bias].update({"weight_decay": 0})
                    param_dict[m.weight].update({"weight_decay": 0})
        if no_bias_decay:
            # update bias decay
            for name, m in self.model.named_modules():
                if hasattr(m, "bias"):
                    param_dict[m.bias].update({"weight_decay": 0})

        # convert param dict to optimizer's param groups
        param_groups = []
        for p, pconfig in param_dict.items():
            name = pconfig.pop("name", None)
            if "weight_decay" in pconfig or "lr" in pconfig:
                LOGGER.info(f"special optimizer hyperparameter: {name} - {pconfig}")
            param_groups += [{"params": p, **pconfig}]

        optimizer = optim_cls(param_groups, **config)
        return optimizer
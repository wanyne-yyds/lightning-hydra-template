import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset
import math
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Union, Callable
from lightning_utilities.core.rank_zero import rank_zero_only

from src.utils.plotting import plot_images, plot_labels, plot_results
from src.models.base import BaseTrainModule
from .detect_validator_module import DetectionValidator

__all__ = [ 'DetectionTrainModule' ]

class DetectionTrainModule(BaseTrainModule):
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
        close_mosaic: int,
        multi_scale: bool,
        hyp: DictConfig,
        task: str,
    ) -> None:
        super().__init__(net, optimizer, lr_schedule, warmup, pre, val, imgsz, save_dir, names, close_mosaic, hyp, task)

        self.loss_items = torch.zeros(3, device=self.device)  # (box, obj, cls)
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(self.hparams.val, self.save_dir)
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    @rank_zero_only
    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        dataset =  self.trainer.datamodule.train_dataloader().dataset
        _indices = dataset.indices
        if isinstance(dataset, Subset):
            boxes = np.concatenate([lb["bboxes"] for index, lb in enumerate(dataset.dataset.labels) if index in _indices], 0)
            cls = np.concatenate([lb["cls"] for index, lb in enumerate(dataset.dataset.labels) if index in _indices], 0)
            plot_labels(boxes, cls.squeeze(), names=self.hparams.names, save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].float() / 255
        if self.hparams.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.hparams.imgsz * 0.5, self.hparams.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    @rank_zero_only
    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        if self.csv.exists():
            plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    @rank_zero_only
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = len(self.hparams.names)  # attach number of classes to model
        self.model.names = self.hparams.names  # attach class names to model
        self.model.hyp = self.hparams.hyp  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
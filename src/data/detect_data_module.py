import random
from torch.utils.data import Subset
from omegaconf import DictConfig, ListConfig
from .components.base_data_module import BaseDataModule
from .components.build import build_dataloader, build_yolo_dataset

class DetectDataModule(BaseDataModule):
    
    def __init__(
        self,
        train_dir: str = None,
        test_dir: str = None,
        task: str = None,
        imgsz: int = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
        single_cls: bool = False,
        names: DictConfig = None,
        split_dataset: DictConfig = None,
        augmentation: DictConfig = None,
    ) -> None:
        super().__init__()

    def build_dataset(self, mode='train'):
        """Build YOLO Dataset

        Args:
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        """
        if (mode != 'test') and self.hparams.split_dataset.enable:
            traindataset = build_yolo_dataset(self.hparams.train_dir, self.hparams.imgsz, self.hparams.batch_size, 
                                              self.hparams.augmentation, self.hparams.single_cls, self.hparams.names,
                                              mode=mode, rect=False, stride=32)
            valdataset = build_yolo_dataset(self.hparams.train_dir, self.hparams.imgsz, self.hparams.batch_size, 
                                            self.hparams.augmentation, self.hparams.single_cls, self.hparams.names,
                                            mode='val', rect=True, stride=32)
            _len = len(traindataset)
            train_set_size = int(_len * self.hparams.split_dataset.split_ratio[0])
            valid_set_size = int(_len * self.hparams.split_dataset.split_ratio[1])
            data_list = range(_len)
            if self.hparams.split_dataset.shuffle:
                data_list = random.sample(range(_len), _len)
                train_dataset = Subset(traindataset, data_list[:train_set_size])
                val_dataset = Subset(valdataset, data_list[train_set_size:train_set_size + valid_set_size])
            else:
                train_dataset = Subset(traindataset, range(train_set_size))
                val_dataset = Subset(valdataset, range(train_set_size, train_set_size + valid_set_size))
            
            return train_dataset, val_dataset
        
        return build_yolo_dataset(self.hparams.test_dir, self.hparams.imgsz, self.hparams.batch_size, 
                                  self.hparams.augmentation, self.hparams.single_cls, self.hparams.names, 
                                  mode=mode, rect=mode == 'test', stride=32)

    def dataloader(self, dataset, batch_size=16, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        assert mode in ['train', 'test']
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            # self.logger.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.hparams.num_workers if mode == 'train' else self.hparams.num_workers * 2
        pin_memory = self.hparams.pin_memory
        return build_dataloader(dataset, batch_size, workers, pin_memory, shuffle)  # return dataloader
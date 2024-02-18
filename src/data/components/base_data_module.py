from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, Optional, Tuple
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils import LOGGER

class BaseDataModule(LightningDataModule):

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

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.mosaic_enabled = True
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage == 'fit':
            self.train_dataset, self.val_dataset = self.build_dataset('train')
            if not self.mosaic_enabled:
                if hasattr(self.train_dataset.dataset, "close_mosaic"):
                    LOGGER.info("Closing dataloader mosaic")
                    self.train_dataset.dataset.close_mosaic(hyp=self.hparams.augmentation)


            # split the train set into two
            LOGGER.info('training data loader called')
            LOGGER.info(f'train dataset len: {len(self.train_dataset)}')
            LOGGER.info('valid data loader called')
            LOGGER.info(f'val dataset len: {len(self.val_dataset)}')

        if stage == 'test':
            LOGGER.info('testing data loader called')
            self.test_dataset = self.build_dataset('test')
            LOGGER.info(f'test dataset len: {len(self.test_dataset)}')

        if stage == 'predict':
            LOGGER.info('predicting data loader called')
            raise NotImplementedError('predicting data loader not implemented')


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.dataloader(self.train_dataset, self.hparams.batch_size, mode='train')

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self.dataloader(self.val_dataset, self.hparams.batch_size, mode='test')

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self.dataloader(self.test_dataset, self.hparams.batch_size, mode='test')

    def predict_dataloader(self) -> DataLoader[Any]:
        """
        Return a data loader.
        """
        raise NotImplementedError('predict_dataloader not implemented')

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def dataloader(self, dataset_path: str, batch_size: int = 16, mode: str = 'train') -> DataLoader[Any]:
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def build_dataset(self, mode: str = 'train') -> Dataset[Any]:
        """Build dataset"""
        raise NotImplementedError('build_dataset function not implemented in trainer')

    def enable_mosaic(self, enable: bool) -> None:
        self.mosaic_enabled = enable
    
    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
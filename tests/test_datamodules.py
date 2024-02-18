from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule, LightningModule
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import LOGGER
from src.utils.plotting import plot_images
from src.data.detect_data_module import DetectDataModule

@hydra.main(version_base="1.3", config_path="../configs/data", config_name="detect")
def main(cfg: DictConfig):
    """
    Main training routine specific for this project
    """

    LOGGER.info(f"Instantiating datamodule <{cfg._target_}>")
    dm: LightningModule = hydra.utils.instantiate(cfg)
    save_dir = Path("./")

    #* Train
    if 0:
        # dm.setup(stage='fit')
        # dm.enable_mosaic(enable=False)
        dm.setup('fit')
        for batch in dm.train_dataloader():
            batch_ = batch
            fname = save_dir.joinpath('train_batch.jpg')
            plot_images(images=batch_['img'],
                        batch_idx=batch_['batch_idx'],
                        cls=batch_['cls'].squeeze(-1),
                        bboxes=batch_['bboxes'],
                        paths=batch_['im_file'],
                        fname=fname)
            break

    #* Valid
    if 0:
        dm.setup(stage='fit')
        for batch in dm.val_dataloader():
            batch_ = batch
            fname = save_dir.joinpath('val_batch.jpg')
            plot_images(images=batch_['img'],
                        batch_idx=batch_['batch_idx'],
                        cls=batch_['cls'].squeeze(-1),
                        bboxes=batch_['bboxes'],
                        paths=batch_['im_file'],
                        fname=fname)
            break

    # #* Test
    # dm.teardown(stage="fit")
    if 0:
        dm.setup(stage='test')
        for batch in dm.test_dataloader():
            batch_ = batch
            fname = save_dir.joinpath('test_batch.jpg')
            plot_images(images=batch_['img'],
                        batch_idx=batch_['batch_idx'],
                        cls=batch_['cls'].squeeze(-1),
                        bboxes=batch_['bboxes'],
                        paths=batch_['im_file'],
                        fname=fname)
            break

if __name__ == '__main__':
    main()

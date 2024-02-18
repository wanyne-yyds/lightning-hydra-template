# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import torch
import random
import numpy as np
from torch.utils.data import dataloader, Subset
from torch.utils.data.dataloader import DataLoader

from .dataset import YOLODataset


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers. Uses same syntax as vanilla DataLoader."""

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """Reset iterator.
        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(img_path, imgsz, batch, augmentation, single_cls, 
                       classes, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return YOLODataset(
        img_path=img_path,
        imgsz=imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=augmentation,  # TODO: probably add a get_hyps_from_cfg function
        rect=rect,  # rectangular batches
        single_cls=single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=f'{mode}: ',
        classes=classes)


def build_dataloader(dataset, batch, workers, pin_memory=True, shuffle=True, collate_fn=None):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None
    # generator = torch.Generator()
    # generator.manual_seed(6148914691236517205)

    # 通用的参数
    dataloader_params = {
        'dataset': dataset,
        'batch_size': batch,
        'shuffle': shuffle and sampler is None,
        'num_workers': nw,
        'sampler': sampler,
        'pin_memory': pin_memory,
        'persistent_workers': True,
        # 'worker_init_fn': seed_worker,
        # 'generator': generator
    }

    if collate_fn==None:
        # 处理 collate_fn
        if isinstance(dataset, Subset):
            collate_fn = getattr(dataset.dataset, 'collate_fn', None)
        else:
            collate_fn = getattr(dataset, 'collate_fn', None)

        dataloader_params['collate_fn'] = collate_fn
        return DataLoader(**dataloader_params)
        # return InfiniteDataLoader(**dataloader_params)

    else:
        dataloader_params['collate_fn'] = collate_fn
        return DataLoader(**dataloader_params)
        # return InfiniteDataLoader(**dataloader_params)

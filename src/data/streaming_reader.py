import torch
from torch.utils.data import DataLoader

from .distributed_dataset import DistributedDataset


class ChunkedDataReader:
    """
    Adapter that wraps DistributedDataset + DataLoader to provide the
    sample_batch() interface expected by the training loop.
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        sequence_length,
        seed=1337,
        num_workers=0,
        is_eval=False,
        dp_rank=0,
        dp_world_size=1,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._dp_world_size = dp_world_size
        self.is_eval = is_eval

        self.dataset = DistributedDataset(
            data_dir=data_dir,
            seq_len=sequence_length,
            shuffle=not is_eval,
            seed=seed,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            strict=not is_eval,
        )

        self.num_tokens = self.dataset.total_tokens

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=not is_eval,
        )

        self.step = 0
        self._epoch = 0
        self._dataloader_iter = None

    def sample_batch(self):
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.dataloader)

        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._epoch += 1
            self.dataset.set_epoch(self._epoch)
            self.dataset.global_skip_batches = 0
            self._dataloader_iter = iter(self.dataloader)
            batch = next(self._dataloader_iter)

        self.step += 1
        x = batch["input_ids"]
        y = batch["labels"]
        return x, y

    def set_step(self, step):
        self.step = step
        if step == 0:
            self._dataloader_iter = None
            return

        total_consumed = step * self.batch_size * self._dp_world_size
        self._epoch = total_consumed // self.dataset.total_samples
        skip_in_epoch = total_consumed % self.dataset.total_samples
        self.dataset.epoch = self._epoch
        self.dataset.global_skip_batches = skip_in_epoch
        self._dataloader_iter = None

    def num_batches(self):
        return self.dataset.total_samples // self.batch_size

    def __len__(self):
        return self.num_tokens - self.sequence_length - 1

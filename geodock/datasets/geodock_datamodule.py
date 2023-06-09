from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from geodock.datasets.geodock_dataset import GeoDockDataset


class GeoDockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set: str,
        val_set: str,
        test_set: str,
        batch_size: int = 1,
        is_training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = GeoDockDataset(
            dataset=self.train_set,
            is_training=self.is_training,
        )
        self.data_val = GeoDockDataset(
            dataset=self.val_set,
            is_training=self.is_training,
        )
        self.data_test = GeoDockDataset(
            dataset=self.test_set,
            is_training=self.is_training,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

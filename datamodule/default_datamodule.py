from abc import ABC, abstractmethod

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class DefaultDataModule(LightningDataModule):
    """Base class for pytorch-lightning DataModule datasets. Subclass this if you have a standard train, validation,
    test split."""

    def __init__(self, train_conf, test_conf, num_workers):
        super().__init__()

        self.train_conf = train_conf
        self.test_conf = test_conf
        self.num_workers = num_workers

    @property
    @abstractmethod
    def train_ds(self):
        """Build the train pytorch dataset.

        :return: the train pytorch dataset
        """

    @property
    @abstractmethod
    def val_ds(self):
        """Build the validation pytorch dataset.

        :return: the validation pytorch dataset
        """

    @property
    @abstractmethod
    def test_ds(self):
        """Build the test pytorch dataset.

        :return: the test pytorch dataset
        """

    def train_dataloader(self):
        train_dl = DataLoader(self.train_ds,
                              self.train_conf.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_ds,
                            self.test_conf.batch_size,
                            num_workers=self.num_workers)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_ds,
                             self.test_conf.batch_size,
                             num_workers=self.num_workers)
        return test_dl

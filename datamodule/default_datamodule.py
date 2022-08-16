from abc import abstractmethod
from typing import Optional, Callable, Iterable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Dataset

from datamodule import DatasetSplit


class AbstractDefaultDataModule(LightningDataModule):
    """Base class for pytorch-lightning DataModule datasets. Subclass this if you have a standard train, validation,
    test split."""

    def __init__(self,
                 train_batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 persistent_workers: bool,
                 pin_memory: bool):
        super().__init__()

        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size

        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        # setup train and validation datasets
        if stage in (None, 'fit'):
            self._train_ds, self._val_ds = self.train_ds(), self.val_ds()
        # setup test dataset
        if stage in (None, 'test'):
            self._test_ds = self.test_ds()

    @abstractmethod
    def train_ds(self) -> Dataset:
        """Build the train pytorch dataset.

        :return: the train pytorch dataset.
        """

    @abstractmethod
    def val_ds(self) -> Dataset:
        """Build the validation pytorch dataset.

        :return: the validation pytorch dataset.
        """

    @abstractmethod
    def test_ds(self) -> Dataset:
        """Build the test pytorch dataset.

        :return: the test pytorch dataset.
        """

    def train_dataloader(self) -> DataLoader:
        train_dl = DataLoader(self._train_ds,
                              self._train_batch_size,
                              shuffle=True,
                              num_workers=self._num_workers,
                              pin_memory=self._pin_memory,
                              collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
                              persistent_workers=self._persistent_workers)
        return train_dl

    def val_dataloader(self) -> DataLoader:
        val_dl = DataLoader(self._val_ds,
                            self._test_batch_size,
                            num_workers=self._num_workers,
                            pin_memory=self._pin_memory,
                            collate_fn=self.build_collate_fn(DatasetSplit.VALIDATION),
                            persistent_workers=self._persistent_workers)
        return val_dl

    def test_dataloader(self) -> DataLoader:
        test_dl = DataLoader(self._test_ds,
                             self._test_batch_size,
                             num_workers=self._num_workers,
                             pin_memory=self._pin_memory,
                             collate_fn=self.build_collate_fn(DatasetSplit.TEST),
                             persistent_workers=self._persistent_workers)
        return test_dl

    # noinspection PyMethodMayBeStatic
    def build_collate_fn(self, split: DatasetSplit = None) -> [Callable, None]:
        """Override to define a custom collate function. Build a function for collating multiple data instances into
        a batch. Defaults to returning `None` since it's the default for DataLoader's collate_fn argument.

        While different collate functions might be needed depending on the dataset split, in most cases the same
        function can be returned for all data splits.

        :param split: The split that the collate function is used on to build batches. Can be ignored when train and
        test data share the same structure.
        :return: a single argument function that takes a list of tuples/instances and returns a batch as tensor or a
        tuple of multiple batch tensors.
        """

        return None


class ClassificationDataModule(AbstractDefaultDataModule):
    """Datamodule for a standard classification setting with in-memory training instances and labels.
    """

    @abstractmethod
    def instances_and_labels(self, split: DatasetSplit) -> Iterable:
        """Get tuple of instances and labels for classification based on the requested split.

        :param split: the dataset split use.
        :return (tuple): instances and labels for a specific split.
        """

    def train_ds(self) -> Dataset:
        return TensorDataset(*self.instances_and_labels(DatasetSplit.TRAIN))

    def val_ds(self) -> Dataset:
        return TensorDataset(*self.instances_and_labels(DatasetSplit.VALIDATION))

    def test_ds(self) -> Dataset:
        return TensorDataset(*self.instances_and_labels(DatasetSplit.TEST))

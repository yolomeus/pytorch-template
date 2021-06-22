import gzip
import os
from typing import Optional
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from datamodule import DatasetSplit
from datamodule.default_datamodule import ClassificationDataModule


# noinspection PyAbstractClass
class FashionMNISTDataModule(ClassificationDataModule):
    """DataModule for the fashion MNIST dataset.
    """

    def __init__(self,
                 download: bool,
                 download_base_url: str,
                 data_dir: str,
                 train_img_file: str,
                 train_label_file: str,
                 test_img_file: str,
                 test_label_file: str,
                 val_size: float,
                 train_conf: DictConfig,
                 test_conf: DictConfig,
                 num_workers: int,
                 pin_memory: bool):
        """

        :param download: whether to download the dataset or not. Only downloads if files aren't  present already.
        :param download_base_url:  url where to download the files (without filename).
        :param data_dir: directory where to download files to and read from.
        :param train_img_file: file containing the training images.
        :param train_label_file: file containing the training labels.
        :param test_img_file: file containing the test images.
        :param test_label_file: file containing the test labels.
        :param val_size: percentage of the train set to use for validation e.g. 0.2 for 20%.
        :param train_conf: training configuration.
        :param test_conf: testing configuration
        :param num_workers: number of workers for DataLoaders to use.
        :param pin_memory: tell the DataLoaders whether to pin_memory or not.
        """
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

        self.download = download
        self.download_base_url = download_base_url
        self.data_dir = to_absolute_path(data_dir)

        self.train_img_file = train_img_file
        self.train_label_file = train_label_file
        self.test_img_file = test_img_file
        self.test_label_file = test_label_file

        self.val_size = val_size

        self.train_images = None
        self.train_labels = None

        self.val_images = None
        self.val_labels = None

        self.test_images = None
        self.test_labels = None

    def instances_and_labels(self, split: DatasetSplit):
        if split == DatasetSplit.TEST:
            return self.test_images, self.test_labels
        elif split == DatasetSplit.VALIDATION:
            return self.val_images, self.val_labels
        elif split == DatasetSplit.TRAIN:
            return self.train_images, self.train_labels
        raise NotImplementedError(f'Implementation for {split} does not exist.')

    def prepare_data(self, *args, **kwargs):
        if self.download:
            os.makedirs(self.data_dir, exist_ok=True)
            for filename in [self.train_img_file, self.test_img_file, self.train_label_file, self.test_label_file]:
                downloaded_file = os.path.join(self.data_dir, filename)
                if not os.path.exists(downloaded_file):
                    print(f'downloading {filename}...')
                    full_url = urljoin(self.download_base_url, filename)
                    urlretrieve(full_url, downloaded_file)

    def setup(self, stage: Optional[str] = None):
        def _data_dir(filename):
            return os.path.join(self.data_dir, filename)

        # val train split
        train_img_path = _data_dir(self.train_img_file)
        train_label_path = _data_dir(self.train_label_file)

        test_img_path = _data_dir(self.test_img_file)
        test_label_path = _data_dir(self.test_label_file)

        train_images, train_labels = self.load_mnist(train_img_path, train_label_path)
        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(
            train_images,
            train_labels,
            test_size=self.val_size
        )
        self.test_images, self.test_labels = self.load_mnist(test_img_path, test_label_path)

    @staticmethod
    def load_mnist(img_path, label_path):
        """Helper for loading the fashion mnist dataset from disk.

        :param img_path: path to image data.
        :param label_path: path to label data.
        :return: a tuple containing the images and labels tensors.
        """
        with gzip.open(label_path, 'rb') as lbl_path:
            labels = np.frombuffer(lbl_path.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(img_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        # create writeable copy
        images = np.array(images)
        labels = np.array(labels)
        return torch.as_tensor(images, dtype=torch.float) / 255.0, torch.as_tensor(labels, dtype=torch.long)

import gzip
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np
import torch
from hydra.utils import to_absolute_path
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from datamodule.default_datamodule import DefaultDataModule


# noinspection PyAbstractClass
class FashionMNISTDataModule(DefaultDataModule):
    """DataModule for the fashion MNIST dataset.
    """

    def __init__(self,
                 download,
                 download_base_path,
                 data_dir,
                 train_img_file,
                 train_label_file,
                 test_img_file,
                 test_label_file,
                 val_size,
                 train_conf,
                 test_conf,
                 num_workers):
        super().__init__(train_conf, test_conf, num_workers)

        self.download = download
        self.download_base_path = download_base_path
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

    @property
    def train_ds(self):
        return FashionMNIST(self.train_images, self.train_labels)

    @property
    def val_ds(self):
        return FashionMNIST(self.val_images, self.val_labels)

    @property
    def test_ds(self):
        return FashionMNIST(self.test_images, self.test_labels)

    def prepare_data(self, *args, **kwargs):
        if self.download:
            os.makedirs(self.data_dir, exist_ok=True)
            for filename in [self.train_img_file, self.test_img_file, self.train_label_file, self.test_label_file]:
                downloaded_file = os.path.join(self.data_dir, filename)
                if not os.path.exists(downloaded_file):
                    print(f'downloading {filename}...')
                    full_url = urljoin(self.download_base_path, filename)
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
        with gzip.open(label_path, 'rb') as lbl_path:
            labels = np.frombuffer(lbl_path.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(img_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        images = np.array(images)
        labels = np.array(labels)
        # create writeable copy
        return torch.as_tensor(images, dtype=torch.float) / 255.0, torch.as_tensor(labels, dtype=torch.long)


@dataclass
class FashionMNIST(Dataset):
    """Dataset for loading Fashion MNIST from disk.

    :param images: path to image dataset file (assumed to be in .gz format)
    :param labels: path to label file (assumed to be in .gz format)
    :param autoencoder_mode:  return inputs as labels if true.
    """

    images: Tensor
    labels: Tensor
    autoencoder_mode: bool = False

    def __getitem__(self, index):
        if self.autoencoder_mode:
            return self.images[index], self.images[index]
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

"""Training loop related utilities.
"""
from copy import deepcopy
from typing import List

from pytorch_lightning import LightningModule
from torch.nn import Module, ModuleList

from datamodule import DatasetSplit


class Metrics(Module):
    """Stores and manages metrics during training/testing for log.
    """

    def __init__(self, metrics: List[Module], to_probabilities: Module):
        """
        :param metrics: list of modules for computing metrics.
        :param to_probabilities: module for converting model outputs to probabilities
        """
        super().__init__()

        self._to_probabilities = to_probabilities
        # copy the metrics for each split, leave if empty.
        per_split_metrics = [[] if metrics is None else [deepcopy(metric) for metric in metrics]
                             for _ in range(3)]
        self.train_metrics, self.val_metrics, self.test_metrics = [ModuleList(metrics)
                                                                   for metrics in per_split_metrics]

    def forward(self, loop: LightningModule, y_pred, y_true, split: DatasetSplit):
        y_prob = self._to_probabilities(y_pred)

        if split == DatasetSplit.TRAIN:
            metrics = self.train_metrics
        elif split == DatasetSplit.TEST:
            metrics = self.test_metrics
        else:
            metrics = self.val_metrics

        for metric in metrics:
            metric.update(y_prob, y_true)
            loop.log(f'{split.value}/' + self.classname(metric),
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

    def metric_log(self, loop, y_pred, y_true, split: DatasetSplit):
        return self.forward(loop, y_pred, y_true, split)

    @staticmethod
    def classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name

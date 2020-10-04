from abc import ABC

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.nn import Module

from datamodule import DatasetSplit


class DefaultTraining(LightningModule, ABC):
    """Default Wrapper for training a pytorch module using pytorch-lightning.
    """

    def __init__(self, hparams: DictConfig):
        """
        :param hparams: contains all hyperparameters.
        """
        super().__init__()

        self.hparams = hparams
        self.loss = instantiate(hparams.loss)
        self.metrics = Metrics(self.loss, hparams.metrics)
        self.optimizer_cfg = hparams.optimizer
        self.model = instantiate(hparams.model)

    def configure_optimizers(self):
        return instantiate(self.optimizer_cfg, self.parameters())

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        logs = {'batch_loss': loss}
        return {'loss': loss, 'log': logs, 'y_pred': y_pred, 'y_true': y_true}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        return {'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, DatasetSplit.TRAIN)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, DatasetSplit.VALIDATION)

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, DatasetSplit.TEST)

    def _epoch_end(self, outputs, split: DatasetSplit):
        """Compute loss and all metrics at the end of an epoch.

        :param split: prefix for logs e.g. train, test, validation
        :param outputs: gathered outputs from *_epoch_end
        :return: a dict containing loss and metric logs.
        """
        logs = self.metrics.compute_logs(outputs, split)
        return {'log': logs}


class Metrics:
    """Stores and manages metrics during training/testing.
    """

    def __init__(self, loss: Module, metrics_config: DictConfig):
        self.metrics = [] if metrics_config is None else [instantiate(metric) for metric in metrics_config]
        self.loss = loss

    def compute_logs(self, outputs, split: DatasetSplit):
        """Compute a global log dict from multiple single step dicts.

        :param split: split for prefixing metric names in log dict.
        :param outputs: set of output dicts, each containing y_pred and y_true.
        :return: a dict mapping from metric names to their values.
        """
        y_pred, y_true = self._unpack_outputs('y_pred', outputs), self._unpack_outputs('y_true', outputs)

        logs = {f'{split.value}_' + self._classname(metric): metric(y_pred, y_true) for metric in self.metrics}
        loss = self.loss(y_pred, y_true)
        # when testing we want to log a scalar and not a tensor
        if split == DatasetSplit.TEST:
            loss = loss.item()
        logs[f'{split.value}_loss'] = loss

        return logs

    @staticmethod
    def _unpack_outputs(key, outputs):
        """Get the values of each output dict at key.

        :param key: key that gets the values from each output dict.
        :param outputs: a list of output dicts.
        :return: the concatenation of all output dict values at key.
        """

        outs_at_key = list(map(lambda x: x[key], outputs))
        # we assume a dict of outputs if the elements aren't tensors
        if isinstance(outs_at_key[0], dict):
            total_outs = {key: torch.cat([outs[key] for outs in outs_at_key])
                          for key in outs_at_key[0].keys()}

            return total_outs

        return torch.cat(outs_at_key)

    @staticmethod
    def _classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


# noinspection PyAbstractClass
class DefaultLightningModel(LightningModule):
    """Default pytorch-lightning model for models with a single loss and optimizer.
    """

    def __init__(self, loss: DictConfig, optimizer: DictConfig, hparams: DictConfig):
        """
        :param loss: configuration for the loss object.
        :param optimizer: configuration for the optimizer object.
        :param hparams: contains all hyperparameters.
        """
        super().__init__()

        self.save_hyperparameters(hparams)
        self.metrics = [] if hparams.metrics is None else [instantiate(metric) for metric in hparams.metrics]
        self.loss = instantiate(loss)
        self.optimizer_cfg = optimizer

    def forward(self, inputs):
        return self(inputs)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y_true)

        logs = {'batch_loss': loss}
        return {'loss': loss, 'log': logs, 'y_pred': y_pred, 'y_true': y_true}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        return {'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        return self._epoch_end('train', outputs)

    def validation_epoch_end(self, outputs):
        return self._epoch_end('val', outputs)

    def test_epoch_end(self, outputs):
        return self._epoch_end('test', outputs)

    def _epoch_end(self, prefix, outputs):
        """Compute loss and all metrics at the end of an epoch.

        :param prefix: prefix for logs e.g. train, test, validation
        :param outputs: gathered outputs from *_epoch_end
        :return: a dict containing loss and metric logs.
        """
        y_pred, y_true = self._unpack_outputs('y_pred', outputs), self._unpack_outputs('y_true', outputs)

        logs = {f'{prefix}_' + self._classname(metric): metric(y_pred, y_true) for metric in self.metrics}
        loss = self.loss(y_pred, y_true)
        # when testing we want to log a scalar and not a tensor
        if prefix == 'test':
            loss = loss.item()
        logs[f'{prefix}_loss'] = loss

        return {'log': logs}

    def configure_optimizers(self):
        return instantiate(self.optimizer_cfg, self.parameters())

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

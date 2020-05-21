import torch
from hydra.utils import instantiate
from ignite.metrics import Accuracy
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class LightningModel(LightningModule):
    """General model wrapper for training pytorch models using the pytorch-lightning library. This class is responsible
    for configuring the whole training and evaluation process.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = dict(hparams)

        self.model = instantiate(hparams.model)
        self.loss = instantiate(hparams.loss)
        self.optimizer = instantiate(hparams.optimizer, self.model.parameters())

        self.dataset_conf = hparams.dataset
        self.train_conf = hparams.training

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        return {'y_pred': y_pred, 'y_true': y_true}

    def validation_epoch_end(self, outputs):
        y_pred, y_true = zip(*map(lambda x: (x['y_pred'], x['y_true']), outputs))
        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)

        val_loss = self.loss(y_pred, y_true)
        logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': logs}

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        train_conf = self.train_conf
        train_ds = instantiate(self.dataset_conf.train)
        train_dl = DataLoader(train_ds,
                              train_conf.batch_size,
                              shuffle=True,
                              num_workers=train_conf.num_workers)
        return train_dl

    def val_dataloader(self):
        train_conf = self.train_conf
        train_ds = instantiate(self.dataset_conf.validation)
        val_dl = DataLoader(train_ds,
                            train_conf.batch_size,
                            num_workers=train_conf.num_workers)
        return val_dl

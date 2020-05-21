from hydra.utils import instantiate
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

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

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

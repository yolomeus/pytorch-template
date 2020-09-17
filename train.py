import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    model = instantiate(cfg.model, hparams=cfg)

    train_cfg = cfg.training

    ckpt_path = os.path.join(os.getcwd(), 'checkpoints/{epoch:03d}-{' + train_cfg.monitor + ':.3f}')
    model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                       monitor=train_cfg.monitor,
                                       mode=train_cfg.mode,
                                       verbose=True,
                                       filepath=ckpt_path)

    early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                   patience=train_cfg.patience,
                                   mode=train_cfg.mode,
                                   verbose=True)
    logger = WandbLogger()
    logger.experiment.watch(model)

    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      deterministic=True,
                      logger=logger,
                      checkpoint_callback=model_checkpoint,
                      early_stop_callback=early_stopping)

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    train()

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from lightning_wrapper import LightningModel


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    model = LightningModel(cfg)

    train_cfg = cfg.training
    model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                       monitor=train_cfg.monitor,
                                       mode=train_cfg.mode,
                                       verbose=True)

    early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                   patience=train_cfg.patience,
                                   mode=train_cfg.mode,
                                   verbose=True)

    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      deterministic=True,
                      checkpoint_callback=model_checkpoint,
                      early_stop_callback=early_stopping)

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    train()

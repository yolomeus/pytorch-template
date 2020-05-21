import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from lightning_wrapper import LightningModel


@hydra.main(config_path='conf/config.yaml')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""
    model = LightningModel(cfg)
    trainer = Trainer(max_epochs=cfg.training.epochs, gpus=cfg.gpus)
    trainer.fit(model)


if __name__ == '__main__':
    train()

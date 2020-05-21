import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from lightning_wrapper import LightningModel


@hydra.main(config_path='conf/config.yaml')
def train(cfg: DictConfig):
    model = LightningModel(cfg)
    trainer = Trainer(max_epochs=cfg.training.epochs)
    trainer.fit(model)


if __name__ == '__main__':
    train()

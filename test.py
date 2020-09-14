import os

import hydra
from hydra.utils import get_class, to_absolute_path, instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer


def test_checkpoint(ckpt_path, test_cfg, trainer, datamodule):
    """Load model state from checkpoint and test it.
    """

    model_cls = get_class(test_cfg.model._target_)
    model = model_cls.load_from_checkpoint(ckpt_path)
    # make sure we're using the current test config and not the saved one
    model.test_conf = test_cfg.testing
    trainer.test(model, datamodule=datamodule)


@hydra.main(config_path='conf', config_name='config')
def test(cfg: DictConfig):
    """Test a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers)
    trainer = Trainer(gpus=cfg.gpus, deterministic=True)

    ckpt_dir = to_absolute_path(cfg.testing.ckpt_dir)

    for i, file in enumerate(reversed(os.listdir(ckpt_dir))):
        if i == cfg.testing.test_best_k:
            break
        ckpt_path = os.path.join(ckpt_dir, file)
        test_checkpoint(ckpt_path, cfg, trainer, datamodule)


if __name__ == '__main__':
    test()

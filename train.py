import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer


@hydra.main(config_path='conf', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed, workers=True)

    model = instantiate(cfg.model)
    training_loop = instantiate(cfg.loop,
                                # hparams for saving
                                cfg,
                                model=model,
                                # pass model params to optimizer constructor
                                optimizer={"params": model.parameters()})

    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]

    if cfg.logger is not None:
        logger = instantiate(cfg.logger)
        if cfg.log_gradients:
            logger.experiment.watch(training_loop.model)
    else:
        # setting to True will use the default logger
        logger = True

    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    datamodule = instantiate(cfg.datamodule)

    trainer.fit(training_loop, datamodule=datamodule)

    # only look at this in the very end ;)
    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    train()

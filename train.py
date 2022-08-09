import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer

from logger.utils import Metrics


@hydra.main(config_path='conf', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed, workers=True)

    model = instantiate(cfg.model)
    metrics = Metrics(metrics=[instantiate(m) for m in cfg.metrics.values()],
                      to_probabilities=instantiate(cfg.to_probs))

    training_loop = instantiate(cfg.loop,
                                # hparams for saving
                                cfg,
                                model=model,
                                metrics=metrics,
                                # pass model params to optimizer constructor
                                optimizer={"params": model.parameters()})

    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]

    logger = instantiate(cfg.logger)
    # only applies to wandb logger
    if hasattr(logger, 'log_gradients') and logger.log_gradients:
        logger.watch_gradients(training_loop)

    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    datamodule = instantiate(cfg.datamodule)

    trainer.fit(training_loop, datamodule=datamodule)
    # only look at this in the very end ;)
    trainer.test(ckpt_path='best', datamodule=datamodule)

    # only applies to wandb logger
    if hasattr(logger, 'finish'):
        logger.finish()


if __name__ == '__main__':
    train()

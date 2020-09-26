from omegaconf import DictConfig
from torch.nn import Linear, ReLU, Sequential, Dropout

from model.default_lightning_model import DefaultLightningModel


class MLP(DefaultLightningModel):
    """Simple Multi-Layer Perceptron also known as Feed-Forward Neural Network."""

    def __init__(self,
                 in_dim: int,
                 h_dim: int,
                 out_dim: int,
                 dropout: float,
                 loss: DictConfig,
                 optimizer: DictConfig,
                 hparams: DictConfig):
        """

        :param in_dim: input dimension
        :param h_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout rate
        :param loss: config object representing loss.
        :param optimizer: config object representing optimizer.
        :param hparams: all hyperparameters.
        """

        super().__init__(loss, optimizer, hparams)
        self.classifier = Sequential(Dropout(dropout),
                                     Linear(in_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x

from omegaconf import DictConfig
from torch.nn import Linear, ReLU, Sequential, Dropout

from model.default_lightning_model import DefaultLightningModel


class MLP(DefaultLightningModel):
    """Simple Multi-Layer Perceptron also known as Feed Forward Neural Network."""

    def __init__(self, in_dim, h_dim, out_dim, dropout, loss: DictConfig, optimizer: DictConfig, hparams: DictConfig):
        """Builds the MLP

        :param in_dim: dimension of the input vectors.
        :param h_dim: hidden dimension.
        :param out_dim: output dimension.
        """

        super().__init__(loss, optimizer, hparams)
        self.classifier = Sequential(Dropout(dropout),
                                     Linear(in_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x

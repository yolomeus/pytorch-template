from abc import ABC, abstractmethod

import torch
from sklearn.metrics import accuracy_score
from torch import Tensor


class Metric(ABC):
    @abstractmethod
    def compute(self, y_pred: Tensor, y_true: Tensor):
        """Compute the metric given predictions and labels.

        :param y_pred: predicted scores
        :param y_true: ground truth labels
        :return: the computed metric value
        """

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        return self.compute(y_pred, y_true)


class Accuracy(Metric):
    """
    Compute accuracy using scikit-learn's `accuracy_score`.
    """

    def compute(self, y_pred: Tensor, y_true: Tensor):
        y_pred = torch.softmax(y_pred, dim=-1).argmax(dim=-1)
        return accuracy_score(y_true.cpu(), y_pred.cpu())

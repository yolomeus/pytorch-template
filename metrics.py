from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score


class Metric(ABC):
    @abstractmethod
    def compute(self, y_pred, y_true):
        """Compute the metric given predictions and labels.

        :param y_pred: predicted scores
        :param y_true: ground truth labels
        :return: the computed metric value
        """

    def __call__(self, y_pred, y_true):
        return self.compute(y_pred, y_true)


class Accuracy(Metric):
    def compute(self, y_pred, y_true):
        return accuracy_score(y_true, y_pred)

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class WandbMinMaxLogger(WandbLogger):
    """Extension of the WandbLogger that tracks minimum and maximum of all metrics over time.
    """

    LOGGER_JOIN_CHAR = '/'

    def __init__(self, postfix='', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._postfix = postfix

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Extends log_metrics with updating the wandb summary with maximum and minimum of each metric over time.

        :param metrics: mapping from metric names to values.
        :param step: step at which the metric was measured.
        """
        metrics = {k + self._postfix if k != 'epoch' else k: v
                   for k, v in metrics.items()}
        super().log_metrics(metrics, step)
        for name, value in metrics.items():
            min_name = self.min_name(name)
            max_name = self.max_name(name)
            self.experiment.summary[min_name] = min(self.experiment.summary.get(min_name, value),
                                                    value)
            self.experiment.summary[max_name] = max(self.experiment.summary.get(max_name, value),
                                                    value)

    def min_name(self, name):
        """Name for the minimum value of this metric.
        """
        return name + self.LOGGER_JOIN_CHAR + 'min'

    def max_name(self, name):
        """Name for the maximum value of this metric.
        """
        return name + self.LOGGER_JOIN_CHAR + 'max'

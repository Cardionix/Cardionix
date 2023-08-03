"""
Contains a data structure ``MetricsStorage`` for accumulating predictions (probabilities)
for a certain number of training iterations and a ``LightMetrics`` class
that inherits the ability to accumulate predictions
and calculate metrics for their subsequent logging.
"""

__all__ = ["LightMetrics"]

from typing import Union, Literal
import numpy as np
import torch
import torch.functional as F
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score


class MetricsStorage:
    """
    A data structure ``MetricsStorage`` for accumulating predictions
    for a certain number of training iterations
    """
    def __init__(self):
        self.__y_prob = np.array([])
        self.__y_true = np.array([])

    def __len__(self) -> int:
        return len(self.__y_prob)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.__y_prob[idx], self.__y_true[idx].ravel()

    def append(self, value: Union[list[np.ndarray | torch.Tensor] | tuple[np.ndarray | torch.Tensor]]):
        """
        The method concatenates model predictions (probabilities) and true class labels
        into two pre-existing arrays to accumulate class labels and model predictions on each call.
        """
        y_prob, y_true = self.check_sanity(value)
        if len(self) != 0:
            y_prob = np.concatenate([self.__y_prob, y_prob], axis=0)
            y_true = np.concatenate([self.__y_true, y_true], axis=0)
        self.__y_prob = y_prob
        self.__y_true = y_true

    def clear(self):
        """
        Clears all accumulated class labels and model predictions (probabilities),
        redefining arrays to store them.
        """
        self.__y_prob = np.array([])
        self.__y_true = np.array([])

    def check_sanity(self, value: tuple[np.ndarray | torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """
        Checking the type of incoming data.
        If it is np.ndarray, it will return unchanged.
        If it's a torch.Tensor, it will convert to an np.ndarray array and return it.
        If none of the conditions are met, it will return an error.
        """
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Expected value must be a "
                f"list or tuple, but got {type(value)}"
            )
        if not (isinstance(value[0], (np.ndarray, torch.Tensor)) and isinstance(value[1], (np.ndarray, torch.Tensor))):
            raise TypeError(
                f"Expected argument must contain "
                f"np.ndarray or torch.Tensor, but got "
                f"{type(value[0])} and {type(value[1])}"
            )
        if type(value[0]) != type(value[1]):
            raise TypeError(
                f"Expected argument must contain the same types, "
                f"but got {type(value[0])} and {type(value[1])}"
            )
        if isinstance(value[0], torch.Tensor) and isinstance(value[1], torch.Tensor):
            value = self.to_array(value)
        return value

    @staticmethod
    def to_array(value: list[torch.Tensor, torch.Tensor]) -> list[np.ndarray, np.ndarray]:
        """
        Returns a copy of the tensor on the CPU that will never be derived again,
        and then cast to the np.ndarray array that will eventually be returned.
        """
        return [
            tensor.detach().cpu().numpy()
            for tensor in value
        ]


class LightMetrics(MetricsStorage):
    """
    Inherits the ability to accumulate predictions
    and calculate metrics for their subsequent logging.

    Args:
        from_logites: (bool) if true, then logits will be expected,
            which first pass through the softmax activation function,
            and then the resulting probabilities and the corresponding class labels
            are accumulated for the subsequent calculation of metrics.
            (logits are the output of the last layer of the neural network,
            to which the softmax activation function was not applied).
            Otherwise, not logs are expected at the input,
            but probabilities that will be accumulated without any changes.

        beta: (float) beta parameter represents the ratio of recall importance to precision importance.
            beta > 1 gives more weight to recall, while beta < 1 favors precision.
            For example, beta = 2 makes recall twice as important as precision,
            while beta = 0.5 does the opposite.
            Asymptotically, beta -> +inf considers only recall, and beta -> 0 only precision.
            By default beta is 1.10.
    """
    def __init__(self,
                 from_logites: bool = True,
                 beta: float = 1.10,
                 ):

        super().__init__()
        self.classes = ("normal", "murmur", "extrahls", "extrastole", "artifact")
        self.beta = beta
        self.from_logites = from_logites

    def define_metrics(self,
                       stage: str,
                       roc_auc: float,
                       fb_score: float,
                       class_report: dict
                       ) -> dict:
        """
        This method takes as input a set of metrics, which are dictionaries.
        Each metric gets a new key, according to its logging stage (train, val, test).
        A dictionary is returned that contains all computed metrics with new keys.
        """
        report = {
            f"{stage}/fbeta_score": fb_score,
            f"{stage}/roc_auc": roc_auc
        }

        for key, item in class_report.items():
            if isinstance(item, dict):
                item.pop("support", None)
            if key.isdigit():
                key = self.classes[int(key)]
            report[f"{stage}/{key}"] = item
        return report

    def accumulate(self, y_prob: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        The method concatenates model predictions and true class labels
        into two pre-existing arrays to accumulate class labels and model predictions on each call.
        if from_logites = true, then logits will be expected,
        which first pass through the softmax activation function,
        and then the resulting probabilities and the corresponding class labels
        are accumulated for the subsequent calculation of metrics.
        (logits are the output of the last layer of the neural network,
        to which the softmax activation function was not applied).
        Otherwise, not logs are expected at the input,
        but probabilities that will be accumulated without any changes.

        Args:
            y_prob: (torch.Tensor) tezor with class probabilities or logits.
            y_true: (torch.Tensor) tezor with class labels.
        """
        if self.from_logites:
            y_prob = F.softmax(y_prob, dim=1)
        self.append([y_prob, y_true])

    def compute_metrics(self, stage: Literal["train", "val", "test"]) -> dict:
        """
        Calculates the metrics from the accumulated array of probabilities
        and the corresponding array with class labels for each set of probabilities.
        Values accumulated over N iterations are averaged
        """
        y_prob, y_true = self[:]
        y_pred = np.argmax(y_prob, axis=1)
        self.clear()

        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovo")
        fb_score = fbeta_score(y_true, y_pred, beta=self.beta, average="micro")
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics = self.define_metrics(stage, roc_auc, fb_score, class_report)
        return metrics

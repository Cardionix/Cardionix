"""
Contains a data structure ``ProbaStorage`` for accumulating predictions (probabilities)
for a certain number of training iterations and a ``CardioMetrics`` class
that inherits the ability to accumulate predictions
and calculate metrics for their subsequent logging.
"""

__all__ = ["CardioMetrics"]

from typing import Union, Literal, Optional
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score


class MetricsStorage:
    def __init__(self,
                 stage: Literal["train", "val", "test"],
                 metrics: Optional[Union[list[str, ...], tuple[str, ...]]] = None
                 ):

        self.stage = stage
        self.list_metrics = list(map(lambda name: f"{stage}/{name}", metrics))
        self.metrics = {}.fromkeys(self.list_metrics, [])

    def __len__(self) -> int:
        return sum(map(len, self.metrics.values()))

    def clear(self) -> None:
        self.metrics = {}.fromkeys(self.list_metrics, [])

    def average(self, axis: Optional[int] = 0) -> dict:
        return {key: np.mean(value, axis=axis) for key, value in self.metrics.items()}

    def append(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            key = f"{self.stage}/{key}"
            if key in self.metrics.keys():
                self.metrics[key].append(value)


class ProbaStorage:
    """
    A data structure ``MetricsStorage`` for accumulating predictions
    for a certain number of training iterations
    """

    def __init__(self):
        self.__y_prob = np.array([], dtype=np.float32)
        self.__y_true = np.array([], dtype=np.int8)

    def __len__(self) -> int:
        return len(self.__y_prob)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.__y_prob[idx], self.__y_true[idx].ravel()

    def _append(self, value: Union[list[np.ndarray | torch.Tensor] | tuple[np.ndarray | torch.Tensor]]):
        """
        The method concatenates model predictions (probabilities) and true class labels
        into two pre-existing arrays to accumulate class labels and model predictions on each call.
        """
        y_prob, y_true = self.check_sanity(value)
        if len(self) != 0:
            y_prob = np.concatenate([self.__y_prob, y_prob], axis=0, dtype=np.float32)
            y_true = np.concatenate([self.__y_true, y_true], axis=0, dtype=np.int8)
        self.__y_prob = y_prob
        self.__y_true = y_true

    def clear(self):
        """
        Clears all accumulated class labels and model predictions (probabilities),
        redefining arrays to store them.
        """
        self.__y_prob = np.array([], dtype=np.float32)
        self.__y_true = np.array([], dtype=np.int8)

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


class CardioMetrics(ProbaStorage):
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
                 stage: Literal["train", "val", "test"],
                 from_logites: bool = True,
                 beta: float = 1.10,
                 external_metrics: Optional[Union[list[str, ...], tuple[str, ...]]] = None,
                 ):

        super().__init__()
        self.classes = ("normal", "murmur", "extrahls", "extrastole", "artifact")
        self.beta = beta
        self.from_logites = from_logites
        self.stage = stage
        self.external_metrics = MetricsStorage(stage, external_metrics) if external_metrics else None

    def __define_metrics(self,
                         roc_auc: float,
                         fb_score: float,
                         class_report: dict,
                         ) -> dict:
        """
        This method takes as input a set of metrics, which are dictionaries.
        Each metric gets a new key, according to its logging stage (train, val, test).
        A dictionary is returned that contains all computed metrics with new keys.
        """

        report = {
            f"{self.stage}/fbeta_score": fb_score,
            f"{self.stage}/roc_auc": roc_auc
        }

        for key, item in class_report.items():
            if key.isdigit():
                key = self.classes[int(key)]
            if isinstance(item, dict):
                item.pop("support", None)
                for key1, value1 in item.items():
                    report[f"{self.stage}/{key}/{key1}"] = value1
            else:
                report[f"{self.stage}/{key}"] = item
        return report

    def add(self, **kwargs):
        if self.external_metrics != None:
            self.external_metrics.append(**kwargs)
        else:
            raise RuntimeError(
                "The external metrics argument was not passed "
                "during initialization due to which the storage "
                "for external metrics was not created"
            )

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
        self._append([y_prob, y_true])

    def compute_metrics(self) -> dict:
        """
        Calculates the metrics from the accumulated array of probabilities
        and the corresponding array with class labels for each set of probabilities.
        Values accumulated over N iterations are averaged
        """
        y_prob, y_true = self[:]
        y_pred = np.argmax(y_prob, axis=1).astype(dtype=np.int8)
        self.clear()

        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovo")
        fb_score = fbeta_score(y_true, y_pred, beta=self.beta, average="micro")
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics = self.__define_metrics(roc_auc, fb_score, class_report)

        if self.external_metrics != None:
            metrics.update(self.external_metrics.average())
            self.external_metrics.clear()
        return metrics

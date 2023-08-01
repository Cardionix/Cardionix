"""
Docstring
"""

__all__ = ["LightMetrics"]

from typing import Union, Literal
import numpy as np
import torch
import torch.functional as F
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score


class MetricsStorage:
    """
    Docstring
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
        Docstring
        """
        y_prob, y_true = self.check_sanity(value)
        if len(self) != 0:
            y_prob = np.concatenate([self.__y_prob, y_prob], axis=0)
            y_true = np.concatenate([self.__y_true, y_true], axis=0)
        self.__y_prob = y_prob
        self.__y_true = y_true

    def clear(self):
        """
        Docstring
        """
        self.__y_prob = np.array([])
        self.__y_true = np.array([])

    def check_sanity(self, value: tuple[np.ndarray | torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """
        Docstring
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
        Docstring
        """
        return [
            tensor.detach().cpu().numpy()
            for tensor in value
        ]


class LightMetrics(MetricsStorage):
    """
    Docstring
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
        Docstring
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
        Docstring
        """
        if self.from_logites:
            y_prob = F.softmax(y_prob, dim=1)
        self.append([y_prob, y_true])

    def compute_metrics(self, stage: Literal["train", "val", "test"]) -> dict:
        """
        Docstring
        """
        y_prob, y_true = self[:]
        y_pred = np.argmax(y_prob, axis=1)
        self.clear()

        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovo")
        fb_score = fbeta_score(y_true, y_pred, beta=self.beta, average="micro")
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics = self.define_metrics(stage, roc_auc, fb_score, class_report)
        return metrics

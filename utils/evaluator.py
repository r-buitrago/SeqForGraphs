from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict


def eval_to_wandb(eval_dict: Dict[str, Dict[str, float]], is_train: bool):
    """Logs the evaluation dictionary to wandb
    :param eval_dict: dictionary of evaluation values
    :param is_train: bool, whether the evaluation is for training or testing"""
    suffix = "train" if is_train else "test"
    log_dict = {}
    for key, val in eval_dict.items():
        log_dict[f"{key}_{suffix}"] = val
    return log_dict


def eval_to_print(eval_dict: Dict[str, Dict[str, float]], is_train: bool):
    """Prints the evaluation dictionary
    :param eval_dict: dictionary of evaluation values
    :param is_train: bool, whether the evaluation is for training or testing
    :param epoch: int, current epoch"""
    print_msg = ""
    suffix = "train" if is_train else "test"
    for key, val in eval_dict.items():
        print_msg += f" | {key}_{suffix} {val:.5f}"
    return print_msg


class Evaluator(ABC):
    def __init__(self, loss_fn, **kwargs):
        super(Evaluator, self).__init__()
        self.loss_fn = loss_fn
        self.n = 0
        self._init(**kwargs)

    @abstractmethod
    def _init(self, **kwargs):
        """Initializes the evaluation dictionary"""
        pass

    def reset(self):
        for key in self.eval_dict:
            self.eval_dict[key] = 0.0
        self.n = 0

    def evaluate(self, y_pred, y_true) -> Dict[str, Dict[str, float]]:
        self._evaluate(y_pred, y_true)
        self.n += y_pred.shape[0]

    @abstractmethod
    def _evaluate(self, y_pred, y_true):
        """Computes the loss
        :return: dictionary Dict[key: val]
        eg. {"Loss/loss": 0.0}}
        """
        raise NotImplementedError

    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        """Flushes the evaluation dictionary and returns the average of the values
        :return: dictionary of evaluation values
        """
        if self.n == 0:
            return {}
        avg_dict = {}
        for key, value in self.eval_dict.items():
            avg_dict[key] = value / self.n
        return avg_dict


class DummyEvaluator(Evaluator):
    def _init(self, **kwargs):
        """Initializes the evaluation dictionary"""
        self.eval_dict = {}

    def _evaluate(self, y_pred, y_true) -> Dict[str, Dict[str, float]]:
        """Does nothing"""
        pass


class LossEvaluator(Evaluator):
    def _init(self, **kwargs):
        self.eval_dict = {"Loss/loss": 0.0}

    def _evaluate(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        self.eval_dict["Loss/loss"] += loss.item()

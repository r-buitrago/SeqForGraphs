from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict
import torch
from sklearn.metrics import average_precision_score
import numpy as np
from collections import defaultdict


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
        self.eval_dict = defaultdict(float)

    def _evaluate(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        self.eval_dict["Loss/loss"] += loss.item()
        self.eval_dict["Loss/mae"] += torch.sum(torch.abs(y_pred - y_true)).item()

class LossMultiAccEvaluator(Evaluator):
    def _init(self, **kwargs):
        self.eval_dict = defaultdict(float)
        self.y_true_np = []
        self.y_pred_np = []

    def _evaluate(self, y_pred, y_true):
        # y_pred: (B, num_classes)
        # y_true: (B, num_classes)
        B, C = y_pred.shape

        # Compute loss
        loss = self.loss_fn(y_pred, y_true)
        self.eval_dict["Loss/loss"] += loss.item()

        # compute accuracy
        # acc = torch.sum(torch.argmax(y_pred, dim=1) == y_true).item()
        # self.eval_dict["Acc/acc"] += acc



        


        # Compute AP for each class
        # aps = []
        # for i in range(y_true_np.shape[1]):  # Loop over each class
        #     # Calculate AP using true labels and predicted probabilities for the class
        #     ap = average_precision_score(y_true_np[:, i], y_pred_np[:, i])
        #     aps.append(ap)
    
        # self.eval_dict["Acc/ap"] += np.mean(aps) * B

        # compute accuracy
        _, indices = torch.max(y_pred, dim=1)
        selected_y_true = torch.gather(y_true, 1, indices.unsqueeze(1)).squeeze(1)
        count = torch.sum(selected_y_true == 1)

        self.eval_dict["Acc/acc"] += count.item()

        # # # compute AP
        # y_pred = torch.sigmoid(y_pred)
        # y_pred = y_pred.detach().cpu().numpy()
        # y_true = y_true.detach().cpu().numpy()
        # AP = average_precision_score(y_true, y_pred, average="macro")
        # # we will divide by B at the end
        # self.eval_dict["Acc/ap"] += AP * B

        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = torch.sigmoid(y_pred).detach().cpu().numpy()  # Convert logits to probabilities
        self.y_true_np.append(y_true_np)
        self.y_pred_np.append(y_pred_np)
    
    def reset(self):
        super().reset()
        self.y_true_np = []
        self.y_pred_np = []
    
    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        if self.n == 0:
            return {}
        avg_dict = {}
        for key, value in self.eval_dict.items():
            avg_dict[key] = value / self.n
        
        # Compute AP
        y_true_np = np.concatenate(self.y_true_np, axis=0)
        y_pred_np = np.concatenate(self.y_pred_np, axis=0)
        AP = average_precision_score(y_true_np, y_pred_np, average="macro")
        avg_dict["Acc/ap"] = AP

        if reset:
            self.reset()
        return avg_dict




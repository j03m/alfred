import torch
import torchmetrics

from alfred.devices import set_device

from abc import ABC, abstractmethod

device = set_device()

# mimics a TorchMetric, maybe this isn't the best idea - thought about inheriting from torchmetrics.TorchMetric
# but wasn't sure if this was appropriate
class StatAccumulator(ABC):
    @abstractmethod
    def update(self, predictions:torch.Tensor, labels:torch.Tensor):
        pass

    def compute(self):
        pass

    def get(self):
        pass

class BCEAccumulator(StatAccumulator):
    def __init__(self):
        self._metric_accuracy = torchmetrics.Accuracy(task="binary").to(device)
        self._metric_precision = torchmetrics.Precision(task="binary").to(device)
        self._metric_recall = torchmetrics.Recall(task="binary").to(device)
        self._metric_f1 = torchmetrics.F1Score(task="binary").to(device)

        self._epoch_accuracy = None
        self._epoch_precision = None
        self._epoch_recall = None
        self._epoch_f1 = None
        self._computed_once = False

    def reset(self):
        self._metric_accuracy.reset()
        self._metric_precision.reset()
        self._metric_recall.reset()
        self._metric_f1.reset()

    def update(self, predictions:torch.Tensor, labels:torch.Tensor):
        predicted_classes = (predictions > 0.5).int()
        self._metric_accuracy.update(predicted_classes, labels.int())
        self._metric_precision.update(predicted_classes, labels.int())
        self._metric_recall.update(predicted_classes, labels.int())
        self._metric_f1.update(predicted_classes, labels.int())

    def compute(self):
        self._computed_once = True
        self._epoch_accuracy = self._metric_accuracy.compute().item()
        self._epoch_precision = self._metric_precision.compute().item()
        self._epoch_recall = self._metric_recall.compute().item()
        self._epoch_f1 = self._metric_f1.compute().item()
        return {
            'accuracy': self._epoch_accuracy,
            'precision': self._epoch_precision,
            'recall': self._epoch_recall,
            'f1': self._epoch_f1
        }

    def get(self):
        if not self._computed_once:
            raise Exception("Not ready, compute must be called once.")

        return {
            'accuracy': self._epoch_accuracy,
            'precision': self._epoch_precision,
            'recall': self._epoch_recall,
            'f1': self._epoch_f1
        }

    def print(self):
        print({
            'accuracy': self._epoch_accuracy,
            'precision': self._epoch_precision,
            'recall': self._epoch_recall,
            'f1': self._epoch_f1
        })


# grok generated this wooo :fire:
class MSEAccumulator(StatAccumulator):
    def __init__(self):
        self._metric_mse = torchmetrics.MeanSquaredError().to(device)
        self._squared_errors = []  # Store for variance calculation
        self._epoch_mse = None
        self._epoch_variance = None
        self._computed_once = False

    def reset(self):
        self._metric_mse.reset()
        self._squared_errors = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        self._metric_mse.update(predictions, labels)
        # Store squared errors for variance
        sq_errors = (predictions - labels) ** 2
        self._squared_errors.append(sq_errors.detach().cpu())  # Detach to avoid memory issues

    def compute(self):
        self._computed_once = True
        self._epoch_mse = self._metric_mse.compute().item()
        # Compute variance of squared errors
        all_sq_errors = torch.cat(self._squared_errors)
        self._epoch_variance = torch.var(all_sq_errors).item()
        self._squared_errors = []  # Reset to save memory
        return {
            'mse': self._epoch_mse,
            'error_variance': self._epoch_variance
        }

    def get(self):
        if not self._computed_once:
            raise Exception("Not ready, compute must be called once.")
        return {
            'mse': self._epoch_mse,
            'error_variance': self._epoch_variance
        }
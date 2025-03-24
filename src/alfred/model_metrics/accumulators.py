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
        self._metric_accuracy.update(predicted_classes.detach(), labels.detach().int())
        self._metric_precision.update(predicted_classes.detach(), labels.detach().int())
        self._metric_recall.update(predicted_classes.detach(), labels.detach().int())
        self._metric_f1.update(predicted_classes.detach(), labels.detach().int())

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


# grok initially generated this class with guidance on how to
# track direction and magnitude
class RegressionAccumulator:
    def __init__(self):
        # Standard regression metrics - mse which will already be
        # calculated with the loss function for this case
        self._metric_mae = torchmetrics.MeanAbsoluteError().to(device)
        self._metric_r2 = torchmetrics.R2Score().to(device)
        self._metric_corr = torchmetrics.PearsonCorrCoef().to(device)

        # Custom accumulators for directional metrics
        self._total_count = 0
        self._sign_correct = 0
        self._directional_error_sum = 0.0

        self._computed_once = False
        self._results = {}

    def reset(self):
        """Reset all metrics and accumulators."""
        self._metric_mae.reset()
        self._metric_r2.reset()
        self._metric_corr.reset()
        self._total_count = 0
        self._sign_correct = 0
        self._computed_once = False

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Update metrics with a batch of predictions and labels."""
        # Update standard metrics
        labels = labels.detach()
        self._metric_mae.update(predictions.detach(), labels)
        self._metric_r2.update(predictions.detach(), labels)
        self._metric_corr.update(predictions.detach(), labels)

        # Compute sign matches
        same_sign = (torch.sign(predictions) == torch.sign(labels)).float()

        # Update sign accuracy
        self._sign_correct += same_sign.sum().item()
        self._total_count += labels.size(0)

    def compute(self):
        """Compute all metrics after accumulating data."""
        self._computed_once = True
        self._results = {
            'mae': self._metric_mae.compute().item(),
            'r2': self._metric_r2.compute().item(),
            'pearson_corr': self._metric_corr.compute().item(),
            'sign_accuracy': self._sign_correct / self._total_count if self._total_count > 0 else 0,
        }
        return self._results

    def get(self):
        """Return computed results, raising an error if not yet computed."""
        if not self._computed_once:
            raise Exception("Not ready, compute must be called once.")
        return self._results

    def print(self):
        """Print computed results."""
        if not self._computed_once:
            raise Exception("Not ready, compute must be called once.")
        print(self._results)
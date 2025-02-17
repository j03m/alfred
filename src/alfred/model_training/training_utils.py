import torch
from torch import nn as nn

import torchmetrics

from alfred.model_persistence import maybe_save_model
from alfred.devices import set_device
from abc import ABC, abstractmethod

device = set_device()

# mimics a TorchMetric, maybe this isn't the best idea - thought about inheriting from torchmetrics.TorchMetric
# but wasn't sure if this was appropriate
class StatAccumulator(ABC):
    @abstractmethod
    def update(self, predictions:torch.Tensor, labels:torch.Tensor):
        pass

    def compute(self, length):
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

    def update(self, predictions:torch.Tensor, labels:torch.Tensor):
        predicted_classes = (predictions > 0.5).int()
        self._metric_accuracy.update(predicted_classes, labels.int())
        self._metric_precision.update(predicted_classes, labels.int())
        self._metric_recall.update(predicted_classes, labels.int())
        self._metric_f1.update(predicted_classes, labels.int())

    def compute(self, length):
        self._epoch_accuracy = self._metric_accuracy.compute()
        self._epoch_precision = self._metric_precision.compute()
        self._epoch_recall = self._metric_recall.compute()
        self._epoch_f1 = self._metric_f1.compute()
        return {
            'accuracy': self._epoch_accuracy,
            'precision': self._epoch_precision,
            'recall': self._epoch_recall,
            'f1': self._epoch_f1
        }

    def get(self):
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

def train_model(model, optimizer, scheduler, train_loader, patience, model_token, training_label, epochs=20,
                loss_function=nn.BCELoss(), stat_accumulator=BCEAccumulator(), verbose=False, verbosity_limit=100):
    model.train()

    patience_count = 0
    last_mean_loss = None
    for epoch in range(epochs):
        count = 0
        total_loss = 0.0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            batch_loss = loss_function(y_pred, labels)
            if torch.isnan(batch_loss):
                raise Exception("Found NaN!")

            batch_loss.backward()
            optimizer.step()
            loss_value = batch_loss.item()
            total_loss += loss_value
            count += 1
            stat_accumulator.update(y_pred.squeeze(), labels)


        mean_loss = total_loss / count
        if verbose and epoch % verbosity_limit == 0:
            print(f'Epoch {epoch} - patience {patience} - mean loss: {mean_loss} - Stats: ')
            print ("BCE: ", stat_accumulator.compute(length=count))
            print("last learning rate:", scheduler.get_last_lr())
            print(f"Predictions: {y_pred.detach().cpu().numpy().flatten()}")  # Flatten and print on one line
            print(f"Labels:      {labels.detach().cpu().numpy().flatten()}")  # Flatten and print on one line

        saved = maybe_save_model(model, optimizer, scheduler, mean_loss, model_token, training_label)

        if last_mean_loss is not None:
            if not saved:
                patience_count += 1
            else:
                patience_count = 0

        last_mean_loss = mean_loss

        if patience_count > patience:
            print(f'Out of patience at epoch {epoch}. Patience count: {patience}/{patience_count}. Limit: {patience}')
            return last_mean_loss

        scheduler.step(mean_loss)

    return last_mean_loss, stat_accumulator
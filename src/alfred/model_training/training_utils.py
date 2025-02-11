import torch
from torch import nn as nn
import numpy as np
from alfred.model_persistence import maybe_save_model
from alfred.devices import set_device

device = set_device()

def train_model(model, optimizer, scheduler, train_loader, patience, model_token, training_label, epochs=20,
                loss_function=nn.MSELoss()):
    model.train()

    patience_count = 0
    last_mean_loss = None
    last_nrmse = None
    last_max = None
    last_min = None
    for epoch in range(epochs):
        count = 0
        total_loss = 0.0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            batch_loss = loss_function(y_pred, labels)
            min_label = torch.min(labels)
            max_label = torch.max(labels)
            if last_min is None or min_label < last_min:
                last_min = min_label
            if last_max is None or max_label > last_max:
                last_max = max_label

            if torch.isnan(batch_loss):
                raise Exception("Found NaN!")

            batch_loss.backward()
            optimizer.step()
            loss_value = batch_loss.item()
            total_loss += loss_value
            count += 1

        loss_mean = total_loss / count

        print(f'Epoch {epoch} loss: {loss_mean}, training iter: {count}, patience: {patience_count}')
        saved = maybe_save_model(model, optimizer, scheduler, loss_mean, model_token, training_label)

        if last_mean_loss is not None:
            if not saved:
                patience_count += 1
            else:
                patience_count = 0

        last_mean_loss = loss_mean
        rmse = np.sqrt(last_mean_loss)
        range_y = last_max - last_min
        last_nrmse = rmse / range_y if range_y != 0 else 0

        if patience_count > patience:
            print(f'Out of patience at epoch {epoch}. Patience count: {patience_count}. Limit: {patience}')
            return last_mean_loss

        scheduler.step(loss_mean)

    return last_nrmse, last_mean_loss
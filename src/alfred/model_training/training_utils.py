from statistics import mean

import torch
from torch import nn as nn

from alfred.model_persistence import maybe_save_model
from alfred.devices import set_device

device = set_device()


def train_model(model, optimizer, scheduler, train_loader, patience, model_path, model_token, training_label, epochs=20,
                loss_function=nn.MSELoss()):
    model.train()

    single_loss = None
    patience_count = 0
    last_mean_loss = None
    for epoch in range(epochs):
        epoch_losses = []
        count = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            if torch.isnan(single_loss):
                raise Exception("Found NaN!")

            single_loss.backward()
            optimizer.step()
            loss_value = single_loss.item()
            epoch_losses.append(loss_value)
            count += 1

        loss_mean = mean(epoch_losses)

        print(f'Epoch {epoch} loss: {loss_mean}, training iter: {count}, patience: {patience_count}')
        # todo: maybe save model really needs to take the optimizer and scheduler as well if its going to resume at an optimzied state
        # otherwise we lose like a 100 epochs prior to it getting to the right place again
        saved = maybe_save_model(model, optimizer, scheduler, loss_mean, model_path, model_token, training_label)

        if last_mean_loss is not None:
            if not saved:
                patience_count += 1
            else:
                patience_count = 0
        last_mean_loss = loss_mean
        if patience_count > patience:
            print(f'Out of patience at epoch {epoch}. Patience count: {patience_count}. Limit: {patience}')
            return
        scheduler.step(loss_mean)

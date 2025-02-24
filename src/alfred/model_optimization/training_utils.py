import torch
from torch import nn as nn

from alfred.model_persistence import maybe_save_model, get_best_loss, prune_old_versions
from alfred.devices import set_device
from alfred.model_metrics import BCEAccumulator

device = set_device()

def train_model(model, optimizer, scheduler, scaler, train_loader, patience, model_token, training_label, epochs=20,
                loss_function=nn.BCELoss(), stat_accumulator=BCEAccumulator(), verbose=False, verbosity_limit=10):
    model.train()

    patience_count = 0
    last_mean_loss = None
    for epoch in range(epochs):
        count = 0
        total_loss = 0.0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()
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
            best_loss = get_best_loss(model_token, training_label)
            print(f'Epoch {epoch} - patience {patience_count}/{patience} - mean loss: {mean_loss} vs best loss: {best_loss} - Stats: ')
            print ("Stats: ", stat_accumulator.compute(length=count))
            print("last learning rate:", scheduler.get_last_lr())
            print(f"Predictions: {y_pred.detach().cpu().numpy().flatten()}")  # Flatten and print on one line
            print(f"Labels:      {labels.detach().cpu().numpy().flatten()}")  # Flatten and print on one line

        saved = maybe_save_model(model, optimizer, scheduler, scaler, mean_loss, model_token, training_label)

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
        if saved:
            prune_old_versions()

    return last_mean_loss, stat_accumulator
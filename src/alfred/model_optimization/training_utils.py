import torch
from torch import nn as nn
import time

from alfred.model_persistence import maybe_save_model, get_best_loss, prune_old_versions
from alfred.devices import set_device
from alfred.model_metrics import BCEAccumulator

device = set_device()

def train_model(model, optimizer, scheduler, scaler, train_loader, patience, model_token, training_label, epochs=20,
                loss_function=nn.BCELoss(), stat_accumulator=BCEAccumulator(), verbose=False, verbosity_limit=10):
    model.train()
    best_stats = None
    best_loss = None
    stats = None
    patience_count = 0
    last_mean_loss = None
    time_per_epoch = None
    for epoch in range(epochs):
        start_time = time.time()  # Record start time
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
        end_time = time.time()  # Record end time
        time_per_epoch = end_time - start_time  # Calculate time per epoch

        mean_loss = total_loss / count
        if verbose and epoch % verbosity_limit == 0:
            best_loss = get_best_loss(model_token, training_label)
            print(f'Epoch {epoch} (time: {time_per_epoch}) patience {patience_count}/{patience} -  loss: {mean_loss} vs best loss: {best_loss}')
            stats = stat_accumulator.compute()
            print ("Stats: ", stats)
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

        scheduler.step(mean_loss)
        if saved:
            prune_old_versions()
            best_stats = stats

        stat_accumulator.reset()

        if patience_count > patience:
            print(f'Out of patience at epoch {epoch}. Patience count: {patience}/{patience_count}. Limit: {patience}')
            break
    if best_stats is None:
        best_stats = stats

    print(f'Best loss: {best_loss}, Best Stats: {stats}')
    return get_best_loss(model_token, training_label), best_stats, time_per_epoch
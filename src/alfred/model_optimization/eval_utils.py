import pandas as pd

import torch
import torch.nn as nn

from alfred.devices import set_device
from alfred.model_metrics import BCEAccumulator

device = set_device()

def evaluate_model(model, loader, stat_accumulator=BCEAccumulator(),  loss_function=nn.BCELoss()):
    model.eval()
    count = 0
    total_loss = 0.0
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(seq).squeeze()
            batch_loss = loss_function(output, labels)

            loss_value = batch_loss.item()
            total_loss += loss_value
            count += 1
            if torch.isnan(batch_loss):
                raise Exception("Found NaN!")
            stat_accumulator.update(output.squeeze(), labels)

    stat_accumulator.compute()
    return total_loss/count, stat_accumulator.get()
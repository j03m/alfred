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

    stat_accumulator.compute(count)
    return total_loss/count, stat_accumulator.get()


def simple_profit_measure(predictions, actuals):
    ledger = []
    cumulative_profit_percentage = 0

    for i in range(len(predictions) - 1):
        predicted_price_next = predictions[i + 1]
        actual_price_current = actuals[i]
        actual_price_next = actuals[i + 1]

        trade_type = None
        predicted_profit = 0
        actual_profit = 0

        # Buy if the prediction is going up, sell/short if prediction is going down
        if predicted_price_next > actual_price_current:
            # Buy and settle on next day price
            trade_type = 'buy'
            predicted_profit = (predicted_price_next - actual_price_current) / actual_price_current * 100
            actual_profit = (actual_price_next - actual_price_current) / actual_price_current * 100
        elif predicted_price_next < actual_price_current:
            # Short sell and settle on next day price
            trade_type = 'short'
            predicted_profit = (actual_price_current - predicted_price_next) / actual_price_current * 100
            actual_profit = (actual_price_current - actual_price_next) / actual_price_current * 100

        cumulative_profit_percentage += actual_profit

        # Record the trade in the ledger
        ledger.append({
            'trade_type': trade_type,
            'actual_price_current': actual_price_current,
            'predicted_price_next': predicted_price_next,
            'actual_price_next': actual_price_next,
            'predicted_profit_percentage': predicted_profit,
            'actual_profit_percentage': actual_profit,
        })

    # Convert ledger to DataFrame
    ledger_df = pd.DataFrame(ledger)
    buy_hold_profit = (actuals[-1] - actuals[0]) / actuals[0] * 100
    return cumulative_profit_percentage, ledger_df, buy_hold_profit

def analyze_ledger(ledger_df):
    # Calculate important metrics
    total_trades = len(ledger_df)
    total_profit_percentage = ledger_df['actual_profit_percentage'].sum()
    win_rate = len(ledger_df[ledger_df['actual_profit_percentage'] > 0]) / total_trades if total_trades > 0 else 0

    metrics = {
        'total_trades': total_trades,
        'total_profit_percentage': total_profit_percentage,
        'win_rate': win_rate
    }

    return metrics
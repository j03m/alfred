import pandas as pd
import torch
from alfred.devices import set_device

# todo change this to a device manager singleton that things call into instead of gbls in each file :/

device = set_device()

def simple_profit_measure(predictions, actuals):
    ledger = []
    cumulative_profit_percentage = 0

    for i in range(len(predictions) - 1):
        predicted_price_current = predictions[i]
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

    return cumulative_profit_percentage, ledger_df

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


def evaluate_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(seq).squeeze(-1)
            predictions.extend(output.cpu().tolist())
            actuals.extend(labels.squeeze().cpu().tolist())

    return predictions, actuals

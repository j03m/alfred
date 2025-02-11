import numpy as np
import pandas as pd
import torch
from alfred.devices import set_device
from sklearn.metrics import mean_squared_error

# todo change this to a device manager singleton that things call into instead of gbls in each file :/

device = set_device()

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

def clamp_round(input, top=3, bottom=1):
    clamped_input = [max(bottom, min(top, x)) for x in input]
    return [round(x) for x in clamped_input]

def nrmse_by_range(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / range_y, mse

def sort_score(input, entries=3):
    return_me = []
    for i in range(0, len(input), entries):
        chunk = input[i:i + entries]
        sorted_ranks = np.argsort(chunk) + 1
        return_me.extend(sorted_ranks.tolist())
    return return_me

def evaluate_model(model, loader, prediction_squeeze=-1):
    model.eval()
    predictions = []
    actuals = []
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if prediction_squeeze is not None:
                output = model(seq).squeeze(prediction_squeeze)
            else:
                output = model(seq).squeeze()

            output_list = sort_score(output.cpu().tolist())
            labels = clamp_round(labels.squeeze().cpu().tolist())
            # print("************")
            # print(output_list)
            # print("============")
            # print(labels)
            # print("************")
            predictions.extend(output_list)
            actuals.extend(labels)

    return predictions, actuals

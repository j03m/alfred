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


def run_basic_signal_backtest(df, entries, model_readings):

    """
    This is a basic backtester for singals that are meant to come from a model but can arguably come from anything.
    The allowed signals are 1 (long) and 0 (short). Anything else will be ignored.

    The strategy will open or close a long/short on seeing a 1 or 0 and very simply reverse the source when the opposite
    is encountered or hold when the same is encountered.

    It is not meant to be a realistic trading environment and serves as a very simple way of sanity checking model accuracy
    against price movements in different environments.
    """

    assert len(df.index) == len(entries)
    assert len(entries) == len(model_readings)
    ledger = []
    current_operation = None # None, 'long', 'short'
    entry_price = None

    for (date, row), entry, reading in zip(df.iterrows(), entries, model_readings):
        close = row["Close"]
        if current_operation is None: # No open position
            if entry == 1: # Buy signal
                operation = 'BUY'
                entry_price = close
                current_operation = 'long'
                ledger.append({'Date': date, 'operation': operation, 'entry_price': entry_price, 'exit_price': None, 'profit': None})
                print(f"{date}, {operation} CREATE, Entry Price={entry_price}, Model Reading={reading}")
            elif entry == 0: # Sell signal (for short)
                operation = 'SELL_SHORT'
                entry_price = close
                current_operation = 'short'
                ledger.append({'Date': date, 'operation': operation, 'entry_price': entry_price, 'exit_price': None, 'profit': None})
                print(f"{date}, {operation} CREATE, Entry Price={entry_price} Model Reading={reading}")
        elif current_operation == 'short': # Currently short
            if entry == 1: # Buy signal (close short and go long)
                exit_price = close
                profit = entry_price - exit_price # Profit for short is entry - exit
                operation = 'CLOSE_SHORT_GO_LONG'
                ledger.append({'Date': date, 'operation': 'CLOSE_SHORT', 'entry_price': None, 'exit_price': exit_price, 'profit': profit})
                ledger.append({'Date': date, 'operation': 'BUY', 'entry_price': exit_price, 'exit_price': None, 'profit': None}) # New long entry
                current_operation = 'long'
                entry_price = exit_price # Entry price for new long position

                print(f"{date}, CLOSE_SHORT, Exit Price={exit_price}, Profit={profit:.2f}, Model Reading={reading}")
                print(f"{date}, BUY CREATE, Entry Price={entry_price} Model Reading={reading}")


            elif entry == 0: # Sell signal (hold short)
                print(f"{date}, HOLD SHORT, Close={close} Model Reading={reading}")


        elif current_operation == 'long': # Currently long
            if entry == 0: # Sell signal (close long and go short)
                exit_price = close
                profit = exit_price - entry_price # Profit for long is exit - entry
                ledger.append({'Date': date, 'operation': 'CLOSE_LONG', 'entry_price': None, 'exit_price': exit_price, 'profit': profit})
                ledger.append({'Date': date, 'operation': 'SELL_SHORT', 'entry_price': exit_price, 'exit_price': None, 'profit': None}) # New short entry
                current_operation = 'short'
                entry_price = exit_price # Entry price for new short position
                print(f"{date}, CLOSE_LONG, Exit Price={exit_price}, Profit={profit:.2f}, Model Reading={reading}")
                print(f"{date}, SELL_SHORT CREATE, Entry Price={entry_price}, Model Reading={reading}")

            elif entry == 1: # Buy signal (hold long)
                print(f"{date}, HOLD_LONG, Close={close}, Model Reading={reading}")

    ledger_df = pd.DataFrame(ledger)
    return ledger_df

def run_nuanced_signal_backtest(df, entries, model_readings):
    """
    This nuanced back test builds on what is in the basic backtest. But rather than only supporting 1 (long) vs
    0 (short) signals it builds some additional signals:

    2 - confident long
    1 - close short
    0 - unsure
   -1 - close long
   -2 - confident short

    In this manner, the system will NOT take a long position or a short position unless the confident score is encountered.
    It will close a long or a short if it encounters a closing level, but won't immediately open another position.

    A zero score will result in no action
    """

    assert len(df.index) == len(entries)
    ledger = []
    current_operation = None # None, 'long', 'short'
    entry_price = None

    for (date, row), entry, reading in zip(df.iterrows(), entries, model_readings):
        close = row["Close"]
        if current_operation is None: # No open position
            if entry == 2: # Confident long
                operation = 'BUY'
                entry_price = close
                current_operation = 'long'
                ledger.append({'Date': date, 'operation': operation, 'entry_price': entry_price, 'exit_price': None, 'profit': None})
                print(f"{date}, {operation} CREATE, Entry Price={entry_price}, Model Reading={reading}")
            elif entry == -2: # Confident short
                operation = 'SELL_SHORT'
                entry_price = close
                current_operation = 'short'
                ledger.append({'Date': date, 'operation': operation, 'entry_price': entry_price, 'exit_price': None, 'profit': None})
                print(f"{date}, {operation} CREATE, Entry Price={entry_price} Model Reading={reading}")
        elif current_operation == 'short': # Currently short
            if entry > 0: # Close the short
                exit_price = close
                profit = entry_price - exit_price # Profit for short is entry - exit
                ledger.append({'Date': date, 'operation': 'CLOSE_SHORT', 'entry_price': None, 'exit_price': exit_price, 'profit': profit})
                current_operation = None
                entry_price = exit_price # Entry price for new long position
                print(f"{date}, CLOSE_SHORT, Exit Price={exit_price}, Profit={profit:.2f}, Model Reading={reading}")



            elif entry <= 0: # Stay short
                print(f"{date}, HOLD SHORT, Close={close} Model Reading={reading}")

        elif current_operation == 'long': # Currently long
            if entry < 0: # Sell signal, we should get out of the trade
                exit_price = close
                profit = exit_price - entry_price # Profit for long is exit - entry
                ledger.append({'Date': date, 'operation': 'CLOSE_LONG', 'entry_price': None, 'exit_price': exit_price, 'profit': profit})
                ledger.append({'Date': date, 'operation': 'SELL_SHORT', 'entry_price': exit_price, 'exit_price': None, 'profit': None}) # New short entry
                current_operation = 'short'
                entry_price = exit_price # Entry price for new short position
                print(f"{date}, CLOSE_LONG, Exit Price={exit_price}, Profit={profit:.2f}, Model Reading={reading}")
                print(f"{date}, SELL_SHORT CREATE, Entry Price={entry_price}, Model Reading={reading}")

            elif entry == 1: # Buy signal (hold long)
                print(f"{date}, HOLD_LONG, Close={close}, Model Reading={reading}")

    ledger_df = pd.DataFrame(ledger)
    return ledger_df

def print_ledger_metrics(df, ledger):
    """
    Calculates and prints buy/hold profit and win/loss rate from a backtest ledger.

    Args:
        df (pd.DataFrame): Original DataFrame with price data (must have 'Close' column and Date index).
        ledger (pd.DataFrame): Ledger DataFrame generated by run_backtest function.
    """

    print("\n--- Ledger Metrics ---")

    # 1. Buy and Hold Profit Calculation
    first_date = df.index.min()
    last_date = df.index.max()
    first_close = df.loc[first_date]['Close']
    last_close = df.loc[last_date]['Close']
    buy_hold_profit = last_close - first_close

    print("\nBuy and Hold Strategy (First to Last Date):")
    print(f"  Start Date: {first_date.date()}, Close Price: {first_close:.2f}")
    print(f"  End Date: {last_date.date()}, Close Price: {last_close:.2f}")
    print(f"  Buy & Hold Profit (per unit): {buy_hold_profit:.2f}")

    # 2. Win/Loss Rate Calculation
    closed_trades = ledger[ledger['exit_price'].notna()] # Filter for closed trades (exit_price is not NaN)
    if not closed_trades.empty:
        profitable_trades = closed_trades[closed_trades['profit'] > 0]
        losing_trades = closed_trades[closed_trades['profit'] < 0]
        win_rate = len(profitable_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
        loss_rate = len(losing_trades) / len(closed_trades) if len(closed_trades) > 0 else 0

        print("\nWin/Loss Rate (from Ledger):")
        print(f"  Total Closed Trades: {len(closed_trades)}")
        print(f"  Profitable Trades: {len(profitable_trades)}")
        print(f"  Losing Trades: {len(losing_trades)}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Loss Rate: {loss_rate:.2%}")
        aggregate_profit = closed_trades['profit'].sum()
        print(f"\nAggregate Profit (from all closed trades): {aggregate_profit:.2f}")

    else:
        print("\nNo closed trades in ledger to calculate win/loss rate.")


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
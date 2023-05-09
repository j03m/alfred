import pandas as pd
import numpy as np
import yfinance as yf
import datetime


def calc_duration(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df_long = df[df['Side'] == 'long']
    df_short = df[df['Side'] == 'short']

    long_durations = (df_long[df_long['Action'] == 'exit']['Date'].reset_index(drop=True) -
                      df_long[df_long['Action'] == 'enter']['Date'].reset_index(drop=True)).dt.days
    short_durations = (df_short[df_short['Action'] == 'exit']['Date'].reset_index(drop=True) -
                       df_short[df_short['Action'] == 'enter']['Date'].reset_index(drop=True)).dt.days

    durations = pd.concat([long_durations, short_durations])
    return durations.min(), durations.max(), durations.mean(), durations.median(), durations.std()


def calc_total_return(df):
    initial_value = df.loc[0, 'Value']
    final_value = df.loc[df.index[-1], 'Value']
    return (final_value - initial_value) / initial_value


def calc_annualized_return(df):
    total_return = calc_total_return(df)
    start_date = df.loc[0, 'Date']
    end_date = df.loc[df.index[-1], 'Date']
    years = (end_date - start_date).days / 365.25
    return (1 + total_return) ** (1 / years) - 1


def calc_volatility(df):
    df['Daily_Return'] = df['Value'].pct_change()
    return df['Daily_Return'].std() * np.sqrt(252)


def calc_maximum_drawdown(df):
    df['Cum_Return'] = (1 + df['Value'].pct_change()).cumprod()
    df['Rolling_Max'] = df['Cum_Return'].cummax()
    df['Drawdown'] = df['Rolling_Max'] - df['Cum_Return']
    return df['Drawdown'].max()


def maximum_loss(df):
    initial_capital = df.iloc[0]['Value']
    losses = df[df['Value'] < initial_capital]['Value'] - initial_capital
    if losses.empty:
        max_loss = 0
    else:
        max_loss = losses.max()
    return max_loss


def calc_win_loss_ratio(df):
    wins = df[df['Profit_Actual'] > 0]['Profit_Actual']
    losses = df[df['Profit_Actual'] < 0]['Profit_Actual']
    loss_count = len(losses)
    if loss_count == 0:
        loss_count = 1

    return len(wins) / loss_count


# caches fetched stocks
stock_map = {}


def calc_performance_against_buy_and_hold(df, symbol, period):
    if symbol in stock_map:
        df_bh = stock_map[symbol]
    else:
        ticker_obj = yf.download(tickers=symbol, interval="1d")
        df_bh = pd.DataFrame(ticker_obj)
        df_bh = df_bh.tail(period)
        stock_map[symbol] = df_bh

    buy_hold_return = df_bh['Close'].iloc[-1] / df_bh['Close'].iloc[0] - 1
    return buy_hold_return


def calc_profit_loss_stats(df):
    profits = df[df['Profit_Actual'] > 0]['Profit_Actual']
    losses = df[df['Profit_Actual'] < 0]['Profit_Actual']

    profit_stats = profits.min(), profits.max(), profits.mean(), profits.median(), profits.std()
    loss_stats = losses.min(), losses.max(), losses.mean(), losses.median(), losses.std()

    return profit_stats, loss_stats

def calc_profit_loss_stats_percent(df):
    current_value = df.iloc[-1]['Value']
    profits_percent = df[df['Profit_Actual'] > 0]['Profit_Actual'] / current_value * 100
    losses_percent = df[df['Profit_Actual'] < 0]['Profit_Actual'] / current_value * 100

    profit_stats = profits_percent.min(), profits_percent.max(), profits_percent.mean(), profits_percent.median(), profits_percent.std()
    loss_stats = losses_percent.min(), losses_percent.max(), losses_percent.mean(), losses_percent.median(), losses_percent.std()

    return profit_stats, loss_stats

def analyze_trades(df, symbol, period):
    metrics = {'duration': calc_duration(df),
               'total_return': calc_total_return(df),
               'annualized_return': calc_annualized_return(df),
               'volatility': calc_volatility(df),
               'maximum_drawdown': calc_maximum_drawdown(df),
               'win_loss_ratio': calc_win_loss_ratio(df),
               'performance_against_buy_and_hold': calc_performance_against_buy_and_hold(df, symbol, period),
               'profit_stats': (calc_profit_loss_stats_percent(df))[0],
               'loss_stats': (calc_profit_loss_stats_percent(df))[1]}

    return metrics


def metrics_to_dataframe(metrics):
    data = {
        'test': [metrics['file']],
        'duration_min': [metrics['duration'][0]],
        'duration_max': [metrics['duration'][1]],
        'duration_mean': [metrics['duration'][2]],
        'duration_median': [metrics['duration'][3]],
        'duration_std': [metrics['duration'][4]],
        'total_return': [metrics['total_return']],
        'annualized_return': [metrics['annualized_return']],
        'volatility': [metrics['volatility']],
        'maximum_drawdown': [metrics['maximum_drawdown']],
        'win_loss_ratio': [metrics['win_loss_ratio']],
        'performance_against_buy_and_hold': [metrics['performance_against_buy_and_hold']],
        'profit_min': [metrics['profit_stats'][0]],
        'profit_max': [metrics['profit_stats'][1]],
        'profit_mean': [metrics['profit_stats'][2]],
        'profit_median': [metrics['profit_stats'][3]],
        'profit_std': [metrics['profit_stats'][4]],
        'loss_min': [metrics['loss_stats'][0]],
        'loss_max': [metrics['loss_stats'][1]],
        'loss_mean': [metrics['loss_stats'][2]],
        'loss_median': [metrics['loss_stats'][3]],
        'loss_std': [metrics['loss_stats'][4]]
    }
    return pd.DataFrame(data)

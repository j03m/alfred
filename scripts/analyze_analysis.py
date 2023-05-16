#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("./backtests/analysis_results.csv")

df['group'] = df['test'].str.extract(r'(h\d\.\d+_l\d\.\d+)')

grouped = df.groupby('group').agg({'duration_min': ['min', 'max', 'mean', 'median'],
                                   'duration_max': ['min', 'max', 'mean', 'median'],
                                   'duration_mean': ['min', 'max', 'mean', 'median'],
                                   'duration_median': ['min', 'max', 'mean', 'median'],
                                   'duration_std': ['min', 'max', 'mean', 'median'],
                                   'total_return': ['min', 'max', 'mean', 'median'],
                                   'buy_and_hold_performance': ['min', 'max', 'mean', 'median'],
                                   'volatility': ['min', 'max', 'mean', 'median'],
                                   'maximum_drawdown': ['min', 'max', 'mean', 'median'],
                                   'win_loss_ratio': ['min', 'max', 'mean', 'median'],
                                   'profit_min': ['min', 'max', 'mean', 'median'],
                                   'profit_max': ['min', 'max', 'mean', 'median'],
                                   'profit_mean': ['min', 'max', 'mean', 'median'],
                                   'profit_median': ['min', 'max', 'mean', 'median'],
                                   'profit_std': ['min', 'max', 'mean', 'median'],
                                   'loss_min': ['min', 'max', 'mean', 'median'],
                                   'loss_max': ['min', 'max', 'mean', 'median'],
                                   'loss_mean': ['min', 'max', 'mean', 'median'],
                                   'loss_median': ['min', 'max', 'mean', 'median'],
                                   'loss_std': ['min', 'max', 'mean', 'median']})

grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped.reset_index(inplace=True)
grouped['risk_free_rate'] = 0.05
grouped['sharpe_ratio'] = (grouped['total_return_mean'] - grouped['risk_free_rate']) / grouped['volatility_mean']
grouped['sortino_ratio'] = (grouped['total_return_mean'] - grouped['risk_free_rate']) / grouped['maximum_drawdown_mean']

grouped.sort_values('sharpe_ratio', ascending=False, inplace=True)

grouped.to_csv("./backtests/specs.csv")
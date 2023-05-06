import pytest
import json
import datetime
import pandas as pd
from machine_learning_finance import TraderEnv, make_env_for

test_data = "./tests/fixtures/SPY-for-test.csv"


def test_expand_index():
    env = make_env_for("SPY", 1, data_source="direct", path=test_data)
    print(type(env.orig_timeseries.index))
    assert isinstance(env.orig_timeseries.index, pd.DatetimeIndex)


def test_ledger_columns():
    env = make_env_for("SPY", 1, data_source="direct", path=test_data)
    env.step(1)
    env.step(2)
    env.step(1)
    expected_columns = ['Date', 'Side', 'Action', 'Product', 'Profit_Percent', 'Profit_Actual', 'Fee', 'Value', 'Price']
    assert isinstance(env.ledger, pd.DataFrame), "env.ledger should be a pandas DataFrame"
    assert set(env.ledger.columns) == set(expected_columns), f"env.ledger should have columns {expected_columns}"


def test_ledger_column_values_long_enter():
    env = make_env_for("SPY", 1, data_source="direct", path=test_data)
    env.step(1)
    df = pd.read_csv(test_data, parse_dates=["Date"], index_col=["Date"])
    # Assert that the values in env.ledger are correct
    assert env.ledger.iloc[0]["Date"] == df.index[0]
    assert env.ledger.iloc[0]["Product"] == "SPY"
    assert env.ledger.iloc[0]["Side"] == "long"
    assert env.ledger.iloc[0]["Action"] == "enter"


def test_ledger_column_values_multiple_steps():
    env = make_env_for("SPY", 1, data_source="direct", path=test_data)
    # go long (tested above)
    env.step(1)
    # close long, go short
    env.step(2)
    # close short, go long again
    env.step(1)
    df = pd.read_csv(test_data, parse_dates=["Date"], index_col=["Date"])
    print(env.ledger)

    # on 2nd step we close long and go short (time 1)
    assert env.ledger.iloc[1]["Date"] == df.index[1]
    assert env.ledger.iloc[1]["Product"] == "SPY"
    assert env.ledger.iloc[1]["Side"] == "long"
    assert env.ledger.iloc[1]["Action"] == "exit"

    assert env.ledger.iloc[2]["Date"] == df.index[1]
    assert env.ledger.iloc[2]["Product"] == "SPY"
    assert env.ledger.iloc[2]["Side"] == "short"
    assert env.ledger.iloc[2]["Action"] == "enter"

    # on 3rd step we close short and go long (time 2)
    assert env.ledger.iloc[3]["Date"] == df.index[2]
    assert env.ledger.iloc[3]["Product"] == "SPY"
    assert env.ledger.iloc[3]["Side"] == "short"
    assert env.ledger.iloc[3]["Action"] == "exit"

    assert env.ledger.iloc[4]["Date"] == df.index[2]
    assert env.ledger.iloc[4]["Product"] == "SPY"
    assert env.ledger.iloc[4]["Side"] == "long"
    assert env.ledger.iloc[4]["Action"] == "enter"

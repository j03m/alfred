import pandas as pd
from unittest.mock import MagicMock, patch
from machine_learning_finance import make_env_for, TraderEnv, DEFAULT_TEST_LENGTH
import math

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
    expected_columns = ['Date',
                        'Side', 'Action', 'Product',
                        'Profit_Percent', 'Profit_Actual', 'Fee',
                        'Value', 'Price',
                        'Shares', 'Cost']
    assert isinstance(env.ledger, pd.DataFrame), "env.ledger should be a pandas DataFrame"
    assert set(env.ledger.columns) == set(expected_columns), f"env.ledger should have columns {expected_columns}"


def test_timeseries_length():
    env = make_env_for("SPY", 1, tail=365, data_source="direct", path=test_data)
    assert(len(env.timeseries) == 365)

def test_ledger_column_values_long_enter():
    env = make_env_for("SPY", 1, data_source="direct", path=test_data)
    env.step(1)
    df = pd.read_csv(test_data, parse_dates=["Date"], index_col=["Date"])
    df = df.tail(DEFAULT_TEST_LENGTH)
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
    df = df.tail(DEFAULT_TEST_LENGTH)
    print("COMPARE: ", df)
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


def test_close_position_win():
    # Prepare the data
    data = {
        "Close": list(range(100, 100 + 180 * 5, 5)),
        "Volume": list(range(100, 100 + 180 * 5, 5))
    }
    df = pd.DataFrame(data)
    cash = 1000
    env = TraderEnv("SPY", df, df, cash=cash)
    prices = [100, 200]
    env.get_price_with_slippage = MagicMock(side_effect=prices)
    env._open_position(df, "SPY")

    # verify cash
    expected_cash = 97.5
    assert (env.cash == expected_cash)

    # verify position_shares - shares should be cash - fee / price floored
    expected_fee = (cash * env.fee)
    expected_shares = math.floor((cash - expected_fee) / prices[0])
    assert (env.position_shares == expected_shares)
    assert (env.ledger.iloc[0]["Fee"] == expected_fee)
    assert (env.ledger.iloc[0]["Profit_Percent"] == 0)
    assert (env.ledger.iloc[0]["Profit_Actual"] == 0)
    assert (env.ledger.iloc[0]["Price"] == prices[0])
    expected_value = expected_cash + (env.position_shares * prices[0])
    assert (env.ledger.iloc[0]["Value"] == expected_value)

    env._close_position(df, "SPY")
    expected_fee = (expected_shares * prices[1]) * env.fee
    expected_value = (expected_cash + expected_shares * prices[1]) - expected_fee
    expected_percent = ((prices[1] - prices[0]) / prices[0]) * 100

    assert (env.cash == expected_value)
    assert (env.position_shares == 0)
    assert (env.ledger.iloc[1]["Fee"] == expected_fee)
    assert (env.ledger.iloc[1]["Profit_Percent"] == expected_percent)

    # after fees
    assert (env.ledger.iloc[1]["Profit_Actual"] == 895.5)
    assert (env.ledger.iloc[1]["Price"] == prices[1])
    assert (env.ledger.iloc[1]["Value"] == env.cash)


def test_close_position_loss():
    # Prepare the data
    data = {
        "Close": list(range(100, 100 + 180 * 5, 5)),
        "Volume": list(range(100, 100 + 180 * 5, 5))
    }
    df = pd.DataFrame(data)
    cash = 1000
    env = TraderEnv("SPY", df, df, cash=cash)
    prices = [100, 50]
    env.get_price_with_slippage = MagicMock(side_effect=prices)
    env._open_position(df, "SPY")

    # verify cash
    expected_cash = 97.5
    assert (env.cash == expected_cash)

    # verify position_shares - shares should be cash - fee / price floored
    expected_fee = (cash * env.fee)
    expected_shares = math.floor((cash - expected_fee) / prices[0])
    assert (env.position_shares == expected_shares)
    assert (env.ledger.iloc[0]["Fee"] == expected_fee)
    assert (env.ledger.iloc[0]["Profit_Percent"] == 0)
    assert (env.ledger.iloc[0]["Profit_Actual"] == 0)
    assert (env.ledger.iloc[0]["Price"] == prices[0])
    expected_value = expected_cash + (env.position_shares * prices[0])
    assert (env.ledger.iloc[0]["Value"] == expected_value)

    env._close_position(df, "SPY")
    expected_fee = (expected_shares * prices[1]) * env.fee
    expected_value = (expected_cash + expected_shares * prices[1]) - expected_fee
    expected_percent = ((prices[1] - prices[0]) / prices[0]) * 100

    assert (env.cash == expected_value)
    assert (env.position_shares == 0)
    assert (env.ledger.iloc[1]["Fee"] == expected_fee)
    assert (env.ledger.iloc[1]["Profit_Percent"] == expected_percent)

    # after fees
    assert (env.ledger.iloc[1]["Profit_Actual"] == -451.125)
    assert (env.ledger.iloc[1]["Price"] == prices[1])
    assert (env.ledger.iloc[1]["Value"] == env.cash)


def test_close_short_win():
    # Prepare the data
    data = {
        "Close": list(range(100, 100 + 180 * 5, 5)),
        "Volume": list(range(100, 100 + 180 * 5, 5))
    }
    df = pd.DataFrame(data)
    cash = 1000
    env = TraderEnv("SPY", df, df, cash=cash)
    prices = [100, 50]
    env.get_price_with_slippage = MagicMock(side_effect=prices)
    env._open_short(df, "SPY")

    # verify cash
    expected_fee = (cash * env.fee)
    expected_cash = cash - expected_fee
    expected_shares = math.floor(expected_cash / prices[0])
    expected_cash += expected_shares * prices[0]
    assert (env.cash == expected_cash)

    # verify position_shares - shares should be cash - fee / price floored
    assert (env.position_shares == 0)
    assert (env.shares_owed == expected_shares)
    assert (env.ledger.iloc[0]["Fee"] == expected_fee)
    assert (env.ledger.iloc[0]["Profit_Percent"] == 0)
    assert (env.ledger.iloc[0]["Profit_Actual"] == 0)
    assert (env.ledger.iloc[0]["Price"] == prices[0])

    expected_value = expected_cash - (expected_shares * prices[0])
    assert (env.ledger.iloc[0]["Value"] == expected_value)

    env._close_short(df, "SPY")
    value = prices[1] * expected_shares
    fee = value * env.fee
    cost = value + fee
    expected_cash -= cost
    expected_percent = ((prices[0] - prices[1]) / prices[0]) * 100

    assert (env.cash == expected_cash)
    assert (env.position_shares == 0)
    assert (env.shares_owed == 0)
    assert (env.ledger.iloc[1]["Fee"] == fee)
    assert (env.ledger.iloc[1]["Profit_Percent"] == expected_percent)

    # after fees
    print(env.ledger)
    assert (env.ledger.iloc[1]["Profit_Actual"] == 448.875)
    assert (env.ledger.iloc[1]["Price"] == prices[1])
    assert (env.ledger.iloc[1]["Value"] == env.cash)

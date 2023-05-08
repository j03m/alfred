from machine_learning_finance import InverseEnv, make_inverse_env_for
import pandas as pd
import math
from unittest.mock import MagicMock, patch

main_data = "./tests/fixtures/SPY-for-test.csv"
inverse_data = "./tests/fixtures/SPY-for-test.csv"

main_fake_price = "./tests/fixtures/spy-control-price.csv"
inverse_fake_price = "./tests/fixtures/spy-control-price.csv"


def test_inverse_actions():
    env = make_inverse_env_for("SPY", "SH", 1, data_source="direct", paths=[
        main_data, inverse_data
    ])
    env.open_inverse()
    env.close_inverse()
    assert (env.ledger.iloc[0]["Product"] == "SH")
    assert (env.ledger.iloc[0]["Side"] == "long")
    assert (env.ledger.iloc[0]["Action"] == "enter")

    assert (env.ledger.iloc[1]["Product"] == "SH")
    assert (env.ledger.iloc[1]["Side"] == "long")
    assert (env.ledger.iloc[1]["Action"] == "exit")


def test_ledger_column_values_multiple_steps():
    env = make_inverse_env_for("SPY", "SH", 1, data_source="direct", paths=[
        main_data, inverse_data
    ])

    # go long
    env.step(1)
    # close long, go short
    env.step(2)
    # close short, go long again
    env.step(1)
    df = pd.read_csv(main_data, parse_dates=["Date"], index_col=["Date"])

    # 1st step we went long
    assert env.ledger.iloc[0]["Date"] == df.index[0]
    assert env.ledger.iloc[0]["Product"] == "SPY"
    assert env.ledger.iloc[0]["Side"] == "long"
    assert env.ledger.iloc[0]["Action"] == "enter"

    # 2nd step went short, so we exit the long and enter an inverse long
    assert env.ledger.iloc[1]["Date"] == df.index[1]
    assert env.ledger.iloc[1]["Product"] == "SPY"
    assert env.ledger.iloc[1]["Side"] == "long"
    assert env.ledger.iloc[1]["Action"] == "exit"

    assert env.ledger.iloc[2]["Date"] == df.index[1]
    assert env.ledger.iloc[2]["Product"] == "SH"
    assert env.ledger.iloc[2]["Side"] == "long"
    assert env.ledger.iloc[2]["Action"] == "enter"

    # 3rd step close the inverse and reopen a long
    assert env.ledger.iloc[3]["Date"] == df.index[2]
    assert env.ledger.iloc[3]["Product"] == "SH"
    assert env.ledger.iloc[3]["Side"] == "long"
    assert env.ledger.iloc[3]["Action"] == "exit"

    assert env.ledger.iloc[4]["Date"] == df.index[2]
    assert env.ledger.iloc[4]["Product"] == "SPY"
    assert env.ledger.iloc[4]["Side"] == "long"
    assert env.ledger.iloc[4]["Action"] == "enter"


def test_close_inverse_win():
    # Prepare the data
    data = {
        "Close": list(range(100, 100 + 180 * 5, 5)),
        "Volume": list(range(100, 100 + 180 * 5, 5))
    }
    df = pd.DataFrame(data)
    cash = 1000
    env = InverseEnv("SPY", df, "SH", df, 1, cash=cash)
    prices = [100, 200]
    env.get_price_with_slippage = MagicMock(side_effect=prices)
    env.open_inverse()

    # verify cash
    expected_cash = 95
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

    env.close_inverse()
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


def test_close_short_loss():
    # Prepare the data
    data = {
        "Close": list(range(100, 100 + 180 * 5, 5)),
        "Volume": list(range(100, 100 + 180 * 5, 5))
    }
    df = pd.DataFrame(data)
    cash = 1000
    env = InverseEnv("SPY", df, "SH", df, 1, cash=cash)
    prices = [200, 100]
    env.get_price_with_slippage = MagicMock(side_effect=prices)
    env.get_current_close = MagicMock(side_effect=prices)
    env.open_inverse()

    # verify cash
    expected_cash = 195
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

    env.close_inverse()
    expected_fee = (expected_shares * prices[1]) * env.fee
    expected_value = (expected_cash + expected_shares * prices[1]) - expected_fee
    expected_percent = ((prices[1] - prices[0]) / prices[0]) * 100

    assert (env.cash == expected_value)
    assert (env.position_shares == 0)
    assert (env.ledger.iloc[1]["Fee"] == expected_fee)
    assert (env.ledger.iloc[1]["Profit_Percent"] == expected_percent)

    # after fees
    assert (env.ledger.iloc[1]["Profit_Actual"] == -401.0)
    assert (env.ledger.iloc[1]["Price"] == prices[1])
    assert (env.ledger.iloc[1]["Value"] == env.cash)


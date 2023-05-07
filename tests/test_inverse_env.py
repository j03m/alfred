from machine_learning_finance import InverseEnv, make_inverse_env_for
import pandas as pd

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
    assert(env.ledger.iloc[0]["Product"] == "SH")
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

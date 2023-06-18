import pytest
import pandas as pd
import numpy as np
from machine_learning_finance import SimpleEnv

known_prices = [100, 110, 90, 80, 100]

# A fixture to create a simple environment
@pytest.fixture
def simple_env():
    # Creating static test and historical data
    static_test_data = pd.DataFrame({
        'Close': known_prices * 2,
        'Open': [100, 110, 120, 130, 140] * 2,
        'High': [105, 115, 125, 135, 145] * 2,
        'Low': [95, 105, 115, 125, 135] * 2,
        'Volume': [1000, 2000, 3000, 4000, 5000] * 2
    })
    static_historical_data = pd.DataFrame({
        'Close': [90, 100, 110, 120, 130] * 2,
        'Open': [90, 100, 110, 120, 130] * 2,
        'High': [95, 105, 115, 125, 135] * 2,
        'Low': [85, 95, 105, 115, 125] * 2,
        'Volume': [900, 1000, 2000, 3000, 4000] * 2
    })

    # Creating random test and historical data
    random_test_data = pd.DataFrame(np.random.rand(180, 5), columns=['Close', 'Open', 'High', 'Low', 'Volume'])
    random_historical_data = pd.DataFrame(np.random.rand(720, 5), columns=['Close', 'Open', 'High', 'Low', 'Volume'])

    # Concatenating static and random data
    test_period_df = pd.concat([static_test_data, random_test_data], ignore_index=True)
    historical_period_df = pd.concat([static_historical_data, random_historical_data], ignore_index=True)

    return SimpleEnv(test_period_df=test_period_df, historical_period_df=historical_period_df)



def test_reset(simple_env):
    obs, _ = simple_env.reset()
    assert simple_env.index == 0
    assert simple_env.state == -1
    assert obs.all() == simple_env.expanded_df.iloc[0].values.all()


def test_step(simple_env):
    simple_env.reset()

    obs, reward, done, _, _ = simple_env.step(0)

    # Check if step is updated
    assert simple_env.index == 1
    assert np.array_equal(obs, simple_env.expanded_df.iloc[1].values)

    obs, reward, done, _, _ = simple_env.step(1)
    assert simple_env.index == 2
    assert np.array_equal(obs, simple_env.expanded_df.iloc[2].values)

    # Repeat for multiple steps


def test_done(simple_env):
    simple_env.reset()
    for _ in range(len(simple_env.test_period_df) - 1):
        obs, reward, done, _, _ = simple_env.step(0)
    assert not done

    obs, reward, done, _, _ = simple_env.step(0)
    assert done  # The episode should be done after stepping through all the data points


def test_profit(simple_env):
    simple_env.reset()
    simple_env.cash = 10000

    # this should give us a buy, price should now be 100 and action shou\
    # recall our initial data is: [100, 110, 90, 80, 100]
    simple_env.step(0)
    assert simple_env.price == known_prices[0]
    assert simple_env.state == 0

    # now on our next step lets sell the buy
    simple_env.step(1)
    assert simple_env.price == known_prices[1]
    assert simple_env.state == 1

    # we should now have some profit
    first_profit = simple_env.cash * (10/known_prices[0])
    assert simple_env.profit == first_profit

    # now lets sell the short
    simple_env.step(0)
    assert simple_env.price == 90
    assert simple_env.state == 0

    next_profit = simple_env.cash * (20/110)
    assert simple_env.profit == first_profit + next_profit

    # now we're long 90, lets lose money on the next step
    simple_env.step(1)
    assert simple_env.price == 80
    assert simple_env.state == 1
    first_loss = simple_env.cash * (10 / 90) * -1
    assert simple_env.profit == first_profit + next_profit + first_loss

    # now we're short 80, lets loose money here
    simple_env.step(0)
    assert simple_env.price == 100
    assert simple_env.state == 0
    next_loss = simple_env.cash * (20 / 80) * -1
    assert simple_env.profit == first_profit + next_profit + first_loss + next_loss




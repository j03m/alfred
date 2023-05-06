import pytest
import json
import datetime
import pandas as pd
from machine_learning_finance import TraderEnv, make_env_for

def test_expand_columns():
    env = make_env_for("SPY", 1, data_source="file")
    assert isinstance(env.orig_timeseries.index, pd.DatetimeIndex)

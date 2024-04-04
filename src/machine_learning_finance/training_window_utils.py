import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

'''
GPT4 generated this for us (with some guidance)
I would like you to write three classes, that are similar to this function. They are:

TailTrainingWindowUtil, RangeTrainingWindowUtil and RandomTrainingWindowUtil

TailTrainingWindowUtil should accept in its constuctor a dataframe a test data size and a historical data size. 
It should then separate out test + historical dataframes in the manner above where historical precedes test at the 
specified lengths. Keep parameters and results, including any automatically determined dates and size as read only 
properties on the class. That idea will apply to all classes.

RangeTrainingWindowUtil will receive a start and end date. It should optionally also accept a historical length
multiplier which defaults to 4. Start and End will represent the portion of the time series that will be under test. 
The class should then automatically determine the start and end date for historical period which will be 
end - start * multiplier in length. 

RandomTrainingWindowUtil will just receive a dataframe and a test set size. It will then randomly select a range in 
the dataframe that fits the size requested. It will then also generate a test + historical frame and capture the dates 
Tused as read only properties. 

Where possible, use a base class or utility functions to avoid repetition of code. 
'''


class BaseTrainingWindowUtil:
    def __init__(self, df):
        self._df = df

    @property
    def df(self):
        return self._df

    @property
    def full_hist_df(self):
        raise NotImplementedError()

    @property
    def test_df(self):
        raise NotImplementedError()


class TailTrainingWindowUtil(BaseTrainingWindowUtil):
    def __init__(self, df, test_size):
        super().__init__(df)
        assert test_size <= len(df), "Test size + historical size is larger than the dataframe."
        self._test_size = test_size
        self._full_hist_size = len(df)
        self._test_df = self.df.tail(test_size)
        historical_end = self._test_df.index[0] - pd.Timedelta(days=1)
        self._full_hist_df = df.loc[:historical_end]


    @property
    def full_hist_size(self):
        return self._full_hist_size

    @property
    def test_size(self):
        return self._test_size

    @property
    def full_hist_df(self):
        return self._full_hist_df

    @property
    def test_df(self):
        return self._test_df


class RangeTrainingWindowUtil(BaseTrainingWindowUtil):
    def __init__(self, df, start, end):
        super().__init__(df)
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        historical_end = self._start - pd.Timedelta(days=1)
        assert self._start < self._end, "Start date should be before end date."
        self._full_hist_df = self.df.loc[:historical_end]
        self._test_df = self.df.loc[self._start:self._end]
        assert(len(self.full_hist_df) != 0)
        assert (len(self._test_df) != 0)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def full_hist_df(self):
        return self._full_hist_df

    @property
    def test_df(self):
        return self._test_df


class RandomTrainingWindowUtil(BaseTrainingWindowUtil):
    def __init__(self, df, test_size):
        super().__init__(df)
        self._test_size = test_size
        if len(df) - test_size < 0:
            assert(f"Invalid length: {len(df) - test_size}")
        self._random_start = np.random.randint(0, len(df) - test_size)
        historical_end = self._random_start + test_size
        self._full_hist_df = df
        self._test_df = self.df.iloc[self._random_start:historical_end]

    @property
    def test_size(self):
        return self._test_size

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def full_hist_df(self):
        return self._full_hist_df

    @property
    def test_df(self):
        return self._test_df

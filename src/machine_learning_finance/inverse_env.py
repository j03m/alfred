import yfinance as yf
from .logger import info, debug, error, verbose
import pandas as pd
from .trader_env import TraderEnv
from .defaults import DEFAULT_TEST_LENGTH, \
    DEFAULT_HISTORICAL_MULT, \
    DEFAULT_CASH, \
    DEFAULT_TOP_PERCENT, \
    DEFAULT_BOTTOM_PERCENT

def read_df_from_file(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def make_inverse_env_for(symbol,
                         inverse_symbol,
                         code,
                         tail=DEFAULT_TEST_LENGTH,
                         data_source="yahoo",
                         paths=[None, None],
                         cash=DEFAULT_CASH,
                         prob_high=DEFAULT_TOP_PERCENT,
                         prob_low=DEFAULT_BOTTOM_PERCENT,
                         hist_tail=None):
    if hist_tail is None:
        hist_tail = tail * DEFAULT_HISTORICAL_MULT
    if data_source == "yahoo":
        ticker_obj = yf.download(tickers=symbol)
        df = pd.DataFrame(ticker_obj)
        ticker_obj = yf.download(tickers=inverse_symbol)
        inverse_df = pd.DataFrame(ticker_obj)
    elif data_source == "file":
        df = read_df_from_file(f"./data/{symbol}.csv")
        inverse_df = read_df_from_file(f"./data/{inverse_symbol}.csv")
    elif data_source == "direct":
        df = read_df_from_file(paths[0])
        inverse_df = read_df_from_file(paths[1])
    else:
        raise Exception("Implement me")

    hist_df = df.head(len(df)-tail)
    hist_df = hist_df.tail(hist_tail)

    df = df.tail(tail)
    inverse_df = inverse_df.tail(tail)

    env = InverseEnv(symbol,
                     inverse_symbol,
                     df,
                     hist_df,
                     inverse_df,
                     code,
                     cash,
                     prob_high,
                     prob_low)
    return env


class InverseEnv(TraderEnv):

    def __init__(self, product, inverse_product, test_df, hist_df, inverse_df, code, cash=5000,
                 prob_high=0.8, prob_low=0.2):
        super(InverseEnv, self).__init__(product, test_df, hist_df, code, cash, prob_high, prob_low)
        self.inverse_product = inverse_product
        self.inverse_df = inverse_df
        self.status = 0  # 1 long 0 none 2 short

    def _apply_action(self, action):
        # AI says hold
        if action == 0:
            info("**holding")
            pass
        elif action == 1 and self.status == 1:
            info("**holding long")
            pass
        elif action == 1 and self.status == 0:
            info("**opening long.")
            self.open_position()
        # AI says long, but we're short. Close the short, open a long.
        elif action == 1 and self.status == -1:
            info("**close short, open long")
            self.close_inverse()
            self.open_position()
            pass
        # AI says short, but we're already short
        elif action == 2 and self.status == -1:
            info("**hold short")
            pass
        # AI says short, we're not in a position so go short
        elif action == 2 and self.status == 0:
            info("**opening short")
            self.open_inverse()
            pass
        # AI says short but we're long, close it
        elif action == 2 and self.status == 1:
            info("**closing long, opening short")
            self.close_position()
            self.open_inverse()
        else:  # assume hold
            info("**unknown - hold")
            pass
        self.update_position_value()

    def clear_trades(self):
        if self.status == 1:
            verbose("done so closing position")
            self.close_position()
        if self.status == -1:
            self.close_inverse()

    def open_position(self):
        self.status = 1
        super().open_position()

    def close_position(self):
        self.status = 0
        super().close_position()

    def open_inverse(self):
        # opening a short position, is opening a long position on the inverse
        self.status = -1
        self._open_position(self.inverse_df, self.inverse_product)

    def close_inverse(self):
        self.status = 0
        self._close_position(self.inverse_df, self.inverse_product)

    def update_position_value(self):
        if self.status == 0:
            df = self.orig_timeseries
            self.position_value = 0
            self.benchmark_value = self.get_bench_mark_value()
        elif self.status == 1:
            df = self.orig_timeseries
            self.position_value = self.get_position_value(df, self.current_index)
            self.benchmark_value = self.get_bench_mark_value()
        elif self.status == -1:
            df = self.inverse_df
            self.position_value = self.get_position_value(df, self.current_index)
            self.benchmark_value = self.get_bench_mark_value()
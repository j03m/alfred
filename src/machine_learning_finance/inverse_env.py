import yfinance as yf
from .logger import info, debug, error, verbose
import pandas as pd
from .trader_env import TraderEnv


def read_df_from_file(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def make_inverse_env_for(symbol, inverse_symbol, code, tail=-1, head=-1, data_source="yahoo", paths=[None, None]):
    df = None
    inverse_df = None
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

    if tail != -1:
        df = df.tail(tail)
        inverse_df = inverse_df.tail(tail)
    if head != -1:
        df = df.head(head)
        inverse_df = inverse_df.head(head)
    env = InverseEnv(symbol, df, inverse_symbol, inverse_df, code)
    return env


class InverseEnv(TraderEnv):

    def __init__(self, product, df, inverse_product, inverse_df, code):
        super(InverseEnv, self).__init__(product, df, code)
        self.inverse_product = inverse_product
        self.inverse_df = inverse_df
        self.status = 0  # 1 long 0 none 2 short

    def _apply_action(self, action):
        # AI says hold
        if action == 0:
            info("**holding")
            pass
        elif action == 1 and self.in_long:
            info("**holding long")
            pass
        elif action == 1 and not self.in_position:
            info("**opening long.")
            self.open_position()
        # AI says long, but we're short. Close the short, open a long.
        elif action == 1 and self.in_short:
            info("**close short, open long")
            self.close_inverse()
            self.open_position()
            pass
        # AI says short, but we're already short
        elif action == 2 and self.in_short:
            info("**hold short")
            pass
        # AI says short, we're not in a position so go short
        elif action == 2 and not self.in_position:
            info("**opening short")
            self.open_inverse()
            pass
        # AI says short but we're long, close it
        elif action == 2 and self.in_long:
            info("**closing long, opening short")
            self.close_position()
            self.open_inverse()
        else:  # assume hold
            info("**unknown - hold")
            pass
        self.update_position_value()

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
            self.benchmark_value = self.get_bench_mark_value(df, self.current_index)
        elif self.status == 1:
            df = self.orig_timeseries
            self.position_value = self.get_position_value(df, self.current_index)
            self.benchmark_value = self.get_bench_mark_value(df, self.current_index)
        elif self.status == -1:
            df = self.inverse_df
            self.position_value = self.get_position_value(df, self.current_index)
            self.benchmark_value = self.get_bench_mark_value(df, self.current_index)

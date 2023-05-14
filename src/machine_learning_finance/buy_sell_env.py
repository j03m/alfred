import yfinance as yf
from .logger import info, debug, error, verbose
import pandas as pd
from .trader_env import TraderEnv


class BuySellEnv(TraderEnv):

    def __init__(self,
                 product,
                 test_period_df,
                 historical_period_df,
                 code=1,
                 cash=5000,
                 prob_high=0.8,
                 prob_low=0.2):
        super(BuySellEnv, self).__init__(product, test_period_df, historical_period_df, code, cash, prob_high, prob_low)
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
            info("**opening long.")
            self.open_position()
            pass
        # AI says short, but we're already short
        elif action == 2 and self.status == -1:
            info("**not shorting, wait")
            pass
        # AI says short, we're not in a position so go short
        elif action == 2 and self.status == 0:
            info("**not shorting, wait")
            pass
        # AI says short but we're long, close it
        elif action == 2 and self.status == 1:
            info("**closing long, short called")
            self.close_position()

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
            self.position_value = 0
            self.benchmark_value = self.get_bench_mark_value()

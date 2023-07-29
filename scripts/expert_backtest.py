#!/usr/bin/env python3
import argparse
from machine_learning_finance import (back_test_expert, TraderEnv, TailTrainingWindowUtil, RangeTrainingWindowUtil,
                                      download_symbol,
                                      CURRICULUM_BACK_TEST, CURRICULUM_GUIDE)
import pandas as pd
import warnings
from datetime import datetime, timedelta


def main():
    # filter out UserWarning messages
    warnings.filterwarnings("ignore", category=UserWarning)

    now = datetime.now()
    start_default = now - timedelta(days=365)
    start_default_str = start_default.strftime('%Y-%m-%d')
    end_default_str = now.strftime('%Y-%m-%d')

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--symbol", default="SPY", help="Symbol to use (default: SPY)")
    # parser.add_argument("-u", "--curriculum", type=int, choices=[CURRICULUM_GUIDE, CURRICULUM_BACK_TEST],
    #                     default=CURRICULUM_BACK_TEST,
    #                     help="Curriculum level (default: 2)")
    parser.add_argument("-t", "--tail", type=int, default=None, help="Tail size - use instead of start/end")
    # parser.add_argument("-e", "--env-type", type=str, default="long-short",
    #                     help="Environment to use: 'long-short, options, inverse, buy-sell")
    parser.add_argument("-c", "--cash", type=int, default=5000,
                        help="how much cash to trade with")
    parser.add_argument('--start', type=str, default=start_default_str)
    parser.add_argument('--end', type=str, default=end_default_str)

    args = parser.parse_args()

    print(args)

    if args.tail is not None:
        training_window = TailTrainingWindowUtil(download_symbol(args.symbol), args.tail)
    else:
        training_window = RangeTrainingWindowUtil(download_symbol(args.symbol), args.start, args.end)

    env = TraderEnv(args.symbol, training_window.test_df, training_window.full_hist_df,
                    curriculum_code=CURRICULUM_BACK_TEST, cash=args.cash)

    env = back_test_expert(env)
    env.ledger.to_csv(f"./backtests/{args.symbol}-expert-back-test.csv")
    print("result: ", env)

    # These environments need to be revamped agains the new data set
    # if args.env_type == "inverse":
    #     if args.crypto:
    #         inverse_file = "./data/inverse_coins.csv"
    #     else:
    #         inverse_file = "./data/inverse_pairs.csv"
    #
    #     df = pd.read_csv(inverse_file)
    #     df = df.set_index('Main')
    #     if args.symbol in df.index:
    #         inverse_value = df.at[args.symbol, 'Inverse']
    #     else:
    #         print(f"{args.symbol} not found in {inverse_file}")
    #         exit(-1)
    #
    #     env = make_inverse_env_for(args.symbol,
    #                                inverse_value,
    #                                args.curriculum,
    #                                args.tail,
    #                                cash=args.cash,
    #                                start=args.start_time,
    #                                end=args.end_time,
    #                                prob_high=args.high_probability,
    #                                prob_low=args.low_probability,
    #                                data_source=data_source)
    #     env = back_test_expert(env)
    #     env.ledger.to_csv(f"./backtests/backtest_inverse_{gen_file_id(args)}.csv")
    #
    # if args.env_type == "buy-sell":
    #     env = make_env_for(args.symbol,
    #                        args.curriculum,
    #                        args.tail,
    #                        cash=args.cash,
    #                        prob_high=args.high_probability,
    #                        prob_low=args.low_probability,
    #                        env_class=BuySellEnv,
    #                        data_source=data_source)
    #     env = back_test_expert(env)
    #     env.ledger.to_csv(f"./backtests/backtest_buy_sell_{gen_file_id(args)}.csv")
    # if args.env_type == "options":
    #     raise Exception("Implement me")


if __name__ == "__main__":
    main()

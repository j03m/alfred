import backtrader as bt
import torch
import argparse

from datetime import datetime

from alfred.utils import set_deterministic, read_time_series_file, trim_timerange
from alfred.easy import prepare_data_and_model
from alfred.devices import set_device

device = set_device()

from collections import OrderedDict

def print_trade_analysis_tree(trade_analysis, indent=0):

    indent_str = "  " * indent  # Two spaces per indent level

    for key, value in trade_analysis.items():
        print(f"{indent_str}- {key}:")
        if isinstance(value, OrderedDict): # or isinstance(value, AutoOrderedDict) if you were still using that
            print_trade_analysis_tree(value, indent + 1) # Recursive call for nested dictionaries
        else:
            print(f"{indent_str}  - {value}")


class SimpleBackTest(bt.Strategy):
    model = None
    loader = None
    params = dict(init_buy_confidence = 0.7,
                  close_buy_confidence = 0.5,
                  init_short_confidence = 0.3,
                  close_short_confidence = 0.5)
    def __init__(self):
        assert SimpleBackTest.model is not None
        assert SimpleBackTest.loader is not None
        SimpleBackTest.model.eval()
        self.loader_iter = iter(SimpleBackTest.loader)
        self.dataclose = self.datas[0].close


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):

        if not SimpleBackTest.loader:
            raise Exception("Bad configuration - no loader for model")

        if not SimpleBackTest.model:
            raise Exception("Bad configuration - no model set")


        # Prepare input features for the model
        try:
            features, _ = next(self.loader_iter)
            features = features.to(device)
        except StopIteration:
            print("Reached the end of the dataset.")
            if self.position.size != 0:
                self.close()

        # Get model prediction
        with torch.no_grad():
            prediction = SimpleBackTest.model(features).squeeze()

        # Trading logic using the confidence thresholds
        value = prediction.item()
        init_buy_confidence = self.params.init_buy_confidence
        init_short_confidence = self.params.init_short_confidence
        close_buy_confidence = self.params.close_buy_confidence
        close_short_confidence = self.params.close_short_confidence

        # Check for buy signal
        if value >= init_buy_confidence:
            if self.position.size < 0:
                self.log(f'CLOSE SHORT, then BUY CREATE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.close()  # Close short position
                self.order = self.buy() # Open long position immediately after closing short
            elif self.position.size == 0:
                self.log(f'BUY CREATE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.order = self.buy()  # Enter long position

        # Check for sell signal
        elif value <= init_short_confidence:
            if self.position.size > 0:
                self.log(f'CLOSE LONG, then SELL CREATE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.close()  # Close long position
                self.order = self.sell() # Open short position immediately after closing long
            elif self.position.size == 0:
                self.log(f'SELL CREATE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.order = self.sell()  # Enter short position

        # Check for close long position signal (if already long)
        elif self.position.size > 0 and value <= close_buy_confidence:
            self.log(f'CLOSE LONG, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
            self.close()  # Close long position
            if value <= init_short_confidence: # Check for immediate short after closing long
                self.log(f'SELL CREATE AFTER LONG CLOSE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.order = self.sell()


        # Check for close short position signal (if already short)
        elif self.position.size < 0 and value >= close_short_confidence:
            self.log(f'CLOSE SHORT, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
            self.close()  # Close short position
            if value >= init_buy_confidence: # Check for immediate buy after closing short
                self.log(f'BUY CREATE AFTER SHORT CLOSE, Value={value:.2f}, Close={self.dataclose[0]:.2f}')
                self.order = self.buy()



    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            else:  # Sell
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            if order.rejected:
                print(f"Order Rejected Reason: {order.rejectreason_descr}")

        self.order = None  # Reset order

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))


def print_strategy_summary(strategy, strategy_name):
    strat = strategy
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
    cagr = strat.analyzers.returns.get_analysis()['rnorm100']
    max_drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    tret_analyzer = strat.analyzers.getbyname('timereturn')


    print(f"\n--- {strategy_name} Strategy Summary ---")
    print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"  CAGR:                {cagr:.2f}%")
    print(f"  Max Drawdown:        {max_drawdown:.2f}%")

    print("   Trades analysis:")
    print_trade_analysis_tree(trade_analysis)
    print(f"  TRET Analysis:")
    print_trade_analysis_tree(tret_analyzer.get_analysis())


def run_backtest(df, model_strategy_class, args):
    # Backtest Strategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        model_strategy_class,
        init_buy_confidence=args.init_buy_confidence,  # Pass params from args
        close_buy_confidence=args.close_buy_confidence,
        init_short_confidence=args.init_short_confidence,
        close_short_confidence=args.close_short_confidence
    )

    # Add data feed
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years, data=data)

    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    print_strategy_summary(results[0], "ModelStrategy") # SimpleBackTest is now model strategy


    # Plot the results (optional)
    if args.plot:
        cerebro.plot()


def main(args):

    global gbl_model
    global gbl_loader
    set_deterministic(args.seed)  # Set seed for reproducibility

    # Validate dates
    try:
        start_date = datetime.strptime(args.min_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.max_date, '%Y-%m-%d')
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")
    except ValueError as e:
        raise ValueError(f"Invalid date format or range: {e}")

    file = f"data/{args.test_ticker}_quarterly_directional.csv"
    model, _, _, loader, _, _, was_loaded = prepare_data_and_model(model_name=args.model_name,
                                                                   model_size=args.model_size,
                                                                   shuffle=False,
                                                                   files=[file],
                                                                   batch_size=1,
                                                                   augment_func=lambda df: trim_timerange(df, min_date=start_date, max_date=end_date))
    if not was_loaded:
        print("WARNING: you're backtesting with a brand new model. ")


    # back test wants a pandas dataframe. We need to give it one, but we also need to get our
    # loader to the model. We'll reread the file here and capture the loader as a global
    df = trim_timerange(read_time_series_file(file), args.min_date, args.max_date)
    df = df[["Close", "Volume"]]
    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    SimpleBackTest.model = model
    SimpleBackTest.loader = loader

    run_backtest(df, SimpleBackTest, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a model strategy, benchmark vs buy/hold.")
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default="2024-12-31",
                        help='Maximum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument("--test_ticker", type=str, default="AAPL", help="Ticker symbol for the test asset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--model_name", type=str, default="vanilla.medium", help="model to back test")
    parser.add_argument("--model_size", type=int, default=1024, help="model size")
    parser.add_argument("--init_buy_confidence", type=float, default=0.7, help="score >= needed to initiate a buy")
    parser.add_argument("--close_buy_confidence", type=float, default=0.5, help="score <= needed to close a buy")
    parser.add_argument("--init_short_confidence", type=float, default=0.3, help="score <= needed to initiate a buy")
    parser.add_argument("--close_short_confidence", type=float, default=0.5, help="score >= needed to close a buy")
    parser.add_argument("--cash", type=float, default=1_000_000, help="cash start")
    parser.add_argument("--commission", type=float, default=0.0002, help="trade commission")
    parser.add_argument("--plot", action="store_true", help="Plot the backtest results")

    args = parser.parse_args()
    main(args)
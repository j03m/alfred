from alfred.data import YahooNextCloseWindowDataSet
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# graph log return, cumsum, scaler 2, scaler 5, scaler 10
ticker = 'SPY'
start = '1999-01-01'
end = '2021-01-01'
seq_length = 30
num_features = 1
dataset = YahooNextCloseWindowDataSet(ticker, start, end, seq_length, change=1, log_return_scaler=True)


def plot(index, x):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(index, x, color="blue")

    plt.title("Daily close price")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.show(block=False)

data = dataset.scaler.original
index = dataset.df.index[:len(data)]
plot(index, data)

data = dataset.scaler.log_returns
index = dataset.df.index[:len(data)]
plot(index, data)

data = dataset.scaler.cumsum
index = dataset.df.index[:len(data)]
plot(index, data)

data = dataset.scaler.amplified * 10
index = dataset.df.index[:len(data)]
plot(index, data)

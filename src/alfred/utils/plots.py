import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_timeseries(index, x):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(index, x, color="blue")

    plt.title("Daily close price")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.show(block=False)
    return fig


def plot_multi_series(data, title):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))

    # Iterate over the data tuples and plot each series
    for index, series, label, color in data:
        plt.plot(index, series, label=label, color=color)

    plt.title(title)
    plt.grid(which='major', axis='y', linestyle='--')

    # Add a legend to differentiate the series
    plt.legend()

    # Return the figure object for further manipulation (saving, showing, etc.)
    return fig

def plot_evaluation(actuals, predictions):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Values')
    plt.plot(predictions, label='Predictions', alpha=0.7)
    plt.title('Predictions vs Actuals')
    plt.xlabel('Sample Index')
    plt.ylabel('Scaled Price Change')
    plt.legend()
    plt.show(block=False)
    return fig
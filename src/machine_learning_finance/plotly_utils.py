import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def scale_to_price(series, df):
    price_range = [df['Close'].min(), df['Close'].max()]
    scaled_series = (series - series.min()) / (series.max() - series.min()) * (price_range[1] - price_range[0]) + \
                    price_range[0]
    return scaled_series


def histogram(df):
    # Assuming df is your DataFrame containing 'probability', 'Est Close', 'Percent Change' columns

    # Create the histogram
    fig = go.Figure(data=[go.Histogram(x=df['probability'], nbinsx=20)])

    # Set the title and axis labels
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Frequency",
    )

    # Show the plot
    fig.show()

def plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode="lines", name="Value"))
    fig.show()


def plot_expert(df):
    scaled_prob_above_trend = pd.Series(scale_to_price(df["prob_above_trend"], df))
    line_index = df.tail(len(df["trend"])).index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Value"))
    fig.add_trace(go.Scatter(x=df.index, y=df["trend"], mode="lines", name="Trend"))
    fig.add_trace(go.Scatter(x=df.index, y=scaled_prob_above_trend, mode='lines', name='Prob Above Trend'))

    long_entry_dates = df.loc[df['action'] == 1].index
    short_entry_dates = df.loc[df['action'] == 2].index

    # Create long_entry and long_exit columns in df_raw DataFrame
    df['long_entry'] = np.where(df.index.isin(long_entry_dates), df['Close'], np.nan)
    df['short_entry'] = np.where(df.index.isin(short_entry_dates), df['Close'], np.nan)

    fig.add_trace(go.Scatter(x=df.index, y=df['long_entry'], mode='markers', name='Long Enter',
                             marker=dict(symbol='triangle-up', size=8, color='green')))

    fig.add_trace(go.Scatter(x=df.index, y=df['short_entry'], mode='markers', name='Long Exit',
                             marker=dict(symbol='triangle-down', size=8, color='red')))

    fig.update_layout(title='Expert opinions', xaxis_title='Date',
                      height=800)
    fig.show()


def add_marker(fig, df, col, name, mark_type, size, color):
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='markers', name=name,
                             marker=dict(symbol=mark_type, size=size, color=color)))


def add_line(fig, df, col, name, color):
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name, line=dict(color=color)))


def plot_backtest_differences(df):
    fig = go.Figure()

    # Add trace for the main time series plot to the first row of the subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Value"))

    df["differences"] = np.where(df["expert"] == df["model sentiment"], df["Close"], np.nan)

    add_marker(fig, df, "differences", "differences", "triangle-up", 8, "springgreen")

    return fig


def plot_backtest_analysis(df, ledger, save_png=False, png_file=None, inverse=None):
    # Scale probabilities to the same range as the original time series

    fig = go.Figure()

    # Add trace for the main time series plot to the first row of the subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Value"))

    ledger['Date'] = pd.to_datetime(ledger['Date'])

    # Filter ledger DataFrame to get long_entry and long_exit dates
    if inverse is None:
        long_entry_dates = ledger.loc[(ledger['Action'] == 'enter') & (ledger['Side'] == 'long'), 'Date']
        long_exit_dates = ledger.loc[(ledger['Action'] == 'exit') & (ledger['Side'] == 'long'), 'Date']
        short_entry_dates = ledger.loc[(ledger['Action'] == 'enter') & (ledger['Side'] == 'short'), 'Date']
        short_exit_dates = ledger.loc[(ledger['Action'] == 'exit') & (ledger['Side'] == 'short'), 'Date']
    else:
        long_entry_dates = ledger.loc[
            (ledger['Action'] == 'enter') & (ledger['Side'] == 'long') & (ledger['Product'] == inverse), 'Date']
        long_exit_dates = ledger.loc[
            (ledger['Action'] == 'exit') & (ledger['Side'] == 'long') & (ledger['Product'] == inverse), 'Date']
        short_entry_dates = ledger.loc[
            (ledger['Action'] == 'enter') & (ledger['Side'] == 'long') & (ledger['Product'] != inverse), 'Date']
        short_exit_dates = ledger.loc[
            (ledger['Action'] == 'exit') & (ledger['Side'] == 'long') & (ledger['Product'] != inverse), 'Date']

    # Create long_entry and long_exit columns in df_raw DataFrame
    df['long_entry'] = np.where(df.index.isin(long_entry_dates), df['Close'], np.nan)
    df['long_exit'] = np.where(df.index.isin(long_exit_dates), df['Close'], np.nan)
    df['short_entry'] = np.where(df.index.isin(short_entry_dates), df['Close'], np.nan)
    df['short_exit'] = np.where(df.index.isin(short_exit_dates), df['Close'], np.nan)

    fig.add_trace(go.Scatter(x=df.index, y=df['long_entry'], mode='markers', name='Long Enter',
                             marker=dict(symbol='triangle-up', size=8, color='green')))

    fig.add_trace(go.Scatter(x=df.index, y=df['long_exit'], mode='markers', name='Long Exit',
                             marker=dict(symbol='triangle-down', size=8, color='green')))

    fig.add_trace(go.Scatter(x=df.index, y=df['short_entry'], mode='markers', name='Short Enter',
                             marker=dict(symbol='triangle-down', size=8, color='red')))

    fig.add_trace(go.Scatter(x=df.index, y=df['short_exit'], mode='markers', name='Short Exit',
                             marker=dict(symbol='triangle-up', size=8, color='red')))

    if save_png:
        pio.write_image(fig, png_file)
    else:
        fig.update_layout(title='Backtest Analysis', xaxis_title='Date', height=800)

    return fig


def plot_full_analysis(df, trend, prob_above_trend, prob_below_trend, model, df_durations, ledger=None):
    # Scale probabilities to the same range as the original time series
    scaled_prob_above_trend = pd.Series(scale_to_price(prob_above_trend, df))  # .rolling(window=30, center=True).mean()
    scaled_prob_below_trend = pd.Series(scale_to_price(prob_below_trend, df))  # .rolling(window=30, center=True)

    line_index = df.tail(len(trend)).index

    # Create subplots with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Add trace for the main time series plot to the first row of the subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Value"), row=1, col=1)
    fig.add_trace(go.Scatter(x=line_index, y=trend, mode="lines", name="Trend"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=scaled_prob_above_trend, mode='lines', name='Prob Above Trend'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=scaled_prob_below_trend, mode='lines', name='Prob Below Trend'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['high_prob_start'], mode='markers', name='Window Start',
                             marker=dict(symbol='diamond', size=8, color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cross_over_positive'], mode='markers', name='Up Cross',
                             marker=dict(symbol='diamond', size=8, color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cross_over_negative'], mode='markers', name='Down Cross',
                             marker=dict(symbol='diamond', size=8, color='red')), row=1, col=1)

    last_30_days_trend = trend[-30:]
    x = np.arange(1, 60)
    line_pred = model.predict(x.reshape(-1, 1))
    new_index = pd.date_range(start=last_30_days_trend.index[0] + pd.DateOffset(days=1),
                              end=df.index[-1] + pd.DateOffset(days=31))

    fig.add_trace(go.Scatter(x=new_index, y=line_pred, mode='lines', name='Linear Regression Line'), row=1, col=1)

    if ledger is not None:
        ledger['Date'] = pd.to_datetime(ledger['Date'])
        # Filter ledger DataFrame to get long_entry and long_exit dates
        long_entry_dates = ledger.loc[(ledger['Action'] == 'enter') & (ledger['Side'] == 'long'), 'Date']
        long_exit_dates = ledger.loc[(ledger['Action'] == 'exit') & (ledger['Side'] == 'long'), 'Date']
        short_entry_dates = ledger.loc[(ledger['Action'] == 'enter') & (ledger['Side'] == 'short'), 'Date']
        short_exit_dates = ledger.loc[(ledger['Action'] == 'exit') & (ledger['Side'] == 'short'), 'Date']

        # Create long_entry and long_exit columns in df_raw DataFrame
        df['long_entry'] = np.where(df.index.isin(long_entry_dates), df['Close'], np.nan)
        df['long_exit'] = np.where(df.index.isin(long_exit_dates), df['Close'], np.nan)
        df['short_entry'] = np.where(df.index.isin(short_entry_dates), df['Close'], np.nan)
        df['short_exit'] = np.where(df.index.isin(short_exit_dates), df['Close'], np.nan)

        fig.add_trace(go.Scatter(x=df.index, y=df['long_entry'], mode='markers', name='Long Enter',
                                 marker=dict(symbol='triangle-up', size=8, color='green')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['long_exit'], mode='markers', name='Long Exit',
                                 marker=dict(symbol='triangle-down', size=8, color='green')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['short_entry'], mode='markers', name='Short Enter',
                                 marker=dict(symbol='triangle-down', size=8, color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['short_exit'], mode='markers', name='Short Exit',
                                 marker=dict(symbol='triangle-up', size=8, color='red')), row=1, col=1)
    # Add trace for the durations plot to the second row of the subplot
    if df_durations is not None:
        for i, row in df_durations.iterrows():
            duration_trace = go.Scatter(x=[row['start'], row['end']], y=[5, 5], mode='lines',
                                        line=dict(color='red', width=row['duration'] / 2),
                                        name=f'Duration {row["duration"]}')
            fig.add_trace(duration_trace, row=2, col=1)

    fig.update_layout(title='Time Series with Trend, Scaled Seasonal Component, and Probabilities', xaxis_title='Date',
                      height=800)
    fig.show()


def bar_chart(df, long=True):
    # Normalize the 'Percent Change' column to be between 0 and 1

    # Create a gradient color scale based on the normalized 'Percent Change' and the 'long' value
    colors = ["blue"]

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(x=df.index, y=df['probability'], marker_color=colors)])

    # Set the title and axis labels
    fig.update_layout(
        title="Probabilities Over Time",
        xaxis_title="Date",
        yaxis_title="Probability",
    )

    # Show the plot
    fig.show()


def prob_chart(df, values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=values, mode='lines', name='Prob Above Trend'))
    fig.update_layout(
        title="Probabilities Over Time",
        xaxis_title="Date",
        yaxis_title="Probability",
    )
    # Show the plot
    fig.show()


def graph_pdf_bar(df):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df["Percentage Deviation"], y=df["PDF Value"], marker_color='blue', name="PDF")
    )

    fig.update_layout(
        title="Probability Density Function",
        xaxis_title="Percentage Deviation from Trend (%)",
        yaxis_title="Probability",
    )

    fig.show()

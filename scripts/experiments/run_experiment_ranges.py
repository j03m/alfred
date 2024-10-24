from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS

SETTINGS["CAPTURE_MODE"]="no"

import argparse
import zlib
import random

from alfred.metadata import ExperimentSelector, TickerCategories, ColumnSelector
from alfred.devices import set_device, build_model_token
from alfred.data import CachedStockDataSet
from alfred.model_persistence import get_latest_model
from alfred.model_evaluation import simple_profit_measure, analyze_ledger, evaluate_model
from alfred.model_training import train_model
from alfred.models import LSTMModel, AdvancedLSTM, LSTMConv1d, TransAm
from alfred.utils import plot_multi_series, plot_evaluation
from sklearn.metrics import mean_squared_error

import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = set_device()
gbl_args = None

scaler_config = [
    {'regex': r'^Close$', 'type': 'yeo-johnson'},
    {'columns': ['^VIX'], 'type': 'standard'},
    {'columns': ['SPY', 'CL=F', 'BZ=F'], 'type': 'yeo-johnson'},
    {'regex': r'^Margin.*', 'type': 'standard'},
    {'regex': r'^Volume$', 'type': 'yeo-johnson'},
    {'columns': ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage'], 'type': 'standard'},
    {'regex': r'\d+year', 'type': 'standard'}
]

# Initialize Sacred experiment
ex = Experiment("experiment_runner")
#ex.observers.append(FileStorageObserver.create('sacred_runs'))
ex.observers.append(MongoObserver(
    url='mongodb://localhost:27017/',
    db_name='sacred_db'
))

# Experiment configuration (default values)
@ex.config
def config():

    model_token = None,
    size = None,
    sequence_length = None
    bar_type = None
    data = None


def crc32_columns(strings):
    # Sort the array of strings
    sorted_strings = sorted(strings)

    # Concatenate the sorted strings into one string
    concatenated_string = ''.join(sorted_strings)

    # Convert the concatenated string to bytes
    concatenated_bytes = concatenated_string.encode('utf-8')

    # Compute the CRC32 hash
    crc32_hash = zlib.crc32(concatenated_bytes)

    # Return the hash in hexadecimal format
    return f"{crc32_hash:#08x}"


def model_from_config(config_token,
                      num_features,
                      sequence_length,
                      size,
                      output,
                      descriptors,
                      model_path, layers=2):
    if config_token == 'lstm':
        model = LSTMModel(features=num_features, hidden_dim=size, output_size=output,
                          num_layers=layers)
    elif config_token == 'advanced-lstm':
        model = AdvancedLSTM(features=num_features, hidden_dim=size, output_dim=output)
    elif config_token == 'lstm-conv1d':
        # size 10 kernel should smooth about 2 weeks of data
        model = LSTMConv1d(features=num_features, seq_len=sequence_length, hidden_dim=size, output_size=output,
                           kernel_size=10)
    elif config_token == 'trans-am':
         model = TransAm(features=num_features, model_size=size, heads=size/16, output=output, last_bar=True)
    else:
        raise Exception("Model type not supported")

    model.to(DEVICE)

    model_token = build_model_token(descriptors)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    model_checkpoint = get_latest_model(model_path, model_token)
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler, model_token


# Main function to run the experiment
@ex.main
def run_experiment(model_token, size, sequence_length, bar_type, data):
    print(
        f"Running experiment: Model={model_token}, Size={size}, Sequence length={sequence_length}, bar_type={bar_type}, Data={data}")

    _args = gbl_args
    assert gbl_args, "additional args required"
    print(f"Additional args: ", _args)

    column_selector = ColumnSelector(f"{_args.metadata_path}/column-descriptors.json")

    columns = column_selector.get(data)

    output = 1
    model, optimizer, scheduler, real_model_token = model_from_config(
        num_features=len(columns),
        config_token=model_token,
        sequence_length=sequence_length, size=size, output=output,
        descriptors=[
            model_token, sequence_length, size, output, crc32_columns(columns)
        ], model_path=_args.model_path)

    ticker_categories = TickerCategories(f"{_args.metadata_path}/ticker-categorization.json")

    model.train()

    # train on all training tickers
    for ticker in ticker_categories.get(["training"]):
        print("training against: ", ticker)
        dataset = CachedStockDataSet(symbol=ticker,
                                     seed=_args.seed,
                                     scaler_config=scaler_config,
                                     period_length=_args.period,
                                     sequence_length=sequence_length,
                                     feature_columns=columns,
                                     target_columns=["Close"])  # todo ... hmmm will need to mature this past just Close
        train_loader = DataLoader(dataset, batch_size=_args.batch_size, shuffle=False, drop_last=True)

        train_model(model, optimizer, scheduler, train_loader, _args.patience, _args.model_path, real_model_token,
                    epochs=_args.epochs, training_label=ticker)

    # Initialize variables to accumulate values
    total_mse = 0
    total_profit = 0
    total_bhp = 0
    ledger_metrics_aggregate = {}  # To accumulate metrics like win rate, trades count, etc.
    eval_tickers = ticker_categories.get(["evaluation"])
    ticker_count = len(eval_tickers)

    for ticker in eval_tickers:
        dataset = CachedStockDataSet(symbol=ticker,
                                     seed=_args.seed,
                                     scaler_config=scaler_config,
                                     period_length=_args.period,
                                     sequence_length=sequence_length,
                                     feature_columns=columns,
                                     bar_type=bar_type,
                                     target_columns=["Close"])
        eval_loader = DataLoader(dataset, batch_size=_args.batch_size, shuffle=False, drop_last=True)

        # Get predictions and actual values
        predictions, actuals = evaluate_model(model, eval_loader)

        if _args.plot:
            fig = plot_evaluation(actuals, predictions)
            fig.savefig(f"{_args.model_path}/{real_model_token}_{ticker}_eval.png", dpi=600, bbox_inches='tight',
                        transparent=True)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(actuals, predictions)
        total_mse += mse

        # Unscale predictions and actuals
        unscaled_predictions = dataset.scaler.inverse_transform_column("Close", np.array(predictions).reshape(-1, 1))
        unscaled_actuals = dataset.scaler.inverse_transform_column("Close", np.array(actuals).reshape(-1, 1))

        # Calculate simple profit and ledger
        profit, ledger_df, bhp = simple_profit_measure(unscaled_predictions.squeeze(), unscaled_actuals.squeeze())
        total_bhp += bhp
        total_profit += profit

        # Analyze the ledger for key metrics
        metrics = analyze_ledger(ledger_df)

        # Accumulate ledger metrics, e.g., summing up win rates, number of trades, etc.
        for key, value in metrics.items():
            if key not in ledger_metrics_aggregate:
                ledger_metrics_aggregate[key] = value
            else:
                ledger_metrics_aggregate[key] += value

    # After looping through all tickers, calculate the aggregates

    # Average MSE across all tickers
    average_mse = total_mse / ticker_count

    # Average or aggregated ledger metrics, depending on how you want to handle them
    # Example: For win rate, you might want the average win rate
    if 'win_rate' in ledger_metrics_aggregate:
        ledger_metrics_aggregate['win_rate'] /= ticker_count  # Averaging the win rates

    # Return aggregate results for all tickers
    aggregate_results = {
        'average_mse': average_mse,
        'total_profit': total_profit,
        'trade_vs_bhp': total_profit - total_bhp,
        'ledger_metrics': ledger_metrics_aggregate
    }

    # Optionally, log the aggregate results to Sacred
    ex.log_scalar('average_mse', average_mse)
    ex.log_scalar('total_profit', total_profit)
    ex.log_scalar('trade_vs_bhp', total_profit - total_bhp)
    for key, value in ledger_metrics_aggregate.items():
        ex.log_scalar(f'aggregate_{key}', value)

    return aggregate_results


def main(args):
    # Use ExperimentSelector to select experiments based on ranges
    selector = ExperimentSelector(args.index_file)
    experiments = selector.get(include_ranges=args.include, exclude_ranges=args.exclude)
    random.shuffle(experiments)

    # Run the experiments using Sacred
    for experiment in experiments:
        if experiment:
            ex.run(config_updates=experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected experiments using Sacred.")
    parser.add_argument("--index-file", type=str, default="./metadata/experiment-index.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--include", type=str, default="",
                        help="Ranges of experiments to include (e.g., 1-5,10-15)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Ranges of experiments to exclude (e.g., 4-5,8)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed is combined with a ticker to produce a consistent random training and eval period")
    parser.add_argument("--period", type=int, default=365 * 2,
                        help="length of training data")
    parser.add_argument("--epochs", type=int, default=2500,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=125,
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    parser.add_argument("--metadata-path", type=str, default='./metadata', help="experiment descriptors live here")
    parser.add_argument("--plot", action="store_true", help="make plots")
    gbl_args = parser.parse_args()
    main(gbl_args)

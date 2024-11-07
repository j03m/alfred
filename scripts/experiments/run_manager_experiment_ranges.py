from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS

import pandas as pd
import numpy as np

import argparse
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from alfred.data import CustomScaler, DEFAULT_SCALER_CONFIG
from alfred.metadata import ExperimentSelector
from alfred.model_persistence import model_from_config, prune_old_versions
from alfred.model_training import train_model
from alfred.model_evaluation import evaluate_model

from sklearn.metrics import mean_squared_error

MONGO = 'mongodb://localhost:27017/'
DB = 'sacred_db'

experiment_namespace = "mgmt_experiment_set"
ex = Experiment(experiment_namespace)
ex.observers.append(MongoObserver(
    url=MONGO,
    db_name=DB
))

gbl_args = None


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    data_values = data.values
    target_index = data.columns.get_loc('Rank')
    feature_indices = [i for i in range(data.shape[1]) if i != target_index]

    for i in range(len(data_values) - sequence_length):
        # Extract sequence of features
        seq = data_values[i:i + sequence_length, feature_indices]
        # Extract target value at the next time step
        target = data_values[i + sequence_length, target_index]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def build_experiment_descriptor_key(config):
    model_token = config["sequence_length"]
    size = config["sequence_length"]
    sequence_length = config["sequence_length"]
    return f"{model_token}:{size}:{sequence_length}"


@ex.config
def config():
    model_token = None,
    size = None,
    sequence_length = None


@ex.main
def run_experiment(model_token, size, sequence_length):
    # Read the training file

    df = pd.read_csv(gbl_args.input_file)
    date_column = "Unnamed: 0"
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    scaler = CustomScaler(config=DEFAULT_SCALER_CONFIG, df=df)
    df = scaler.fit_transform(df)

    # Split into 2/3 training and 1/3 evaluation
    split_index = int(len(df) * 2 / 3)

    # Split the DataFrame
    train_df = df.iloc[:split_index]
    eval_df = df.iloc[split_index:]

    output = 1
    model, optimizer, scheduler, real_model_token = model_from_config(
        num_features=len(train_df.columns),
        config_token=model_token,
        sequence_length=sequence_length, size=size, output=output,
        descriptors=[
            "port_mgmt", model_token, sequence_length, size, output
        ], model_path=gbl_args.model_path)

    X_train, y_train = create_sequences(train_df, sequence_length)
    X_eval, y_eval = create_sequences(eval_df, sequence_length)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(-1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
    train_loader = DataLoader(train_dataset, batch_size=gbl_args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=gbl_args.batch_size, shuffle=False)

#params busted here
    train_model(model, optimizer, scheduler, train_loader, gbl_args.patience, epochs=gbl_args.epochs)
    prune_old_versions(gbl_args.model_path)

    predictions, actuals = evaluate_model(model, eval_loader)
    mse = mean_squared_error(actuals, predictions)

    # take our predicted Ranks for each prediction
    predictions = np.array(predictions)

    padding_length = sequence_length

    # Create the padded predictions array
    padded_predictions = np.concatenate([np.full((padding_length,), np.nan), predictions])

    # Append as a new column to eval_df
    eval_df['predicted_rank'] = padded_predictions

    ex.log_scalar('mse', mse)
    ex.info["model_token"] = real_model_token


def main(args):
    selector = ExperimentSelector(args.index_file, mongo=MONGO, db=DB)
    experiments = selector.get(include_ranges=args.include, exclude_ranges=args.exclude)
    random.shuffle(experiments)

    # get a list of past or in flight experiments
    past_experiments = selector.get_current_state(experiment_namespace, build_experiment_descriptor_key)

    # Run the experiments using Sacred
    for experiment in experiments:
        if experiment:
            key = build_experiment_descriptor_key(experiment)
            if key not in past_experiments:
                ex.run(config_updates=experiment)
                # update the list in case another machine is running (poor man's update)
                past_experiments = selector.get_current_state("analysts", build_experiment_descriptor_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run selected experiments using Sacred.")
    parser.add_argument("--index-file", type=str, default="./metadata/mgmt-experiment-index.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--input-file", type=str, default="./results/portfolio_manager_training.csv",
                        help="Training and eval data for the manager")
    parser.add_argument("--column-file", type=str, default="./metadata/column-descriptors.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--include", type=str, default="",
                        help="Ranges of experiments to include (e.g., 1-5,10-15)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Ranges of experiments to exclude (e.g., 4-5,8)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size")
    parser.add_argument("--period", type=int, default=60,
                        help="length of training data windows")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=75,
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--model-path", type=str, default='./models',
                        help="where to store models and best loss data")
    parser.add_argument("--metadata-path", type=str, default='./metadata', help="experiment descriptors live here")
    parser.add_argument("--plot", action="store_true", help="make plots")
    gbl_args = parser.parse_args()
    main(gbl_args)

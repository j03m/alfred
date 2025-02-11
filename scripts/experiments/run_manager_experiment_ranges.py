from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS

SETTINGS["CAPTURE_MODE"] = "no"

import pandas as pd
import numpy as np

import argparse
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from alfred.data import CustomScaler, PM_SCALER_CONFIG
from alfred.metadata import ExperimentSelector, ColumnSelector
from alfred.model_persistence import model_from_config, prune_old_versions, crc32_columns
from alfred.model_training import train_model
from alfred.model_evaluation import evaluate_model, nrmse_by_range
from alfred.utils import MongoConnectionStrings

connect_data = MongoConnectionStrings()
# sacred will init its own client
# this is just called to init the client and get the right strings.
# Bad design :/ see imp for details why (i had my reasons)
connect_data.get_mongo_client()
MONGO = connect_data.connection_string()
DB = 'sacred_db'

experiment_namespace = "mgmt_experiment_set"
ex = Experiment(experiment_namespace)
ex.add_config({'token': experiment_namespace})
ex.observers.append(MongoObserver(
    url=MONGO,
    db_name=DB
))

gbl_args = None


def generate_sequences(input_df, features, prediction, window_size, date_column="Date"):
    unique_dates = input_df.index.unique()
    no_index_df = input_df.reset_index()
    num_windows = len(unique_dates) - window_size + 1
    sequences = []
    consistent_ids = None
    consistent_dates = None
    for i in range(num_windows):
        # Get dates for the current window
        window_dates = unique_dates[i:i + window_size]
        # Get data for these dates
        window_data = no_index_df[no_index_df[date_column].isin(window_dates)].copy()
        # Ensure data is sorted by Date and ID
        window_data.sort_values(by=[date_column, 'ID'], inplace=True)
        window_data.reset_index(drop=True, inplace=True)
        # every window we create should have the number of IDs and Dates
        # if we don't the tensor shape will be jagged.
        if consistent_ids is None:
            consistent_ids = len(window_data["ID"].unique())
        else:
            # still choking here. We need the final data to be for
            assert consistent_ids == len(window_data["ID"].unique())

        if consistent_dates is None:
            consistent_dates = len(window_data[date_column].unique())
        else:
            assert consistent_dates == len(window_data[date_column].unique())

        # Append the sequence
        sequences.append(window_data)

    # Each sequence is a DataFrame containing data for window_size dates and all IDs
    # Optionally, convert each DataFrame to a NumPy array
    x_sequences_array = [np.array(seq[features].values) for seq in sequences]
    y_sequences_array = [np.array(seq[prediction].values) for seq in sequences]

    return np.array(x_sequences_array), np.array(y_sequences_array)


def build_experiment_descriptor_key(config):
    model_token = config["model_token"]
    size = config["size"]
    sequence_length = config["sequence_length"]
    data = crc32_columns(config["columns"])
    return f"{model_token}:{size}:{sequence_length}:{data}"


@ex.config
def config():
    model_token = None,
    size = None,
    sequence_length = None,
    columns = None


import pandas as pd


def process_dataframe(mode, file_path, all_columns, save_scaled=None):
    date_column = "Date"
    df = pd.read_csv(file_path)
    df = df[all_columns]
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    scaler = CustomScaler(config=PM_SCALER_CONFIG, df=df)
    df = scaler.fit_transform(df)

    if save_scaled:
        df.to_csv(f"{save_scaled}_{mode}.csv")

    return df


def prepare_and_train_model(model, optimizer, scheduler, train_df, sequence_length, real_model_token):
    # Define features and prediction for training
    features_train = train_df.columns.drop('Rank')
    prediction_train = ["Rank"]

    # Generate sequences
    X_train, y_train = generate_sequences(input_df=train_df,
                                          features=features_train,
                                          prediction=prediction_train,
                                          window_size=sequence_length)

    # Create tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=gbl_args.batch_size, shuffle=False)

    # Perform training
    return train_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       train_loader=train_loader,
                       patience=gbl_args.patience,
                       epochs=gbl_args.epochs,
                       model_token=real_model_token,
                       training_label="manager")


def eval_model(model, eval_df, gbl_args, sequence_length, real_model_token):
    # Define features and prediction for evaluation
    features_eval = eval_df.columns.drop('Rank')
    prediction_eval = ["Rank"]

    # Generate sequences
    X_eval, y_eval = generate_sequences(input_df=eval_df,
                                        features=features_eval,
                                        prediction=prediction_eval,
                                        window_size=sequence_length)

    # Create tensors
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32)

    # Create dataset and dataloader
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor.squeeze(-1))
    eval_loader = DataLoader(eval_dataset, batch_size=gbl_args.batch_size, shuffle=False)

    print(f"Evaluating pm: {real_model_token}")
    predictions, actuals = evaluate_model(model, eval_loader, prediction_squeeze=None)
    nrmse, mse = nrmse_by_range(actuals, predictions)
    print("pm mse: ", mse)
    print("pm nrmse: ", nrmse)

    ex.log_scalar('mse', mse)
    ex.info["model_token"] = real_model_token

    return nrmse, mse


@ex.main
def run_experiment(model_token, size, sequence_length, columns):
    # Read the training file

    column_selector = ColumnSelector(gbl_args.column_file)
    columns = sorted(column_selector.get(columns))
    all_columns = columns + ["Date", "Rank", "ID"]

    column_count = None
    ids = None
    if gbl_args.mode == "train" or gbl_args.mode == "both":
        train_df = process_dataframe("training", gbl_args.training_file, all_columns, gbl_args.save_scaled)
        column_count = len(train_df.columns) - 1
        ids = len(train_df["ID"].unique())  # number of companies

    if gbl_args.mode == "eval" or gbl_args.mode == "both":
        eval_df = process_dataframe("eval", gbl_args.eval_file, all_columns, gbl_args.save_scaled)
        column_count = len(eval_df.columns) - 1
        ids = len(eval_df["ID"].unique())  # number of companies

    assert column_count is not None, "Invalid mode. If args is working, this should never happen"
    assert ids is not None, "Invalid mode. If args is working, this should never happen"

    output = ids * sequence_length  # output should be a prediction of rank for each sequence entry
    model_sequence_length = ids * sequence_length

    model, optimizer, scheduler, real_model_token = model_from_config(
        num_features=column_count,
        config_token=model_token,
        sequence_length=model_sequence_length, size=size, output=output,
        descriptors=[
            "port_mgmt", model_token, sequence_length, size, output, crc32_columns(columns)
        ])

    # Usage with mode guard
    last_train_mse = None
    eval_mse = None
    eval_nrmse = None
    train_nrmse = None
    if gbl_args.mode == "train" or gbl_args.mode == "both":
        last_nrmse, last_train_mse = prepare_and_train_model(model=model,
                                                             optimizer=optimizer,
                                                             scheduler=scheduler,
                                                             train_df=train_df,
                                                             sequence_length=sequence_length,
                                                             real_model_token=real_model_token)
        prune_old_versions()
        if gbl_args.mode == "train":
            return {
                'eval_mse': -1,
                'train_nrmse' : last_nrmse,
                'train_mse': last_train_mse,
            }

    if gbl_args.mode == "eval" or gbl_args.mode == "both":
        eval_nrmse, eval_mse = eval_model(model=model,
                                          eval_df=eval_df,
                                          gbl_args=gbl_args,
                                          sequence_length=sequence_length,
                                          real_model_token=real_model_token)
        prune_old_versions()
        if gbl_args.mode == "eval":
            return {
                'eval_mse': eval_mse,
                'eval_nrmse': eval_nrmse,
                'train_mse': -1,
            }

    assert gbl_args.mode == "both", "something is broken, should not be here!"
    return {
        'eval_mse': eval_mse,
        'eval_nrmse': eval_nrmse,
        'train_mse': last_train_mse,
        'train_nrmse': train_nrmse
    }

def main(args):
    selector = ExperimentSelector(args.index_file, mongo=MONGO, db=DB)
    experiments = selector.get(include_ranges=args.include, exclude_ranges=args.exclude)

    # get a list of past or in flight experiments
    past_experiments = selector.get_current_state(experiment_namespace, build_experiment_descriptor_key)

    # Run the experiments using Sacred
    for experiment in experiments:
        if experiment:
            key = build_experiment_descriptor_key(experiment)

            # if we're training, we shouldn't re-train for this experiment set
            should_run = False
            if args.mode == "eval":
                should_run = True
            else:
                should_run = key not in past_experiments

            if should_run:
                print("Running key:", key)
                ex.run(config_updates=experiment)
                # update the list in case another machine is running (poor man's update)
                past_experiments = selector.get_current_state(experiment_namespace, build_experiment_descriptor_key)
            else:
                print("skipping", key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run selected experiments using Sacred.")
    parser.add_argument("--index-file", type=str, default="./metadata/mgmt-experiment-index.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--training-file", type=str, default="./results/training-pm-data.csv",
                        help="Training data for the manager")
    parser.add_argument("--eval-file", type=str, default="./results/eval-pm-data.csv",
                        help="Eval data for the manager")
    parser.add_argument("--column-file", type=str, default="./metadata/column-descriptors.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--include", type=str, default="",
                        help="Ranges of experiments to include (e.g., 1-5,10-15)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Ranges of experiments to exclude (e.g., 4-5,8)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--period", type=int, default=60,
                        help="length of training data windows")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=75,
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--metadata-path", type=str, default='./metadata', help="experiment descriptors live here")
    parser.add_argument("--mode", default="both", choices=["train", "eval", "both"], help="train eval or both")
    parser.add_argument("--save-scaled", default=None, type=str, help="For debugging save out the scaled dataset")
    gbl_args = parser.parse_args()
    main(gbl_args)

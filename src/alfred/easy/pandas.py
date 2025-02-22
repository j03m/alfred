import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alfred.data import CustomScaler, PM_SCALER_CONFIG
from alfred.model_metrics import BCEAccumulator
from alfred.model_persistence import model_from_config, crc32_columns
from alfred.model_optimization import train_model, evaluate_model
from alfred.utils import read_time_series_file


# read the datafile and scale it

def noop(df):
    return df


def prepare_data_and_model(category="easy_model",
                           model_name="vanilla",
                           model_size=256,
                           files=["data/AAPL_quarterly_directional.csv"],
                           scaler_config=PM_SCALER_CONFIG,
                           features=[],
                           labels=["PQ"],
                           batch_size=32,
                           shuffle=True,
                           date_column="Unnamed: 0",
                           augment_func=noop,
                           data_frames=None):
    print("reading input pandas")
    dfs = []
    if data_frames is None:
        for file in files:
            df = read_time_series_file(file, date_column)
            df = augment_func(df)
            dfs.append(df)

    else:
        dfs= data_frames

    df = pd.concat(dfs)

    # if no features are specified, assume it's all columns
    if len(features) == 0:
        size = len(df.columns) - len(labels)
        features = list(set(df.columns) - set(labels))
    else:
        size = len(features)

    print("loading model from config or creating model")
    model, optimizer, scheduler, scaler, real_model_token, was_loaded = model_from_config(
        num_features=size,
        config_token=model_name,
        sequence_length=-1, size=model_size, output=len(labels),
        descriptors=[
            #TODO: Im busted, features are blank!
            category, model_name, model_size, len(labels), crc32_columns(features)
        ])

    # The scaler is persisted with the model to avoid issues with distribution shift with future values
    # If we don't have a scaler that means we have a new model, make a scaler, and fit_transform
    # otherwise just transform
    if scaler is None:
        print("creating a scaler and scaling")
        scaler = CustomScaler(scaler_config, df)
        df = scaler.fit_transform(df)
    else:
        print("scaling with known scaler")
        df = scaler.transform(df)

    df.dropna(inplace=True)

    if len(features) == 0:
        features_train = df.drop(labels, axis=1)
        labels_train = df[labels]
    else:
        features_train = df[features]
        labels_train = df[labels]

    # train data prep
    x_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(labels_train.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor.squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return model, optimizer, scheduler, train_loader, real_model_token, scaler, was_loaded  # Return necessary variables


def trainer(category="easy_model",
            model_name="vanilla",
            model_size=256,
            files=["data/AAPL_quarterly_directional.csv"],
            scaler_config=PM_SCALER_CONFIG,
            epochs=5000,
            features=[],
            labels=["PQ"],
            patience=1000,
            batch_size=32,
            shuffle=True,
            date_column="Unnamed: 0",
            augment_func=noop,
            verbose=False,
            loss_function=nn.MSELoss()):

    model, optimizer, scheduler, train_loader, real_model_token, scaler, was_loaded = prepare_data_and_model(
        category=category,
        model_name=model_name,
        model_size=model_size,
        files=files,
        scaler_config=scaler_config,
        features=features,
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        date_column=date_column,
        augment_func=augment_func
    )

    print("Starting training:") # todo we need a dataclass, this param list is out of control
    return train_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       train_loader=train_loader,
                       patience=patience,
                       epochs=epochs,
                       model_token=real_model_token,
                       training_label=category,
                       verbose=verbose,
                       loss_function=loss_function,
                       scaler=scaler)


def evaler(category="easy_model",
           model_name="vanilla",
           model_size=256,
           files=["data/LNC_quarterly_directional.csv"],
           scaler_config=PM_SCALER_CONFIG,
           features=[],
           labels=["PQ"],
           batch_size=32,
           date_column="Unnamed: 0",
           augment_func=noop,
           loss_function=nn.MSELoss()):
    model, _, _, eval_loader, real_model_token, scaler, was_loaded = prepare_data_and_model(
        category=category,
        model_name=model_name,
        model_size=model_size,
        files=files,
        scaler_config=scaler_config,
        features=features,
        labels=labels,
        batch_size=batch_size,
        shuffle=False,
        date_column=date_column,
        augment_func=augment_func
    )
    if not was_loaded:
        print("WARNING: You are evaluating an empty model. This model was unknown to the system.")

    loss, data = evaluate_model(model, eval_loader, stat_accumulator=BCEAccumulator(), loss_function=loss_function)

    print(f"Evaluation: Loss: {loss} stats: {data}")

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alfred.data import CustomScaler, PM_SCALER_CONFIG
from alfred.model_metrics import BCEAccumulator
from alfred.model_persistence import model_from_config, crc32_columns
from alfred.model_optimization import train_model, evaluate_model
from alfred.utils import read_time_series_file

from dataclasses import dataclass
from typing import List, Optional, Callable, Union

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

@dataclass
class ModelPrepConfig:
    category: str = "easy_model"
    model_name: str = "vanilla"
    model_size: int = 256
    files: List[str] = None
    scaler_config: dict = None
    features: List[str] = None
    labels: List[str] = None
    batch_size: int = 32
    shuffle: bool = True
    date_column: str = "Unnamed: 0"
    augment_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    data_frames: Optional[List[pd.DataFrame]] = None
    seq_len: Optional[int] = None

    # Default initializations
    def __post_init__(self):
        self.files = self.files or ["data/AAPL_quarterly_directional.csv"]
        self.features = self.features or []
        self.labels = self.labels or ["PQ"]
        self.scaler_config = self.scaler_config or PM_SCALER_CONFIG
        self.augment_func = self.augment_func or (lambda df: df)


@dataclass
class RawModelPrepResult:
    features_train: Union[pd.DataFrame, List[pd.DataFrame]]  # Flat DataFrame or list for sequences
    labels_train: Optional[pd.DataFrame]  # None for sequential case until processed
    model: nn.Module
    optimizer: optim.Optimizer
    real_model_token: str
    scaler: CustomScaler
    scheduler: Optional[LRScheduler]
    was_loaded: bool

@dataclass
class ModelPrepResult:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[LRScheduler]
    loader: torch.utils.data.DataLoader
    dataset: torch.utils.data.Dataset
    real_model_token: str
    scaler: 'CustomScaler'
    was_loaded: bool

def noop(df):
    return df


def dfs_from_files(files, date_column="Unnamed: 0", augment_func=noop):
    dfs = []
    for file in files:
        df = read_time_series_file(file, date_column)
        df = augment_func(df)
        dfs.append(df)
    return dfs


def create_sequences(df, seq_len, features, labels):
    sequences = []
    targets = []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i + seq_len][features].values  # Shape: (seq_len, num_features)
        target = df.iloc[i + seq_len][labels].values  # Shape: (len(labels),)
        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def create_sequences_from_dfs(dfs, seq_len, features, labels, config):
    all_sequences = []
    all_targets = []
    for df in dfs:
        sequences, targets = create_sequences(df, seq_len, features, labels)
        all_sequences.append(sequences)
        all_targets.append(targets)
    x_tensor = torch.cat(all_sequences, dim=0)  # Shape: (num_samples, seq_len, num_features)
    y_tensor = torch.cat(all_targets, dim=0)    # Shape: (num_samples, len(labels))
    return x_tensor, y_tensor

def prepare_data_and_model(config: ModelPrepConfig):
    if config.seq_len is None:
        result = prepare_data_and_model_raw(config)

        # data prep
        x_tensor = torch.tensor(result.features_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(result.labels_train.values, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor.squeeze(-1))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        return ModelPrepResult(
            model=result.model,
            optimizer=result.optimizer,
            scheduler=result.scheduler,
            loader=loader,
            dataset=dataset,
            real_model_token=result.real_model_token,
            scaler=result.scaler,
            was_loaded=result.was_loaded
        )

    else:
        return prepare_data_and_model_seq(config)

def prepare_data_and_model_seq(config: ModelPrepConfig):
    if config.data_frames is None:
        dfs = dfs_from_files(config.files, config.date_column, config.augment_func)
    else:
        dfs = config.data_frames

    df_all = pd.concat(dfs, ignore_index=True)

    features, size = set_final_feature_and_size(config, df_all)

    print("loading model from config or creating model")
    features_hash = crc32_columns(features)
    model, optimizer, scheduler, scaler, real_model_token, was_loaded = model_from_config(
        num_features=size,
        config_token=config.model_name,
        sequence_length=config.seq_len, size=config.model_size, output=len(config.labels),
        descriptors=[
            config.category, config.model_name, config.model_size, len(config.labels), features_hash
        ])

    if scaler is None:
        print("creating a scaler fit, only")
        scaler = CustomScaler(config.scaler_config, df_all)
        scaler.fit(df_all)

        # Scale each individual DataFrame using the fitted scaler
        dfs_scaled = [scaler.transform(df) for df in dfs]

        # Create sequences from the scaled DataFrames
        x_tensor, y_tensor = create_sequences_from_dfs(dfs_scaled, config.seq_len, features, config.labels)

        # Create DataLoader
        dataset = TensorDataset(x_tensor, y_tensor.squeeze(-1))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        return ModelPrepResult(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            dataset=dataset,
            real_model_token=real_model_token,
            scaler=scaler,
            was_loaded=was_loaded
        )

def prepare_data_and_model_raw(config: ModelPrepConfig):
    if config.data_frames is None:
        dfs = dfs_from_files(config.files, config.date_column, config.augment_func)
    else:
        dfs = config.data_frames
    df = pd.concat(dfs)

    features, size = set_final_feature_and_size(config, df)

    print("loading model from config or creating model")
    features_hash = crc32_columns(features)
    model, optimizer, scheduler, scaler, real_model_token, was_loaded = model_from_config(
        num_features=size,
        config_token=config.model_name,
        sequence_length=-1, size=config.model_size, output=len(config.labels),
        descriptors=[
            config.category, config.model_name, config.model_size, len(config.labels), features_hash
        ])

    df, scaler = handle_scale_data(config, df, scaler)

    features_train = df[features]
    labels_train = df[config.labels]

    assert features_hash == crc32_columns(features_train.columns)
    return RawModelPrepResult(features_train=features_train,
                              labels_train=labels_train,
                              model=model,
                              optimizer=optimizer,
                              real_model_token=real_model_token,
                              scaler=scaler,
                              scheduler=scheduler,
                              was_loaded=was_loaded)


def handle_scale_data(config, df, scaler):
    # The scaler is persisted with the model to avoid issues with distribution shift with future values
    # If we don't have a scaler that means we have a new model, make a scaler, and fit_transform
    # otherwise just transform
    if scaler is None:
        print("creating a scaler and scaling")
        scaler = CustomScaler(config.scaler_config, df)
        df = scaler.fit_transform(df)
    else:
        print("scaling with known scaler")
        df = scaler.transform(df)
    df.dropna(inplace=True)
    return df, scaler


def set_final_feature_and_size(config, df):
    # if no features are specified, assume it's all columns
    if len(config.features) == 0:
        size = len(df.columns) - len(config.labels)
        features = sorted(list(set(df.columns) - set(config.labels)))
    else:
        size = len(config.features)
        features = sorted(list(set(config.features)))
    return features, size


def trainer(category="easy_model",
            model_name="vanilla",
            model_size=256,
            files=["data/AAPL_quarterly_directional.csv"],
            scaler_config=PM_SCALER_CONFIG,
            epochs=5000,
            features=[],
            labels=["PQ"],
            patience=500,
            batch_size=32,
            shuffle=True,
            date_column="Unnamed: 0",
            augment_func=noop,
            verbose=False,
            loss_function=nn.BCELoss(),
            stat_accumulator=BCEAccumulator,
            seq_len=None):

    # todo debug seq creation, try out LSTMS
    result:ModelPrepResult = prepare_data_and_model(
        ModelPrepConfig(category=category,
        model_name=model_name,
        model_size=model_size,
        files=files,
        scaler_config=scaler_config,
        features=features,
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        date_column=date_column,
        augment_func=augment_func,
        seq_len=seq_len
    ))

    print("Starting training:")  # todo we need a dataclass, this param list is out of control
    return train_model(model=result.model,
                       optimizer=result.optimizer,
                       scheduler=result.scheduler,
                       train_loader=result.loader,
                       patience=patience,
                       epochs=epochs,
                       model_token=result.real_model_token,
                       training_label=category,
                       verbose=verbose,
                       loss_function=loss_function,
                       scaler=result.scaler,
                       stat_accumulator=stat_accumulator)


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
           loss_function=nn.BCELoss(),
           stat_accumulator=BCEAccumulator()):
    result:ModelPrepResult = prepare_data_and_model(
        ModelPrepConfig(category=category,
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
    ))
    if not result.was_loaded:
        print("WARNING: You are evaluating an empty model. This model was unknown to the system.")

    loss, data = evaluate_model(result.model, result.loader, stat_accumulator=stat_accumulator, loss_function=loss_function)

    print(f"Evaluation: Loss: {loss} stats: {data}")
    return loss, data

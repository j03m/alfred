import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alfred.data import CustomScaler, PM_SCALER_CONFIG
from alfred.model_persistence import model_from_config, crc32_columns
from alfred.model_training import train_model
from alfred.utils import read_time_series_file

# read the datafile and scale it

def noop(df):
    return df

# todo test me out with a new model, am I actually easy?
# fill in some default params
def trainer(category="easy_model",
         model_name="vanilla",
         model_size=256,
         file="data/AAPL_quarterly_directional.csv",
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

    df = read_time_series_file(file,date_column)
    df = augment_func(df)
    scaler = CustomScaler(scaler_config, df)
    df = scaler.fit_transform(df)
    df.dropna(inplace=True)
    # if no features are specified, assume its all columns
    if len(features) == 0:
        features_train = df.drop(labels, axis=1)
        labels_train = df[labels]
        features = features_train.columns
    else:
        features_train = df[features]
        labels_train = df[labels]

    model, optimizer, scheduler, real_model_token = model_from_config(
        num_features=len(features),
        config_token=model_name,
        sequence_length=-1, size=model_size, output=len(labels),
        descriptors=[
            category, model_name, model_size, len(labels), crc32_columns(features)
        ])

    # train
    x_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(labels_train.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor.squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       train_loader=train_loader,
                       patience=patience,
                       epochs=epochs,
                       model_token=real_model_token,
                       training_label=category,
                       verbose=verbose,
                       loss_function=loss_function)
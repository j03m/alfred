import torch
import torch.nn as nn
import math
import devices
import pandas as pd
from model_persistence import get_latest_model, maybe_save_model
import torch.optim as optim
import torch.utils.data as data

from .datasets import SlidingWindowPandasDataset


g_tensor_board_writer = None

g_num_epochs = 2000  # Epochs
g_update_interval = 10
g_eval_save_interval = 100
g_model_size = 250

class PositionalEncoding(nn.Module):
    def __init__(self, model_size=g_model_size, max_len=5000):
        super().__init__()
        # http://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding
        X = torch.arange(max_len).reshape(-1, 1, 1)
        div_tensor = torch.pow(10000, torch.arange(0, model_size, 2).float() / model_size)
        X_ = X / div_tensor
        P = torch.zeros(max_len, 1, model_size)
        P[:, :, 0::2] = torch.sin(X_)
        P[:, :, 1::2] = torch.cos(X_)
        self.register_buffer('pe', P)

    def forward(self, x):
        # x is going to arrive here is batch, rows, features
        # which is 30, 30, 11
        return x + self.pe[:x.size(0)]


class TransformerForecaster(nn.Module):
    def __init__(self, features=11, model_size=250, nlayers=1, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(features, model_size)
        self.pos_encoder = PositionalEncoding(model_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_size, nhead=10, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.decoder = nn.Linear(model_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        output = self.decoder(x)
        return output


def train_tr_forecaster(model_path,
                     model_prefix,
                     training_data_path,
                     eval_data_path=None):
    device = devices.set_device()
    try:
        df = pd.read_csv(training_data_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("No data found, skipping")
        return

    eval_save = False
    if eval_data_path is not None:
        eval_save = True

    # todo arg us!
    data_set = SlidingWindowPandasDataset(dataframe=df,
                                          feature_columns=[
                                              "Close_diff_MA_7", "Volume_diff_MA_7", "Close_diff_MA_30", "Volume_diff_MA_30",
                                              "Close_diff_MA_90", "Volume_diff_MA_90", "Close_diff_MA_180",
                                              "Volume_diff_MA_180", "Close_diff_MA_360", "Volume_diff_MA_360", "Close"
                                          ], input_window=52, output_window=1)

    current_model = get_latest_model(model_path, model_prefix)

    # todo pass different params from args
    model = TransformerForecaster()

    if current_model is not None:
        print("found model, loading previous state.")
        model.load_state_dict(torch.load(current_model))

    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = data.DataLoader(data_set, batch_size=10)
    model.train()
    last_loss = float('inf')
    for epoch in range(g_num_epochs):
        if epoch % g_update_interval == 0:
            print("epoch: ", epoch, "last_loss: ", last_loss)

        if epoch % g_eval_save_interval == 0 and epoch >= g_eval_save_interval:
            # todo save model evaluator
            # maybe_save_model(epoch, evaluator, eval_save, model, model_path, model_prefix)
            pass

        # todo test this loop
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)

            # todo: output and y_batch don't match - have a think about what you REALLY want to predict.
            # we have weekly bars in and some bars out - maybe 4 weeks giving us advice on a 30 day position? Or
            # maybe week, 1, 4, 8, 16 bar price predictions?
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

import sys

# Mock the 'this' module to prevent it from executing
sys.modules['this'] = None

from alfred.models import Stockformer, Transformer, Informer
from alfred.devices import set_device
from alfred.model_persistence import get_latest_model, maybe_save_model
from alfred.data import DatasetStocks
from torchmetrics import R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import argparse
import warnings

# Make all UserWarnings throw exceptions
warnings.simplefilter("error", UserWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', type=str, help="training data")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size")
    parser.add_argument('--shuffle', type=bool, default=False, help="shuffle data?")
    parser.add_argument('--num-workers', type=int, default=1, help="number of workers")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="learning rate")
    parser.add_argument("--sequence-length", type=int, default=240,
                        help="sequence length, should be = max pred distance")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    parser.add_argument("--model-size", type=str, default=1024, help="model size")
    parser.add_argument("--model-token", type=str, choices=['transformer', 'stockformer', 'informer'],
                        default='transformer',
                        help="prefix used to select model architecture, also used as a persistence token to store and load models")
    parser.add_argument("--action", type=str, choices=['train', 'assess'], default='train',
                        help="train to train, assess to check the prediction value")

    args = parser.parse_args()

    data_set = DatasetStocks(args.training_file, args.sequence_length)

    device = set_device()

    if args.model_token == 'stockformer':
        model = Stockformer(
            d_model=512,
            enc_in=data_set.features,
            c_out=data_set.labels
        ).to(device)
    elif args.model_token == 'transformer':
        model = Transformer(model_dim=512, input_dim=data_set.features, output_dim=data_set.labels).to(device)
    elif args.model_token == 'informer':
        model = Informer()
    else:
        raise Exception("unknown model token")
    model_data = get_latest_model(args.model_path, args.model_token)
    if model_data is not None:
        model.load_state_dict(torch.load(model_data))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Define DataLoader
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        # num_workers=args.num_workers
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        epoch_r2 = 0.0
        total_batches = len(train_loader)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            batch_x1 = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).float().to(device)
            batch_y1 = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1]).float().to(device)

            outputs = model(batch_x1)
            loss = criterion(outputs, batch_y1)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            r2_metric = R2Score().to(device)
            flattened_predictions = outputs.view(-1)
            flattened_targets = batch_y1.view(-1)
            r2_value = r2_metric(flattened_targets, flattened_predictions)
            epoch_r2 += r2_value.item()

        avg_epoch_loss = epoch_loss / total_batches
        avg_epoch_r2 = epoch_r2 / total_batches
        print(f'Epoch [{epoch}], Avg Loss: {avg_epoch_loss}, Avg R2: {avg_epoch_r2}')
        maybe_save_model(model, avg_epoch_loss, args.model_path, args.model_token)
        scheduler.step(avg_epoch_loss)

    print('Training complete')


if __name__ == "__main__":
    main()

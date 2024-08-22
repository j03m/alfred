import sys

# Mock the 'this' module to prevent it from executing
sys.modules['this'] = None

from alfred.data import DatasetStocks
from torch import nn, optim
from torch.utils.data import DataLoader
from alfred.models import Transformer
from alfred.devices import set_device
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
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="learning rate")
    parser.add_argument("--sequence-length", type=int, default=24, help="sequence length")
    args = parser.parse_args()

    # Load dataset
    # todo change dataset or loader to deal in batches of all available stocks
    data_set = DatasetStocks(args.training_file, args.sequence_length)

    device = set_device()

    # Define model
    model = Transformer(
        enc_in=data_set.features,
        c_out=data_set.labels
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define DataLoader
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        print("epoch: ", epoch)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            batch_x1 = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).float().to(device)
            batch_y1 = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1]).float().to(device)

            _, outputs = model(batch_x1)
            loss = criterion(outputs, batch_y1)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}], Step [{i}], Loss: {loss.item() / 10:.4f}')


    print('Training complete')

if __name__ == "__main__":
    main()

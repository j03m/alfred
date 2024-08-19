from alfred.data import DatasetStocks
from torch import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', type=str, help="training data")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size")
    parser.add_argument('--shuffle', type=bool, default=False, help="shuffle data?")
    parser.add_argument('--num-workers', type=int, default=3, help="number of workers")
    parser.add_argument('--epochs', type=int, help="number of epochs")
    args = parser.parse_args()
    data_set = DatasetStocks(args.symbol_file)

    # model?

    train_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,  # Define batch size
        shuffle=args.shuffle,  # Whether to shuffle the data at every epoch
        num_workers=args.num_workers  # Number of subprocesses to use for data loading
    )
    for epoch in range(args.epochs):  # Replace num_epochs with the actual number of epochs you want
        for i, (batch_x, batch_y) in enumerate(train_loader):
            pass

if __name__ == "__main__":
    main()
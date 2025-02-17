import torch
import pandas as pd
import argparse
from alfred.metadata import TickerCategories
from alfred.model_training import train_model
from alfred.model_evaluation import evaluate_model
from alfred.data import CustomScaler, PM_SCALER_CONFIG

def train (args, tickerCategories:TickerCategories):
    training_set = tickerCategories.get("training")
    for ticker in training_set:
        df = pd.read_csv(f"./data/{ticker}_unscaled_quarterly_directional.csv")
        df = df.set_index("Date")
        scaler = CustomScaler(config=PM_SCALER_CONFIG, df=df)
        df = scaler.fit_transform(df)




def main():
    parser = argparse.ArgumentParser(description="Basic direction test")
    parser.add_argument('--file', type=str, default="./metadata/basic-tickers.json", help='ticker file')
    parser.add_argument('--mode', chocie=["train", "eval", "both"], default="both", help='mode')
    args = parser.parse_args()

    categories = TickerCategories(args.file)
    if args.mode == "both" or args.mode == "train":
        train(args, categories.get("training"))

    if args.mode == "both" or args.mode == "eval":
        eval(args, categories.get("evaluation"))

if __name__ == "__main__":
    main()
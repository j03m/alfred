from alfred.easy import evaler
import pandas as pd
import argparse

def trim_timerange(df):
    start_date = pd.Timestamp('2004-03-31T00:00:00.000000000', tz='UTC')
    return df[df.index >= start_date]


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model.")
    parser.add_argument('--model', type=str, default='vanilla.small', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=256, help='The size of the model to use')
    args = parser.parse_args()

    print("Starting easy evaler")
    evaler(augment_func=trim_timerange, model_size=args.size, model_name=args.model)

if __name__ == "__main__":
    main()

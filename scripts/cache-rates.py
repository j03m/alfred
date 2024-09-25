import pandas as pd
import os
import argparse
from alfred import data

def main(data_dir):
    alpha = data.AlphaDownloader()
    alpha.treasury_yields_to_csv(csv_file=f"{data_dir}/treasuries.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch rates.")
    parser.add_argument("--data-dir", default="./data", type=str,
                        help="Directory to look for pricing data and save output")

    args = parser.parse_args()
    main(args.data_dir)

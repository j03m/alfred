import os
from alfred.data import download_master_index, parse_master_index, generic_sec_fetch
import pandas as pd
from time import sleep

def cache_indexes(output_folder, pm_df):
    """Cache master index files for the date range specified in pm_df."""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the earliest and latest years
    pm_df['Date'] = pd.to_datetime(pm_df['Date'])
    earliest_year = pm_df['Date'].dt.year.min()
    latest_year = pm_df['Date'].dt.year.max()

    quarters = [1, 2, 3, 4]
    for year in range(earliest_year, latest_year + 1):
        for quarter in quarters:
            file_name = f"{output_folder}/index_{year}_Q{quarter}.idx"
            if not os.path.exists(file_name):
                print(f"Downloading index for {year} Q{quarter}...")
                content = download_master_index(year, quarter)
                with open(file_name, "w", encoding="latin-1") as file:
                    file.write(content)
                print(f"Saved: {file_name}")
            else:
                print(f"Index already exists: {file_name}")


def parse_indexes(output_folder, form_types=['13F-HR']):
    """Parse master index files in the output folder and filter for 13F-HR filings."""
    all_entries = []

    for file_name in os.listdir(output_folder):
        if file_name.endswith(".idx"):
            file_path = os.path.join(output_folder, file_name)
            print(f"Parsing {file_path}...")
            with open(file_path, "r", encoding="latin-1") as file:
                content = file.read()
                entries = parse_master_index(content, form_types)

            for filing in entries:
                filing_file_name = filing['filename']
                filing_file_path = os.path.join(output_folder, filing_file_name)
                if not os.path.exists(filing_file_path):
                    filing_url = f"https://www.sec.gov/Archives/{filing_file_name}"
                    print(f"Filing URL: {filing_url}")
                    content = generic_sec_fetch(filing_url)
                    sleep(0.25)
                    os.makedirs(os.path.dirname(filing_file_path), exist_ok=True)
                    with open(filing_file_path, "w", encoding="latin-1") as filing_file:
                        filing_file.write(content)


def main(output_folder="./filings", range_file="./results/pm-training-final.csv", download=False, parse=False):
    pm_df = pd.read_csv(range_file)

    # Download index files
    if download:
        print("Caching master index files...")
        cache_indexes(output_folder, pm_df)

    # Parse index files
    if parse:
        print("Parsing master index files...")
        parse_indexes(output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and parse 13F filings for tickers.")
    parser.add_argument("--output-folder", default="./filings", help="Output folder for filings.")
    parser.add_argument("--range-file", default="./results/pm-training-final.csv",
                        help="pm training set to define range to fetch.")
    parser.add_argument("--download", action="store_true", help="Download and cache index files.")
    parser.add_argument("--parse", action="store_true", help="Parse index files to extract 13F-HR filings.")

    args = parser.parse_args()

    main(args.output_folder, args.range_file, args.download, args.parse)

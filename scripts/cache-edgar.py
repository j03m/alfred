import os
from alfred.data import download_master_index, parse_master_index, generic_sec_fetch, EdgarFilingProcessor, EdgarDb
from time import sleep
from datetime import datetime


def cache_indexes(output_folder, earliest_year, latest_year):
    """Cache master index files for the date range specified in pm_df."""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the earliest and latest years

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




def parse_indexes(output_folder, dbfile, form_types=['13F-HR']):
    """Parse master index files in the output folder and filter for 13F-HR filings."""
    all_entries = []

    edb = EdgarDb(dbfile)
    efp = EdgarFilingProcessor(edb)

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
                # make a marker?
                if not edb.has_been_crawled(filing_file_name):
                    filing_url = f"https://www.sec.gov/Archives/{filing_file_name}"
                    sleep(0.05)
                    print(f"Filing URL: {filing_url}")
                    content = generic_sec_fetch(filing_url)
                    efp.processor(content)
                    edb.add_url(filing_file_name)
                else:
                    print(f"Filing record already exists: {filing_file_path}")


def main(output_folder="./filings", dbfile="data/edgar.db", start=2014, end=2025, download=True, parse=True):
    # Download index files
    if download:
        print("Caching master index files...")
        cache_indexes(output_folder, start, end)

    # Parse index files
    if parse:
        print("Parsing master index files...")
        parse_indexes(output_folder, dbfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and parse 13F filings for tickers.")
    parser.add_argument("--output-folder", default="./filings", help="Output folder for filings.")
    parser.add_argument("--start", default=2014, type=int,
                        help="start year")
    parser.add_argument("--end", default=datetime.now().year,
                        help="end year")
    parser.add_argument("--no-download", action="store_true", help="Download and cache index files.")
    parser.add_argument("--no-parse", action="store_true", help="Parse index files to extract 13F-HR filings.")
    parser.add_argument("--db-file", default="data/edgar.db", help="edgar db file")

    args = parser.parse_args()

    main(args.output_folder, args.db_file, args.start, args.end, not args.no_download, not args.no_parse)

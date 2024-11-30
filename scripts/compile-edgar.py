from alfred.utils import FileSystemCrawler
from alfred.data import AlphaDownloader
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict

alpha = AlphaDownloader()
aggregated_results = defaultdict(int)

def processor(file_path, content):
    global raw_results
    # Parse XML content
    root = ET.fromstring(content)

    # Extract the filing date
    filing_date = root.findtext("dateFiled")  # Adjust if date is elsewhere in the XML
    filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
    year, month = filing_date.year, filing_date.month

    for info_table in root.findall(".//infoTable"):
        company_name = info_table.findtext("nameOfIssuer")
        shares = int(info_table.find("shrsOrPrnAmt").findtext("sshPrnamt"))

        # Get the ticker from the company name
        ticker = alpha.get_ticker_from_name(company_name)

        if not ticker:
            print(f"Could not find ticker for company: {company_name}")
            continue

        # Update the DataFrame with month/year, ticker, shares
        aggregated_results[(year, month, ticker)] += shares


def main(data_dir, filings_folder):

    fs = FileSystemCrawler(filings_folder, processor)
    fs.crawl()

    global aggregated_results
    aggregated_list = [
        {"year": key[0], "month": key[1], "ticker": key[2], "shares": value}
        for key, value in aggregated_results.items()
    ]
    aggregated_df = pd.DataFrame(aggregated_list)

    # Save results to CSV
    output_path = f"{data_dir}/institutional_ownership.csv"
    aggregated_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and parse 13F filings for tickers.")
    parser.add_argument("--data", default="./data", help="Path to the folder containing ticker data files.")
    parser.add_argument("--filings-folder", default="./filings/edgar/data", help="Output folder for filings.")
    args = parser.parse_args()

    main(args.data, args.filings_folder)
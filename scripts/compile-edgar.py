from alfred.utils import FileSystemCrawler
from alfred.data import OpenFigiDownloader
from alfred.utils import make_datetime_index
import pandas as pd
from datetime import datetime
from lxml import etree
from collections import defaultdict
import re
from dateutil.parser import parse as parse_date
import os

open_figi_downloader = OpenFigiDownloader()
aggregated_results = defaultdict(int)
xml_parser = etree.XMLParser(recover=True)

gbl_output_path = None


def remove_namespaces(xml_content):
    # Remove namespace declarations
    xml_content = re.sub(r'\sxmlns.+(?=>)', '', xml_content, count=0)
    # Remove prefixes in tags
    xml_content = re.sub(r'(<\/?)[\w\d]+:', r'\1', xml_content)
    return xml_content


def extract_embedded_xml(content):
    # Encountering issues because the SEC filing content is not a well-formed XML document—it’s a mix of SGML-like
    # tags and embedded XML sections. Specifically, the presence of an <?xml ... ?> declaration inside the <XML> tags
    # violates XML’s well-formedness rules, causing parsing errors.
    xml_sections = re.findall(r'<XML>(.*?)</XML>', content, re.DOTALL)
    roots = {}
    for section in xml_sections:
        # Remove any embedded XML declarations
        xml_string = re.sub(r'<\?xml.*?\?>', '', section)

        # Remove namespaces to simplify XPath expressions down the road
        xml_string = remove_namespaces(xml_string)

        # Parse the XML string
        subroot = etree.fromstring(xml_string, parser=xml_parser)

        # Get the root tag name
        root_tag_name = subroot.tag

        # Store the parsed XML tree in the dictionary with the root tag name as the key
        roots[root_tag_name] = subroot

    return roots


def extract_filing_date(root):
    # Try to get <dateFiled> first
    date_filed = root.findtext("dateFiled")
    if date_filed:
        date_obj = parse_date(date_filed)
        return date_obj.year, date_obj.month

    # Fallback to <signatureDate> from <signatureBlock>
    signature_block = root.find(".//signatureBlock")
    if signature_block is not None:
        signature_date = signature_block.findtext("signatureDate")
        if signature_date:
            date_obj = parse_date(signature_date)
            return date_obj.year, date_obj.month

    raise Exception("Could not find date in dateFiled or signatureBlock")


def dump_csv():
    rows = [
        (datetime(year, month, 1), ticker, value)
        for (year, month, ticker), value in aggregated_results.items()
    ]

    df = pd.DataFrame(rows, columns=["date", "ticker", "value"])
    assert gbl_output_path is not None
    df.to_csv(str(gbl_output_path))


def processor(file_path, content):
    global raw_results

    extension = os.path.splitext(file_path)[1]
    done_file = os.path.splitext(file_path)[0] + ".done"

    if extension != ".done":
        # Parse XML content
        roots = extract_embedded_xml(content)

        # Extract the filing date
        year, month = extract_filing_date(roots['edgarSubmission'])
        table = roots.get("informationTable", None)
        if table is None:
            print(f"{file_path} does not contain informationTable")
            return

        for info_table in table.findall(".//infoTable"):
            cusip = info_table.findtext("cusip")
            company_name = info_table.findtext("nameOfIssuer")
            shares = int(info_table.find("shrsOrPrnAmt").findtext("sshPrnamt"))
            try:
                ticker = open_figi_downloader.get_ticker_for_cusip(cusip)
            except Exception as e:
                print("Failed hitting open figi:", e, " treating as non fatal")
                ticker = None

            if ticker is None or ticker == "Unknown":
                print(f"Could not find ticker for company: {cusip}/{company_name}")
                continue
            else:
                print(f"{cusip}/{company_name} maps to {ticker} for {shares} shares")

            # Update the DataFrame with month/year, ticker, shares
            aggregated_results[(year, month, ticker)] += shares

        # dump the results at every pass, because we won't process
        # old filings again
        dump_csv()
        open_figi_downloader.dump_cache()

        # Rename the file so we're not reprocessing, we can always rename back to txt to reprocess
        os.rename(file_path, done_file)
        print(f"Renamed to: {done_file}")

        # Optionally, delete the old file
        if os.path.exists(file_path):
            os.remove(file_path)
    else:
        print(f"Skipping {file_path}")


def main(data_dir, filings_folder):
    global aggregated_results
    global gbl_output_path
    # Read the CSV file into a DataFrame
    if os.path.exists(gbl_output_path):
        df = pd.read_csv(gbl_output_path)

        df['year'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.year
        df['month'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.month

        # Group by year, month, and ticker, then aggregate the 'shares' column
        aggregated_results = df.groupby(['year', 'month', 'ticker'])['value'].sum().to_dict()

    fs = FileSystemCrawler(filings_folder, processor)
    fs.crawl()

    dump_csv()
    print(f"Done. Results saved to {gbl_output_path}")


if __name__ == "__main__":
    import argparse

    # this will turn all the saved edgars from [filename].txt to [filename].done to avoid reprocessing
    # to reset run `find filings/edgar/data -name "*.done" -exec sh -c 'mv "$0" "${0%.done}.txt"' {} \;`

    parser = argparse.ArgumentParser(description="Download and parse 13F filings for tickers.")
    parser.add_argument("--data", default="./data", help="Path to the folder containing ticker data files.")
    parser.add_argument("--filings-folder", default="./filings/edgar/data", help="Output folder for final results.")
    parser.add_argument("--output-file", default="institutional_ownership.csv", help="Output file final results.")
    args = parser.parse_args()

    # global var, yea I know...booo
    gbl_output_path = os.path.join(args.data, args.output_file)
    main(args.data, args.filings_folder)

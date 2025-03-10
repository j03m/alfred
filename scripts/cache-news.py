#!/usr/bin/env python3
import argparse
from alfred.data import ArticleDownloader
from alfred.metadata import TickerCategories
from datetime import datetime, timedelta, date


def load_symbols_from_file(file):
    tickers = TickerCategories(file)
    return tickers.get(["training", "evaluation", "data"])

def date_type(date_str):
    """Convert a string in YYYY-MM-DD format to a datetime.date object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.")


def main(args):
    tickers = load_symbols_from_file(args.symbol_file)
    dl = ArticleDownloader()
    for ticker in tickers:
        end_date = args.end_date
        start_date = args.start_date
        dl.cache_article_metadata(ticker, start_date, end_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--start-date',
                        type=date_type,
                        default=date(2004, 3, 31),
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                        type=date_type,
                        default=datetime.now().date(),
                        help='Maximum date for timerange trimming (YYYY-MM-DD)')
    args = parser.parse_args()
    main(args)

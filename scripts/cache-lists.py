#!/usr/bin/env python3
import argparse
import requests
from bs4 import BeautifulSoup
import csv
import re

def list_indexes():
    url = "https://uk.finance.yahoo.com/world-indices/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')
    if not table:
        print("Failed to find indices table on Yahoo Finance.")
        return

    indices = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) >= 2:
            symbol = cells[0].text.strip()
            name = cells[1].text.strip()
            indices[symbol] = name

    for symbol, name in indices.items():
        print(f"{symbol}: {name}")


index_urls = {
    "^FTSE": "https://en.wikipedia.org/wiki/FTSE_100_Index",  # FTSE 100
    "^GSPC": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",  # S&P 500
    "^DJI": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",  # Dow Jones Industrial Average
    "^IXIC": "https://en.wikipedia.org/wiki/NASDAQ_Composite",  # NASDAQ Composite
    "^GDAXI": "https://en.wikipedia.org/wiki/DAX",  # DAX PERFORMANCE-INDEX
    "^FCHI": "https://en.wikipedia.org/wiki/CAC_40",  # CAC 40
    "^N225": "https://en.wikipedia.org/wiki/Nikkei_225",  # Nikkei 225
    "^HSI": "https://en.wikipedia.org/wiki/Hang_Seng_Index",  # HANG SENG INDEX
    "000001.SS": "https://en.wikipedia.org/wiki/SSE_Composite_Index",  # SSE Composite Index
    "^AXJO": "https://en.wikipedia.org/wiki/S%26P/ASX_200",  # S&P/ASX 200
    "^GSPTSE": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",  # S&P/TSX Composite index
    "^RUT": "https://en.wikipedia.org/wiki/Russell_2000_Index",  # Russell 2000
    "^VIX": "https://en.wikipedia.org/wiki/VIX",  # CBOE Volatility Index
    "^STOXX50E": "https://en.wikipedia.org/wiki/EURO_STOXX_50",  # ESTX 50 PR.EUR
    "^N100": "https://en.wikipedia.org/wiki/Euronext_100",  # Euronext 100 Index
    "^BFX": "https://en.wikipedia.org/wiki/BEL20",  # BEL 20
    "IMOEX.ME": "https://en.wikipedia.org/wiki/MOEX_Russia_Index",  # MOEX Russia Index
    "^NYA": "https://en.wikipedia.org/wiki/NYSE_Composite",  # NYSE COMPOSITE (DJ)
    "^XAX": "https://en.wikipedia.org/wiki/NYSE_American",  # NYSE AMEX COMPOSITE INDEX
    "^STI": "https://en.wikipedia.org/wiki/Straits_Times_Index",  # STI Index
    "^BSESN": "https://en.wikipedia.org/wiki/BSE_SENSEX",  # S&P BSE SENSEX
    "^JKSE": "https://en.wikipedia.org/wiki/IDX_Composite",  # IDX COMPOSITE
    "^KLSE": "https://en.wikipedia.org/wiki/FTSE_Bursa_Malaysia_KLCI",  # FTSE Bursa Malaysia KLCI
    "^NZ50": "https://en.wikipedia.org/wiki/NZX_50_Index",  # S&P/NZX 50 INDEX GROSS
    "^KS11": "https://en.wikipedia.org/wiki/KOSPI",  # KOSPI Composite Index
    "^TWII": "https://en.wikipedia.org/wiki/Taiwan_Capitalization_Weighted_Stock_Index",  # TSEC weighted index
    "^BVSP": "https://en.wikipedia.org/wiki/Ibovespa",  # IBOVESPA
    "^MXX": "https://en.wikipedia.org/wiki/Índice_de_Precios_y_Cotizaciones",  # IPC MEXICO
    "^IPSA": "https://en.wikipedia.org/wiki/IPSA",  # S&P IPSA
    "^MERV": "https://en.wikipedia.org/wiki/MERVAL",  # MERVAL
    "^TA125.TA": "https://en.wikipedia.org/wiki/Tel_Aviv_125",  # TA-125
    "^CASE30": "https://en.wikipedia.org/wiki/EGX_30",  # EGX 30 Price Return Index
    "^NSEI": "https://en.wikipedia.org/wiki/NIFTY_50"  # NIFTY 50
}


def scrape_constituents(index, url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    constituents = []

    # Find the first table on the Wikipedia page
    table = soup.find('table', {'class': 'wikitable sortable'})
    if table:
        headers = [th.text.strip() for th in table.find('tr').find_all('th')]
        # Find the index of the column with 'Ticker', 'Symbol', or 'Ticker Symbol'
        column_index = None
        for header in ['Ticker', 'Symbol', 'Ticker Symbol']:
            if header in headers:
                column_index = headers.index(header)
                break
        if column_index is None:
            raise ValueError("No 'Ticker', 'Symbol', or 'Ticker Symbol' column found in the table")

        rows = table.find_all('tr')[1:]  # skip header row
        for row in rows:
            cells = row.find_all('td')
            if cells:
                ticker = cells[column_index].text.strip()
                # Remove any exchange prefix from the ticker symbol
                ticker = re.sub(r'^\w+:(\w{2,4})$', r'\1', ticker)
                constituents.append(ticker)

    return constituents

def save_to_csv(index, constituents, dir):
    filename = f"{dir}/{index}_constituents.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbols'])
        for constituent in constituents:
            writer.writerow([constituent])
    print(f"Saved constituents of {index} to {filename}")

parser = argparse.ArgumentParser(description="Cache lists of index constituents from Yahoo Finance")
parser.add_argument("-l", "--list", action='store_true', help="List all available indexes")
parser.add_argument("-i", "--index", type=str, help="Index symbol to cache constituents for")
parser.add_argument("-d", "--directory", default="./lists", help="Directory to save the cached CSV file")
parser.add_argument("-dr", "--dry-run", action='store_true', help="Perform a dry run to list index constituents without downloading")

args = parser.parse_args()

if args.list:
    list_indexes()
elif args.index:
    if args.index not in index_urls:
        print(f"I don't know {args.index} sorry :(")
    elif args.dry_run:
        members = scrape_constituents(args.index, index_urls[args.index])
        for member in members:
            print(member)
    else:
        members = scrape_constituents(args.index, index_urls[args.index])
        save_to_csv(args.index, members, args.directory)

else:
    print("Requires --index to download or --dry-run to list members or --list to show available lists")
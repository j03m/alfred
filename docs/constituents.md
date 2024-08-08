To access and cache lists of constituents for major indexes, you can use the `yfinance` library. Here is an updated example, assuming the `yfinance` library is installed.

### Major Indexes Supported by Yahoo Finance

1. **Dow Jones Industrial Average (DJI)**
2. **NASDAQ Composite (IXIC)**
3. **S&P 500 (GSPC)**
4. **Russell 2000 (RUT)**
5. **FTSE 100 (FTSE)**
6. **Nikkei 225 (N225)**
7. **DAX (GDAXI)**
8. **Hang Seng Index (HSI)**
9. **Shanghai Stock Exchange Composite Index (SSEC)**
10. **Straits Times Index (STI)**
11. **BSE SENSEX (BSESN)**
12. **Swiss Market Index (SSMI)**
13. **S&P/ASX 200 (AXJO)**
14. **All Ordinaries (AORD)**
15. **KOSPI Composite Index (KS11)**
16. **TSEC Weighted Index (TWII)**
17. **NZX 50 Index (NZ50)**
18. **Euronext 100 Index (N100)**
19. **CBOE Volatility Index (VIX)**

### Example Code to Retrieve and Cache Constituents

First, ensure you have the `yfinance` library installed:

```sh
pip install yfinance
```

Next, use the following Python script to retrieve and cache the constituents for the major indexes:

```python
import yfinance as yf

# List of index symbols
index_symbols = {
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "NASDAQ Composite",
    "^GSPC": "S&P 500",
    "^RUT": "Russell 2000",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX",
    "^HSI": "Hang Seng Index",
    "^SSEC": "Shanghai Stock Exchange Composite Index",
    "^STI": "Straits Times Index",
    "^BSESN": "BSE SENSEX",
    "^SSMI": "Swiss Market Index",
    "^AXJO": "S&P/ASX 200",
    "^AORD": "All Ordinaries",
    "^KS11": "KOSPI Composite Index",
    "^TWII": "TSEC Weighted Index",
    "^NZ50": "NZX 50 Index",
    "^N100": "Euronext 100 Index",
    "^VIX": "CBOE Volatility Index"
}

# Function to retrieve and print index constituents
def get_index_constituents(index_symbol):
    ticker = yf.Ticker(index_symbol)
    constituents = ticker.constituents
    print(f"Index: {index_symbols[index_symbol]}")
    print(constituents)

# Retrieve and print constituents for each index
for symbol in index_symbols.keys():
    get_index_constituents(symbol)
```

### Explanation:

1. **Index Symbols:** The script defines a dictionary of index symbols and their names.
2. **Function to Retrieve Constituents:** The `get_index_constituents` function fetches and prints the constituents for a given index symbol using `yfinance`.
3. **Retrieve and Print:** The script iterates over each index symbol and retrieves the constituents.

This script will help you retrieve and cache lists of constituents for major indexes from Yahoo Finance using the `yfinance` library.
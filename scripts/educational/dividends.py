import yfinance as yf

# Fetch the ticker
ticker = yf.Ticker("AAPL")

# Get dividend history
dividends = ticker.dividends

print(dividends)
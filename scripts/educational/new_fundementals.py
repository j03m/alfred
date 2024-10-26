from alfred.data import AlphaDownloader
symbol = 'AAPL'  # Example symbol
alpha = AlphaDownloader()

# Fetch balance sheet and cash flow data
balance_sheet_df = alpha.balance_sheet(symbol)
cash_flow_df = alpha.cash_flow(symbol)

# Initialize lists to store results
dates = []
fcf_yields = []
price_to_books = []

for date in cash_flow_df.index:
    try:
        # Free Cash Flow Yield calculation
        operating_cash_flow = float(cash_flow_df.loc[date]['operatingCashflow'])
        capital_expenditure = float(cash_flow_df.loc[date]['capitalExpenditures'])
        free_cash_flow = operating_cash_flow - capital_expenditure

        # Market Cap approximation
        # You'll need to fetch stock price data separately or approximate it here
        # Example assumes recent price * shares outstanding (if available)

        # Price-to-Book Value calculation
        book_value = float(balance_sheet_df.loc[date]['totalShareholderEquity'])
        price_to_book = market_cap / book_value

        # Store results
        dates.append(date)
        fcf_yields.append(free_cash_flow / market_cap)
        price_to_books.append(price_to_book)

    except KeyError:
        continue

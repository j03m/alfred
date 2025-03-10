from pytickersymbols import PyTickerSymbols

stock_data = PyTickerSymbols()
countries = stock_data.get_all_countries()
indices = stock_data.get_all_indices()
industries = stock_data.get_all_industries()
nasdaq_tickers = stock_data.get_nasdaq_nyc_yahoo_tickers()
pass
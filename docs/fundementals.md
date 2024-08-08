To access the fundamental data from Yahoo Finance, you can use the `yfinance` Python library. Below, I will provide an example of how to access these fundamentals using the `yfinance` library.

First, you need to install the `yfinance` library if you haven't already:

```sh
pip install yfinance
```

Then, you can use the following code to access the specified fundamentals:

```python
import yfinance as yf

# Ticker symbol
ticker = 'AAPL'  # Example with Apple Inc.

# Download the ticker data
stock = yf.Ticker(ticker)

# Retrieve the quarterly financials
quarterly_financials = stock.quarterly_financials
quarterly_balance_sheet = stock.quarterly_balance_sheet

# Display the quarterly financials
print("Quarterly Financials:")
print(quarterly_financials)

# Display the quarterly balance sheet
print("Quarterly Balance Sheet:")
print(quarterly_balance_sheet)

# Accessing specific fields
fundamentals = {
    'oiadpq': quarterly_financials.loc['Operating Income'],
    'revtq': quarterly_financials.loc['Total Revenue'],
    'niq': quarterly_financials.loc['Net Income'],
    'atq': quarterly_balance_sheet.loc['Total Assets'],
    'teqq': quarterly_balance_sheet.loc["Total Stockholder Equity"],
    'epspiy': stock.quarterly_earnings.loc['Earnings'],
    'ceqq': quarterly_balance_sheet.loc['Common Stock Equity'],
    'cshoq': stock.shares,
    'dvpspq': stock.dividends,
    'actq': quarterly_balance_sheet.loc['Total Current Assets'],
    'lctq': quarterly_balance_sheet.loc['Total Current Liabilities'],
    'cheq': quarterly_balance_sheet.loc['Cash And Cash Equivalents'],
    'rectq': quarterly_balance_sheet.loc['Net Receivables'],
    'cogsq': quarterly_financials.loc['Cost Of Revenue'],
    'invtq': quarterly_balance_sheet.loc['Inventory'],
    'apq': quarterly_balance_sheet.loc['Accounts Payable'],
    'dlttq': quarterly_balance_sheet.loc['Long Term Debt'],
    'dlcq': quarterly_balance_sheet.loc['Current Portion of Long Term Debt'],
    'ltq': quarterly_balance_sheet.loc['Total Liabilities']
}

# Print the accessed fundamentals
for key, value in fundamentals.items():
    print(f"{key}: {value}")

```

### Explanation of the Fundamentals:

1. **Operating Income (oiadpq)**: The profit a company makes from its operations.
2. **Quarterly Revenue (revtq)**: Total income from sales and services.
3. **Net Income (niq)**: Profit after all expenses have been deducted from revenues.
4. **Total Assets (atq)**: Sum of all assets owned by the company.
5. **Shareholder's Equity (teqq)**: The residual interest in the assets after deducting liabilities.
6. **EPS including Extraordinary Items (epspiy)**: Earnings per share, including unusual or infrequent items.
7. **Common Equity (ceqq)**: The value of equity attributable to common shareholders.
8. **Common Shares Outstanding (cshoq)**: Number of shares that are currently owned by shareholders.
9. **Dividends per Share (dvpspq)**: The amount of dividends paid out for each share.
10. **Current Assets (actq)**: Assets that are expected to be converted to cash within a year.
11. **Current Liabilities (lctq)**: Obligations the company is expected to pay within a year.
12. **Cash & Equivalents (cheq)**: Liquid assets that can be quickly converted to cash.
13. **Receivables (rectq)**: Money owed to the company by customers.
14. **Cost of Goods Sold (cogsq)**: Direct costs attributable to the production of goods sold.
15. **Inventories (invtq)**: Raw materials, work-in-progress, and finished goods.
16. **Accounts Payable (apq)**: Money the company owes to suppliers.
17. **Long Term Debt (dlttq)**: Loans and financial obligations lasting over one year.
18. **Debt in Current Liabilities (dlcq)**: Portion of debt that is due within a year.
19. **Total Liabilities (ltq)**: The company's total debt and financial obligations.

You can customize the `ticker` variable to any stock symbol you are interested in. The `yfinance` library will fetch the relevant data, and you can then access and print the specific fundamentals as shown. Note that not all fields may be available for every company, and some fields may have different names or structures in the data.
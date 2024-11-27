import requests
from edgar import Company

def get_company_info(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Joe Banjo (jbanjo@gmail.com)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        ticker = ticker.upper()
        for item in data.values():
            if item['ticker'] == ticker:
                cik = str(item['cik_str']).zfill(10)
                return item['title'], cik
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    return None, None

# Example usage
ticker = "AAPL"
company_name, cik = get_company_info(ticker)
if company_name and cik:
    print(f"Company Name: {company_name}, CIK: {cik}")
else:
    print("Ticker not found.")


company = Company(company_name, cik)

# Get 13F filings for a specific year and quarter
filings = company.get_all_filings(filing_type="13F-HR")

print(filings)
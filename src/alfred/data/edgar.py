import requests
import re
import time


def make_sec_header(key_file="./keys/id.txt"):
    with open(key_file, 'r') as file:
        id = file.readline().strip()

    return {
        "User-Agent": id
    }


def get_company_info(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = make_sec_header()
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        ticker = ticker.upper()
        for item in data.values():
            if item['ticker'] == ticker:
                cik = str(item['cik_str']).zfill(10)
                return item['title'], cik
    else:
        raise Exception(f"Failed to retrieve company info: {response.status_code}")


def download_master_index(year, quarter):
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    headers = make_sec_header()
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.content.decode('latin-1')
        return content
    else:
        print(f"Failed to download index file for {year} {quarter}: {response.status_code}")


def generic_sec_fetch(url, retries=3):
    headers = make_sec_header()
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                content = response.content.decode('latin-1')
                return content
            else:
                print(f"Attempt {attempt + 1} failed to fetch {url}: {response.status_code}")
                time.sleep(0.25)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to fetch {url}: with exception {e}")
            time.sleep(0.25)

    print(f"Failed to fetch {url} after {retries} attempts.")
    return None


# todo fix me
def parse_master_index(content, form_types=['13F-HR']):
    lines = content.splitlines()
    entries = []
    start_parsing = False
    separator_pattern = re.compile(r'[-]+')
    for line in lines:
        if start_parsing:
            parts = line.strip().split('|')
            if len(parts) == 5:
                cik, company_name, form_type, date_filed, filename = parts
                if form_type in form_types:
                    entries.append({
                        'cik': cik,
                        'company_name': company_name,
                        'form_type': form_type,
                        'date_filed': date_filed,
                        'filename': filename
                    })
        elif separator_pattern.fullmatch(line.strip()):
            start_parsing = True
    return entries


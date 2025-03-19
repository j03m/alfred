import requests
import time
from warcio.archiveiterator import ArchiveIterator
from io import BytesIO
import gzip


# Generator to fetch pages from Common Crawl Index API
def fetch_common_crawl_pages(index_name, url_pattern, sleep_interval):
    """
    Fetch paginated JSON data from Common Crawl Index API for a URL pattern.

    Args:
        index_name (str): The index name, e.g., 'CC-MAIN-2025-08'.
        url_pattern (str): The URL pattern with wildcards, e.g., '*.example.com/*'.
        sleep_interval (float): Seconds to wait between requests.

    Yields:
        list: A list of records for each page.
    """
    base_url = f"https://index.commoncrawl.org/{index_name}-index"
    params = {"url": url_pattern, "output": "json", "page": 1}

    # Fetch the first page
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}")
        return

    data = response.json()
    total_pages = data.get("pages", 0)
    if total_pages == 0:
        return

    yield data["results"]

    # Fetch subsequent pages
    for page in range(2, total_pages + 1):
        time.sleep(sleep_interval)
        params["page"] = page
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Request failed with status {response.status_code}")
            return
        yield response.json()["results"]


# Function to fetch HTML from WARC record
def fetch_html_from_warc(record):
    """
    Fetch and return the HTML content from a WARC record using its metadata.

    Args:
        record (dict): A dictionary containing 'filename', 'offset', 'length', etc.

    Returns:
        str: The HTML content of the page, or None if an error occurs.
    """
    filename = record["filename"]
    offset = int(record["offset"])
    length = int(record["length"])

    warc_url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    response = requests.get(warc_url, headers=headers)
    if response.status_code != 206:
        print(f"Failed to fetch range for {record['url']}: {response.status_code}")
        return None

    try:
        decompressed_data = gzip.decompress(response.content)
    except Exception as e:
        print(f"Error decompressing data for {record['url']}: {e}")
        return None

    stream = BytesIO(decompressed_data)
    for warc_record in ArchiveIterator(stream):
        if warc_record.rec_type == "response":
            return warc_record.content_stream().read().decode("utf-8", errors="ignore")
    return None


# Main processing loop
index_name = "CC-MAIN-2025-08"
url_pattern = "*.example.com/*"
sleep_interval = 1.0

for page_results in fetch_common_crawl_pages(index_name, url_pattern, sleep_interval):
    print("Processing a new page of results...")
    for record in page_results:
        print(f"Fetching HTML for {record['url']}...")
        html = fetch_html_from_warc(record)
        if html:
            print(f"HTML content (first 100 chars): {html[:100]}...")
        else:
            print(f"Could not retrieve HTML for {record['url']}.")
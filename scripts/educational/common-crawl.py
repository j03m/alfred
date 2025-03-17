from comcrawl import IndexClient
from urllib.parse import urlparse
import os

client = IndexClient()
client.search("www.pro-football-reference.com/*")
client.download()

output_dir = "./webarchive"

for result in client.results:
    url = result["url"]
    html = result["html"]
    parsed = urlparse(url)
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    file_path = os.path.join(output_dir, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
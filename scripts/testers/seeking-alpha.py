from playwright.sync_api import sync_playwright
import time
import os


# Function to extract and save content
def save_content(title, url, page, delay=5):
    # Visit the article page
    page.goto(url)

    # Give some time for the page to load
    time.sleep(delay)

    # Extract the content of the article
    content = page.locator('div[data-test-id="article-content"]').text_content()

    # Save content to a file
    filename = title.replace(" ", "_").replace("/", "_") + ".txt"
    with open(os.path.join("saved_articles", filename), "w", encoding="utf-8") as f:
        f.write(content)

    # Wait before processing the next article
    time.sleep(delay)


# Make sure the directory exists
if not os.path.exists("saved_articles"):
    os.makedirs("saved_articles")

with sync_playwright() as p:
    # Launch the browser in non-headless mode if you want to debug
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Go to the Seeking Alpha AAPL analysis page
    page.goto("https://seekingalpha.com/symbol/FBCV")

    # Wait for the headlines to load
    page.wait_for_selector('a[data-test-id="post-list-title"]')

    # Extract the headlines
    headlines = page.locator('a[data-test-id="post-list-title"]').all()

    for headline in headlines:
        title = headline.text_content().strip()
        link = headline.get_attribute('href')

        # Form the full URL and save content
        full_link = "https://seekingalpha.com" + link
        save_content(title, full_link, page)

    browser.close()

import os
import pandas as pd
import time
import ssl
from datetime import datetime
import binascii
from io import StringIO
import requests
import queue
import threading
import concurrent.futures

from .news_db import NewsDb
from .openai_query import OpenAiQuery
from fuzzywuzzy import process
import re
import json
import requests
from time import sleep
from fake_useragent import UserAgent

from alfred.utils import print_in_place
from downloaders import AlphaDownloader

ssl._create_default_context = ssl._create_unverified_context

SENTINEL = object()  # Unique object to signal the end of processing


class MultiArticleDownloader:
    def __init__(self, workers=5, rate_limit=0.5):
        self.api = AlphaDownloader()
        self.openai = OpenAiQuery()
        self.news_db = NewsDb()
        self.rate_limit = rate_limit
        self.workers = 5
        self.ua = UserAgent()
        self.url_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=100)  # Bounded queue to limit memory usage

    def get(self, url):
        """Fetch a URL with rate limiting."""
        sleep(self.rate_limit)
        return requests.get(url, verify=False)

    def fetch_article_body(self, url):
        """Fetch the article body from a URL without rate limiting (not hitting Alpha API)."""
        headers = {
            'User-Agent': self.ua.random
        }
        response = requests.get(url, headers=headers, timeout=1)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.text

    def generate_article_id(self, url):
        """Generate a unique article ID using CRC32 hash of the URL."""
        return format(binascii.crc32(url.encode()), '08x')

    def cache_article_metadata(self, ticker, time_from, time_to):
        """Cache metadata for articles within a time window for a given ticker."""
        # Check if news already exists and adjust the time window
        # There is open question if sqlite will lock up here - could be.
        has_news, latest = self.news_db.has_news(ticker)
        if has_news:
            if latest < time_to:
                time_from = latest
            else:
                print("skipping: ", ticker)
                return

        # Fetch articles using the Alpha API
        articles = self.api.news_sentiment_for_window_and_symbol(ticker, time_from, time_to)
        self.fetch_articles(articles, ticker)

    def fetch_articles(self, articles, ticker):
        """Fetch articles and process them concurrently."""
        total = len(articles)
        # Populate the URL queue with articles
        for i, article in enumerate(articles):
            print_in_place(f"Fetching article: {i} of {total} for {ticker}")
            published = article.get('time_published', None)
            if published is None:
                published = datetime.today().date().strftime('%Y-%m-%d')
            else:
                published = datetime.strftime(published, '%Y-%m-%d')
            self.url_queue.put((article, published, ticker))

        # Start the saving thread to process results as they come
        results_thread = threading.Thread(target=self.save)
        results_thread.start()

        # Process articles with 4 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            for _ in range(self.workers):
                executor.submit(self.process_article)

        # Wait for all articles to be processed
        self.url_queue.join()

        # Signal the saving thread to stop
        self.result_queue.put(SENTINEL)

        # Wait for the saving thread to finish
        results_thread.join()

    def process_article(self):
        """
            Worker function to process articles from the URL queue.
            Runs until the url queue is empty
        """
        while True:
            try:
                # Get an article from the queue with a timeout
                article, published, ticker = self.url_queue.get(timeout=1)
                try:
                    if not self.news_db.has_article(ticker, article["url"]):
                        # Fetch and process the article if it doesn't exist in the DB
                        body = self.fetch_article_body(article["url"])
                        metadata = self.get_metadata(ticker, body)
                        self.result_queue.put((published, ticker, article["url"], metadata["relevance"],
                                               metadata["sentiment"], metadata["outlook"]))
                    else:
                        # Skip if article already exists
                        pass
                except Exception as e:
                    # Use fallback values if fetching or processing fails
                    self.result_queue.put((published, ticker, article["url"], article['relevance_score'],
                                           article['ticker_sentiment_score'],
                                           1 if article['ticker_sentiment_label'] == 'Bullish' else 0))
                finally:
                    # Mark the task as done
                    self.url_queue.task_done()
            except queue.Empty:
                # Exit when the queue is empty
                break

    def save(self):
        """
            Consumer function to save results from the result queue to the database.
            Runs against the results queue until it gets a SENTINEL item and then breaks
        """
        while True:
            item = self.result_queue.get()
            if item is SENTINEL:
                # Stop when the sentinel is received
                break
            # Unpack and save the result
            date, ticker, url, relevance, sentiment, outlook = item
            self.news_db.save(date, ticker, url, relevance, sentiment, outlook)
            self.result_queue.task_done()

    def get_metadata(self, ticker, body):
        """Extract metadata (relevance, sentiment, outlook) from article body."""
        return self.openai.news_query(body, ticker)
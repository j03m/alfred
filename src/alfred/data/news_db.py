import sqlite3
import pandas as pd
from datetime import datetime


class NewsDb:
    def __init__(self, db_path='data/news.db'):
        """Initialize the news database and create the table if it doesn't exist."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Create the news table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                date TEXT,          -- Date as ISO format (e.g., '2025-03-10')
                ticker TEXT,        -- Ticker symbol (e.g., 'AAPL')
                url TEXT,           -- URL of the article
                relevance REAL,     -- Relevance score (float)
                sentiment REAL,     -- Sentiment score (float)
                outlook REAL,       -- Outlook score (float)
                PRIMARY KEY (ticker, url)  -- Unique combo of ticker and url
            );
        ''')

        # Optional index on ticker for faster lookups (PRIMARY KEY already indexes ticker, url)
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON news(ticker);')

        self.conn.commit()

    def save(self, date, ticker, url, relevance, sentiment, outlook):
        # Insert or ignore if the (ticker, url) pair already exists
        self.cursor.execute('''
            INSERT OR IGNORE INTO news (date, ticker, url, relevance, sentiment, outlook)
            VALUES (?, ?, ?, ?, ?, ?);
        ''', (date, ticker, url, relevance, sentiment, outlook))

        self.conn.commit()

    def has_article(self, ticker, url):
        """Check if an article for this ticker and URL already exists in the database."""
        self.cursor.execute('''
            SELECT 1 FROM news WHERE ticker = ? AND url = ?;
        ''', (ticker, url))

        return self.cursor.fetchone() is not None

    def __del__(self):
        """Close the database connection when the object is destroyed."""
        self.conn.close()


    def get_summary(self, ticker, relevance=0.7, start_date=None, end_date=None):
        # Base query
        query = '''
            SELECT date AS Date,
                   AVG(sentiment) AS mean_sentiment,
                   AVG(outlook) AS mean_outlook
            FROM news
            WHERE ticker = ? AND relevance >= ?
        '''
        params = [ticker, relevance]

        # Add date range filtering if provided
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)

        # Group by date
        query += ' GROUP BY date HAVING COUNT(*) > 0;'

        # Execute query and load into DataFrame
        df = pd.read_sql_query(query, self.conn, params=tuple(params))

        # Return None if no data
        if df.empty:
            return None

        # Convert Date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df[['mean_sentiment', 'mean_outlook']]

        return df

    def has_news(self, ticker):
        """
        Check if the supplied ticker has news articles in the system for the current month and year.

        :param ticker: The stock ticker symbol to check.
        :return: A tuple (has_news, latest_date, today)
                 - has_news: bool, True if there is at least one news article in the current month
                 - latest_date: str or None, the latest date in 'YYYY-MM-DD' format when news is available in the current month, or None if no news
                 - today: str, today's date in 'YYYY-MM-DD' format
        """
        # Get the current date and define the current month's range
        current_date = datetime.now()
        first_day = current_date.replace(day=1)
        first_day_str = first_day.strftime('%Y-%m-%d')

        # Query the latest date for the ticker in the current month
        self.cursor.execute('''
               SELECT MAX(date) FROM news
               WHERE ticker = ? AND date <= ?
           ''', (ticker, first_day_str))

        # Fetch the result
        result = self.cursor.fetchone()
        latest_date = result[0]  # Will be a date string or None
        has_news = latest_date is not None
        if latest_date is not None:
            latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
        # Return the tuple
        return has_news, latest_date

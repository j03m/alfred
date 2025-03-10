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

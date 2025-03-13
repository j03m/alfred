import sqlite3
import pandas as pd

from alfred.utils import make_datetime_index


class EdgarDb:
    def __init__(self, db_file="./data/edgar.db"):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS crawled_urls (url TEXT PRIMARY KEY);')
        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS filings (
                        year INTEGER,
                        month INTEGER,
                        ticker TEXT,
                        shares REAL,
                        PRIMARY KEY (year, month, ticker)
                    );
                ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON filings(ticker);')

    def update(self, year, month, ticker, shares):
        self.cursor.execute('''
            INSERT INTO filings (year, month, ticker, shares)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(year, month, ticker) DO UPDATE SET
            shares = shares + excluded.shares;
        ''', (year, month, ticker, shares))
        self.conn.commit()

    import pandas as pd

    def get_filings(self, ticker):
        query = 'SELECT year, month, shares FROM filings WHERE ticker = ?'
        df = pd.read_sql_query(query, self.conn, params=[ticker])
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.drop(['year', 'month'], axis=1)
        df = make_datetime_index(df, "date")
        df.index = df.index.tz_localize(None)
        df = df.rename(columns={'shares': 'Institutional'})
        return df

    def add_url(self, url):
        self.cursor.execute('INSERT OR IGNORE INTO crawled_urls VALUES (?)', (url,))
        self.conn.commit()

    # Check if URL has been crawled
    def has_been_crawled(self, url):
        self.cursor.execute('SELECT 1 FROM crawled_urls WHERE url = ?', (url,))
        return self.cursor.fetchone() is not None

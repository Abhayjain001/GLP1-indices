import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
from config import STOCK_TICKERS, INCEPTION_DATE

DB_PATH = 'history.db'

def get_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initializes the database and creates the price_history table if it doesn't exist."""
    print("Initializing database...")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                adj_close REAL NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        ''')
        conn.commit()
    print("Database initialized successfully.")

def get_latest_date_for_ticker(ticker):
    """Gets the most recent date for a given ticker from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM price_history WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        return pd.to_datetime(result[0]) if result and result[0] else None

def insert_prices(ticker, data_df):
    """
    Inserts or updates historical price data for a ticker.
    The DataFrame should have 'Date' and 'Adj Close' columns.
    """
    if data_df.empty:
        return

    with get_connection() as conn:
        cursor = conn.cursor()

        # Prepare data for insertion
        data_tuples = [
            (ticker, date.strftime('%Y-%m-%d'), row['Adj Close'])
            for date, row in data_df.iterrows()
        ]

        # Use INSERT OR REPLACE to handle duplicates (e.g., on re-run)
        cursor.executemany(
            "INSERT OR REPLACE INTO price_history (ticker, date, adj_close) VALUES (?, ?, ?)",
            data_tuples
        )
        conn.commit()
        print(f"✓ DB: Inserted/updated {len(data_tuples)} records for {ticker}")

def backfill_all_tickers():
    """
    Performs a one-time backfill of historical data for all tickers.
    Fetches data from the inception date to today.
    """
    print("Starting historical data backfill...")
    end_date = datetime.today()

    # Also include SPY for benchmark comparison
    all_tickers_to_backfill = STOCK_TICKERS + ['SPY']

    for ticker in all_tickers_to_backfill:
        try:
            latest_db_date = get_latest_date_for_ticker(ticker)

            # If we have data, start from the next day. Otherwise, from inception.
            start_date = latest_db_date + pd.Timedelta(days=1) if latest_db_date else pd.to_datetime(INCEPTION_DATE)

            if start_date.date() >= end_date.date():
                print(f"✓ DB: {ticker} is already up-to-date.")
                continue

            print(f"Backfilling {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False) # Use auto_adjust=False to get 'Adj Close'

            if not hist.empty and 'Adj Close' in hist.columns:
                # Select and rename columns to match our needs
                hist_df = hist[['Adj Close']].copy()
                hist_df = hist_df.dropna() # Drop rows with missing values
                insert_prices(ticker, hist_df)
            else:
                print(f"✗ DB: No new data found for {ticker}")

        except Exception as e:
            print(f"✗ DB Error backfilling {ticker}: {e}")
            continue

    print("Historical data backfill complete.")

def get_price_history(ticker, start_date, end_date):
    """
    Retrieves price history for a ticker between two dates.
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, adj_close FROM price_history WHERE ticker = ? AND date BETWEEN ? AND ?",
            conn,
            params=(ticker, start_date, end_date),
            parse_dates=['date'],
            index_col='date'
        )
        return df
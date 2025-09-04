from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from config import STOCK_TICKERS, UPDATE_INTERVAL_SECONDS, PORTFOLIO_WEIGHTS, INCEPTION_DATE, MAX_RETRIES
import database as db

app = FastAPI()

# --- Globals for caching ---
# We still need some in-memory caching for live data, but historical comes from DB.
live_stock_data = {}
last_updated = None
data_ready = False
data_lock = threading.Lock()


@app.on_event("startup")
def startup_event():
    """On startup, initialize DB and start live data updates."""
    db.initialize_db()
    # The backfill is now a separate script (backfill.py)
    
    # Start the live data update thread
    update_thread = threading.Thread(target=update_live_data_periodically, daemon=True)
    update_thread.start()


def get_live_stock_data():
    """
    Fetches the latest available data for each stock.
    This is used for the main table view, not the historical chart.
    """
    results = {}
    today = datetime.today().strftime('%Y-%m-%d')

    for ticker in STOCK_TICKERS:
        try:
            print(f"Fetching live data for {ticker}...")

            # Get latest price from our DB
            price_hist = db.get_price_history(ticker, (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d'), today)

            if price_hist.empty:
                print(f"✗ {ticker}: No recent data in DB")
                continue

            # Basic info from yfinance (less critical than price)
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('longName', ticker)
            currency = stock_info.get('currency', 'USD')

            # Calculations based on DB data
            current_price = price_hist['adj_close'].iloc[-1]

            # Inception price
            inception_price_df = db.get_price_history(ticker, INCEPTION_DATE, INCEPTION_DATE)
            if inception_price_df.empty:
                # Fallback to earliest available date in DB
                all_time_hist = db.get_price_history(ticker, '2000-01-01', today)
                baseline_price = all_time_hist['adj_close'].iloc[0]
                baseline_date = all_time_hist.index[0].strftime('%Y-%m-%d')
            else:
                baseline_price = inception_price_df['adj_close'].iloc[0]
                baseline_date = INCEPTION_DATE

            # Returns
            total_return = ((current_price - baseline_price) / baseline_price) * 100

            day_ago = price_hist['adj_close'].iloc[-2] if len(price_hist) > 1 else current_price
            week_ago = price_hist['adj_close'].iloc[max(0, len(price_hist) - 5)]
            month_ago = price_hist['adj_close'].iloc[max(0, len(price_hist) - 22)]

            day_return = ((current_price - day_ago) / day_ago * 100) if day_ago != 0 else 0
            week_return = ((current_price - week_ago) / week_ago * 100) if week_ago != 0 else 0
            month_return = ((current_price - month_ago) / month_ago * 100) if month_ago != 0 else 0

            weight = PORTFOLIO_WEIGHTS.get(ticker, 0)

            results[ticker] = {
                'ticker': ticker,
                'company_name': company_name,
                'price': round(float(current_price), 2),
                'currency': currency,
                'weight': weight,
                'day_1_return': round(day_return, 2),
                'week_1_return': round(week_return, 2),
                'month_1_return': round(month_return, 2),
                'total_return': round(total_return, 2),
                'baseline_date': baseline_date
            }
            print(f"✓ Live data for {ticker} processed.")

        except Exception as e:
            print(f"✗ Error fetching live data for {ticker}: {e}")
            continue

    return results

def update_live_data_periodically():
    """Periodically fetches and caches live data."""
    global live_stock_data, last_updated, data_ready
    
    while True:
        print("Updating live data...")
        new_data = get_live_stock_data()
        
        with data_lock:
            live_stock_data = new_data
            last_updated = datetime.now()
            if not data_ready and len(live_stock_data) > 0:
                data_ready = True
            
        print(f"Live data cache updated with {len(live_stock_data)} stocks.")
        time.sleep(UPDATE_INTERVAL_SECONDS)


@app.get("/", response_class=HTMLResponse)
async def index():
    # The HTML file was referencing a non-existent templates folder. Correcting path.
    with open("index.html", "r") as f:
        return f.read()

@app.get("/api/stocks")
async def get_stocks():
    with data_lock:
        # This endpoint now serves the cached live data
        current_stock_data = live_stock_data.copy()
        current_last_updated = last_updated
    
    return {
        'stocks': list(current_stock_data.values()),
        'last_updated': current_last_updated.isoformat() if current_last_updated else None,
        'total_stocks': len(current_stock_data)
    }

@app.get("/index")
@app.get("/api/index")
async def get_index():
    """Calculate weighted index performance from cached live data."""
    with data_lock:
        current_stock_data = live_stock_data.copy()
        is_ready = data_ready
    
    if not current_stock_data or not is_ready:
        return {'error': 'Data not ready yet', 'ready': False}
    
    # Calculate weighted returns
    index_total_return = 0  # Cumulative since inception
    index_day_return = 0
    index_week_return = 0
    index_month_return = 0
    total_weight = 0
    
    for ticker, data in current_stock_data.items():
        weight = data['weight'] / 100  # Convert to decimal
        index_total_return += data['total_return'] * weight  # CUMULATIVE
        index_day_return += data['day_1_return'] * weight
        index_week_return += data['week_1_return'] * weight
        index_month_return += data['month_1_return'] * weight
        total_weight += data['weight']
    
    # Index value = cumulative performance since Aug 1
    base_value = 100
    current_value = base_value * (1 + index_total_return / 100)
    
    return {
        'index_name': 'Abhay GLP-1 Index',
        'current_value': round(current_value, 2),
        'base_value': base_value,
        'day_return': round(index_day_return, 2),
        'week_return': round(index_week_return, 2),
        'month_return': round(index_month_return, 2),
        'total_return': round(index_total_return, 2),  # CUMULATIVE since inception
        'total_weight': total_weight,
        'constituents': len(current_stock_data),
        'ready': True,
        'inception_date': INCEPTION_DATE,
        'components': [
            {
                'ticker': data['ticker'],
                'company_name': data['company_name'],
                'weight': data['weight'],
                'price': data['price'],
                'currency': data['currency'],
                'day_1_return': data['day_1_return'],
                'week_1_return': data['week_1_return'],
                'month_1_return': data['month_1_return'],
                'total_return': data['total_return'],  # RSI - Return Since Inception
                'contribution': round(data['total_return'] * data['weight'] / 100, 2)  # Use RSI for contribution
            }
            for ticker, data in current_stock_data.items()
        ]
    }

@app.get("/api/ready")
async def check_ready():
    """Check if data is ready"""
    with data_lock:
        is_ready = data_ready
        # Use live_stock_data for the count
        stock_count = len(live_stock_data)
    
    return {'ready': is_ready, 'stock_count': stock_count}

@app.get("/api/chart")
async def get_chart_data(period: str = '1M', benchmark: str = None):
    """
    Get historical chart data for the index from the database.
    This is much faster as it queries pre-populated data.
    """
    try:
        end_date = pd.to_datetime(datetime.today())
        inception_date_dt = pd.to_datetime(INCEPTION_DATE)

        # Determine start date
        if period == '1D':
            start_date = end_date - pd.Timedelta(days=4) # Fetch 4 days to ensure we get at least 2 data points
        elif period == '1W':
            start_date = end_date - pd.Timedelta(days=7)
        elif period == '1M':
            start_date = end_date - pd.Timedelta(days=30)
        elif period == 'ITD':
            start_date = inception_date_dt
        else:
            start_date = end_date - pd.Timedelta(days=30)

        start_date = max(start_date, inception_date_dt)
        
        print(f"DB: Calculating chart for {period} from {start_date.date()} to {end_date.date()}")

        # --- Fetch all required data from DB first ---
        all_prices = {}
        for ticker in STOCK_TICKERS:
            # Fetch data from inception to ensure we can calculate returns correctly
            prices_df = db.get_price_history(ticker, inception_date_dt.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not prices_df.empty:
                all_prices[ticker] = prices_df.rename(columns={'adj_close': ticker})[ticker]
        
        if not all_prices:
            return {'error': 'No historical data in DB.', 'ready': False}

        # Combine all dataframes, forward-fill missing values
        combined_df = pd.concat(all_prices.values(), axis=1).sort_index()
        combined_df = combined_df.ffill().bfill() # Fill weekends/holidays
        
        # --- Calculate Index Value ---
        # Get baseline prices at inception date for each stock
        baseline_prices = combined_df.loc[combined_df.index >= inception_date_dt].iloc[0]
        
        # Calculate daily return for each stock relative to its baseline
        daily_returns = combined_df.div(baseline_prices, axis=1) - 1
        
        # Apply weights
        weights = pd.Series({ticker: PORTFOLIO_WEIGHTS.get(ticker, 0) / 100 for ticker in STOCK_TICKERS})
        weighted_returns = daily_returns[STOCK_TICKERS].mul(weights, axis=1)
        
        # Sum weighted returns to get daily total index return
        index_daily_total_return = weighted_returns.sum(axis=1)
        
        # Calculate final index value
        index_values = 100 * (1 + index_daily_total_return)

        # Filter for the requested period
        index_values = index_values[index_values.index >= start_date]

        chart_data = [{
            'time': date.strftime('%Y-%m-%d'),
            'value': round(value, 2)
        } for date, value in index_values.items()]

        # --- Handle Benchmark ---
        benchmark_data = []
        benchmark_period_return = None
        if benchmark:
            bench_prices_df = db.get_price_history(benchmark, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not bench_prices_df.empty:
                # Calculate period return for the benchmark
                bench_start_price = bench_prices_df['adj_close'].iloc[0]
                bench_end_price = bench_prices_df['adj_close'].iloc[-1]
                if bench_start_price != 0:
                    benchmark_period_return = round(((bench_end_price - bench_start_price) / bench_start_price) * 100, 2)

                # Normalize benchmark to start at the same value as our index on the start date
                index_start_value = index_values.iloc[0]
                normalized_bench = (bench_prices_df['adj_close'] / bench_start_price) * index_start_value

                benchmark_data = [{
                    'time': date.strftime('%Y-%m-%d'),
                    'value': round(value, 2)
                } for date, value in normalized_bench.items()]
        
        return {
            'period': period,
            'data': chart_data,
            'benchmark_data': benchmark_data,
            'benchmark_period_return': benchmark_period_return,
            'inception_date': INCEPTION_DATE,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }

    except Exception as e:
        print(f"Error generating chart data from DB: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Error generating chart data: {str(e)}', 'ready': False}


if __name__ == "__main__":
    import uvicorn
    # The startup event handles threads now
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import json
import os
from config import STOCK_TICKERS, UPDATE_INTERVAL_SECONDS, PORTFOLIO_WEIGHTS, INCEPTION_DATE, MAX_RETRIES, BENCHMARK_TICKERS
import database as db

app = FastAPI()

# --- Globals for caching ---
live_stock_data = {}
last_updated = None
data_ready = False
data_lock = threading.Lock()

@app.on_event("startup")
def startup_event():
    """On startup, initialize DB and start live data updates."""
    db.initialize_db()
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
            # Get latest price from our DB
            price_hist = db.get_price_history(ticker, (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d'), today)

            if price_hist.empty:
                print(f"✗ {ticker}: No recent data in DB, attempting live fetch...")
                stock_info_live = yf.Ticker(ticker)
                live_hist = stock_info_live.history(period="3mo", auto_adjust=False)
                if live_hist.empty:
                    continue
                db.insert_prices(ticker, live_hist[['Adj Close']])
                price_hist = db.get_price_history(ticker, (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d'), today)

            # Basic info from yfinance (less critical than price)
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('longName', ticker)
            currency = stock_info.get('currency', 'USD')

            # Calculations based on DB data
            current_price = price_hist['adj_close'].iloc[-1]

            # Inception price
            inception_price_df = db.get_price_history(ticker, INCEPTION_DATE, INCEPTION_DATE)
            if inception_price_df.empty:
                all_time_hist = db.get_price_history(ticker, '2000-01-01', today)
                baseline_price = all_time_hist['adj_close'].iloc[0]
                baseline_date = all_time_hist.index[0].strftime('%Y-%m-%d')
            else:
                baseline_price = inception_price_df['adj_close'].iloc[0]
                baseline_date = INCEPTION_DATE

            # Returns
            total_return = ((current_price - baseline_price) / baseline_price) * 100 if baseline_price != 0 else 0

            day_ago = price_hist['adj_close'].iloc[-2] if len(price_hist) > 1 else current_price
            week_ago = price_hist['adj_close'].iloc[max(0, len(price_hist) - 6)] if len(price_hist) > 5 else current_price
            month_ago = price_hist['adj_close'].iloc[max(0, len(price_hist) - 22)] if len(price_hist) > 21 else current_price

            day_return = ((current_price - day_ago) / day_ago * 100) if day_ago != 0 else 0
            week_return = ((current_price - week_ago) / week_ago * 100) if week_ago != 0 else 0
            month_return = ((current_price - month_ago) / month_ago * 100) if month_ago != 0 else 0

            weight = PORTFOLIO_WEIGHTS.get(ticker, 0)

            results[ticker] = {
                'ticker': ticker, 'company_name': company_name, 'price': round(float(current_price), 2),
                'currency': currency, 'weight': weight, 'day_1_return': round(day_return, 2),
                'week_1_return': round(week_return, 2), 'month_1_return': round(month_return, 2),
                'total_return': round(total_return, 2), 'baseline_date': baseline_date
            }
        except Exception as e:
            print(f"✗ Error processing live data for {ticker}: {e}")
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
    with open("index.html", "r") as f:
        return f.read()


@app.get("/api/stocks")
async def get_stocks():
    with data_lock:
        return {
            'stocks': list(live_stock_data.values()),
            'last_updated': last_updated.isoformat() if last_updated else None,
            'total_stocks': len(live_stock_data)
        }


@app.get("/index")
@app.get("/api/index")
async def get_index():
    """Calculate weighted index performance from cached live data."""
    with data_lock:
        current_stock_data = live_stock_data.copy()
    if not current_stock_data or not data_ready:
        return {'error': 'Data not ready yet', 'ready': False}

    index_total_return, index_day_return, index_week_return, index_month_return, total_weight = 0, 0, 0, 0, 0
    for data in current_stock_data.values():
        weight = data['weight'] / 100
        index_total_return += data['total_return'] * weight
        index_day_return += data['day_1_return'] * weight
        index_week_return += data['week_1_return'] * weight
        index_month_return += data['month_1_return'] * weight
        total_weight += data['weight']

    base_value = 100
    current_value = base_value * (1 + index_total_return / 100)

    return {
        'index_name': 'Abhay GLP-1 Index', 'current_value': round(current_value, 2),
        'base_value': base_value, 'day_return': round(index_day_return, 2),
        'week_return': round(index_week_return, 2), 'month_return': round(index_month_return, 2),
        'total_return': round(index_total_return, 2), 'total_weight': total_weight,
        'constituents': len(current_stock_data), 'ready': True, 'inception_date': INCEPTION_DATE,
        'components': [
            {'contribution': round(data['total_return'] * data['weight'] / 100, 2), **data}
            for data in current_stock_data.values()
        ]
    }


@app.get("/api/ready")
async def check_ready():
    with data_lock:
        return {'ready': data_ready, 'stock_count': len(live_stock_data)}


def _period_to_start(period: str, end_date: pd.Timestamp, inception_date_dt: pd.Timestamp) -> pd.Timestamp:
    period = (period or '1M').upper()
    if period == '1D':
        start_date = end_date - pd.Timedelta(days=4)
    elif period == '1W':
        start_date = end_date - pd.Timedelta(days=7)
    elif period == '1M':
        start_date = end_date - pd.Timedelta(days=30)
    elif period == '3M':
        start_date = end_date - pd.Timedelta(days=90)
    elif period == '6M':
        start_date = end_date - pd.Timedelta(days=180)
    elif period == '1Y':
        start_date = end_date - pd.Timedelta(days=365)
    elif period == 'YTD':
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif period == 'ITD':
        start_date = inception_date_dt
    else:
        start_date = end_date - pd.Timedelta(days=30)
    return max(start_date, inception_date_dt)


@app.get("/api/chart")
async def get_chart_data(period: str = '1M', benchmark: str | None = None, benchmarks: str | None = None):
    try:
        end_date = pd.to_datetime(datetime.today())
        inception_date_dt = pd.to_datetime(INCEPTION_DATE)
        start_date = _period_to_start(period, end_date, inception_date_dt)

        # Parse benchmarks (support both singular 'benchmark' and comma-separated 'benchmarks')
        bench_list = []
        if benchmarks:
            bench_list = [b.strip().upper() for b in benchmarks.split(',') if b.strip()]
        elif benchmark:
            bench_list = [benchmark.strip().upper()]

        # Only include known/allowed benchmarks
        bench_list = [b for b in bench_list if b in BENCHMARK_TICKERS]

        # Load all prices from DB (index constituents + requested benchmarks)
        all_prices: dict[str, pd.Series] = {}
        tickers_to_fetch = list(dict.fromkeys(STOCK_TICKERS + bench_list))
        for ticker in tickers_to_fetch:
            prices_df = db.get_price_history(ticker, inception_date_dt.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not prices_df.empty:
                all_prices[ticker] = prices_df.rename(columns={'adj_close': ticker})[ticker]

        if not any(t in all_prices for t in STOCK_TICKERS):
            return {'error': 'No historical data in DB.', 'ready': False}

        combined_df = pd.concat(all_prices.values(), axis=1).sort_index().ffill().bfill()

        # Compute index values from inception baseline
        baseline_prices = combined_df.loc[combined_df.index >= inception_date_dt].iloc[0]
        daily_returns = combined_df.div(baseline_prices, axis=1) - 1
        weights = pd.Series({ticker: PORTFOLIO_WEIGHTS.get(ticker, 0) / 100 for ticker in STOCK_TICKERS})
        weighted_returns = daily_returns[STOCK_TICKERS].mul(weights, axis=1)
        index_daily_total_return = weighted_returns.sum(axis=1)
        index_values = 100 * (1 + index_daily_total_return)

        # Slice period
        idx_period = index_values.loc[index_values.index >= start_date]
        chart_data = [{'time': d.strftime('%Y-%m-%d'), 'value': round(v, 2)} for d, v in idx_period.items()]

        # Benchmarks map
        benchmarks_map = {}
        for b in bench_list:
            series = combined_df[[b]].loc[start_date:end_date]
            if series.empty:
                continue
            bench_start_price = series.iloc[0, 0]
            bench_end_price = series.iloc[-1, 0]
            period_return = None
            if bench_start_price != 0:
                period_return = round(((bench_end_price - bench_start_price) / bench_start_price) * 100, 2)

            # Normalize benchmark to index start value
            index_start_value = float(index_values.loc[index_values.index >= start_date].iloc[0])
            normalized = (series[b] / bench_start_price) * index_start_value
            benchmarks_map[b] = {
                'data': [{'time': d.strftime('%Y-%m-%d'), 'value': round(v, 2)} for d, v in normalized.items()],
                'period_return': period_return
            }

        # Backward compatibility when exactly one benchmark is present
        benchmark_data = []
        benchmark_period_return = None
        if len(bench_list) == 1 and bench_list[0] in benchmarks_map:
            b = bench_list[0]
            benchmark_data = benchmarks_map[b]['data']
            benchmark_period_return = benchmarks_map[b]['period_return']

        return {
            'period': period,
            'data': chart_data,
            'benchmarks': benchmarks_map,
            'benchmark_data': benchmark_data,
            'benchmark_period_return': benchmark_period_return,
            'inception_date': INCEPTION_DATE,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': f'Error generating chart data: {str(e)}', 'ready': False}


@app.get("/api/chart.csv")
async def get_chart_csv(period: str = '1M', benchmarks: str | None = None, benchmark: str | None = None):
    try:
        end_date = pd.to_datetime(datetime.today())
        inception_date_dt = pd.to_datetime(INCEPTION_DATE)
        start_date = _period_to_start(period, end_date, inception_date_dt)

        bench_list = []
        if benchmarks:
            bench_list = [b.strip().upper() for b in benchmarks.split(',') if b.strip()]
        elif benchmark:
            bench_list = [benchmark.strip().upper()]
        bench_list = [b for b in bench_list if b in BENCHMARK_TICKERS]

        # Load DB data
        all_prices: dict[str, pd.Series] = {}
        tickers_to_fetch = list(dict.fromkeys(STOCK_TICKERS + bench_list))
        for ticker in tickers_to_fetch:
            prices_df = db.get_price_history(ticker, inception_date_dt.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not prices_df.empty:
                all_prices[ticker] = prices_df.rename(columns={'adj_close': ticker})[ticker]

        if not any(t in all_prices for t in STOCK_TICKERS):
            return Response("error,no data", media_type='text/csv; charset=utf-8')

        combined_df = pd.concat(all_prices.values(), axis=1).sort_index().ffill().bfill()

        # Index values
        baseline_prices = combined_df.loc[combined_df.index >= inception_date_dt].iloc[0]
        daily_returns = combined_df.div(baseline_prices, axis=1) - 1
        weights = pd.Series({ticker: PORTFOLIO_WEIGHTS.get(ticker, 0) / 100 for ticker in STOCK_TICKERS})
        weighted_returns = daily_returns[STOCK_TICKERS].mul(weights, axis=1)
        index_daily_total_return = weighted_returns.sum(axis=1)
        index_values = 100 * (1 + index_daily_total_return)
        idx_period = index_values.loc[index_values.index >= start_date]
        df = pd.DataFrame({'index': idx_period})

        # Add requested benchmarks, normalized to index start value
        if bench_list:
            index_start_value = float(index_values.loc[index_values.index >= start_date].iloc[0])
            for b in bench_list:
                if b not in combined_df.columns:
                    continue
                series = combined_df[[b]].loc[start_date:end_date]
                if series.empty:
                    continue
                bench_start_price = series.iloc[0, 0]
                if bench_start_price == 0:
                    continue
                normalized = (series[b] / bench_start_price) * index_start_value
                df = df.join(normalized.rename(b), how='outer')

        df = df.loc[df.index >= start_date].sort_index().ffill().bfill()
        df = df.reset_index().rename(columns={'index': 'date'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Only include requested benchmark columns in CSV, maintain specified order
        ordered_cols = ['date', 'index'] + [b for b in ['SPY', 'QQQ', 'ACWI', 'EEM', 'XLV', 'IBB'] if b in df.columns]
        df = df[ordered_cols]

        csv_text = df.to_csv(index=False)
        return Response(content=csv_text, media_type='text/csv; charset=utf-8')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(f"error,{str(e)}", media_type='text/csv; charset=utf-8')


@app.get("/api/events")
async def get_events():
    try:
        events_path = os.path.join('events', 'events.json')
        if not os.path.exists(events_path):
            return []
        with open(events_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        return []

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
from config import STOCK_TICKERS, UPDATE_INTERVAL_SECONDS, PORTFOLIO_WEIGHTS, INCEPTION_DATE, MAX_RETRIES

app = FastAPI()

stock_data = {}
last_updated = None
data_ready = False
data_lock = threading.Lock()

def get_stock_data():
    results = {}
    
    # Get all available data for each stock (max period)
    for ticker in STOCK_TICKERS:
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Fetching {ticker}... (attempt {attempt+1}/{MAX_RETRIES})")
                stock = yf.Ticker(ticker)
                
                # Get currency and company name with robust fallbacks
                currency = 'USD'
                company_name = ticker  # Default fallback
                
                try:
                    info = stock.info
                    if info:
                        currency = info.get('currency', 'USD') or 'USD'
                        # Try multiple fields for company name
                        company_name = (
                            info.get('longName') or 
                            info.get('shortName') or 
                            info.get('displayName') or 
                            info.get('quoteType', '') + ' ' + ticker if info.get('quoteType') else ticker
                        ).strip() or ticker
                except Exception as e:
                    print(f"⚠ {ticker}: Could not fetch company info: {e}")
                    # Keep defaults: currency='USD', company_name=ticker
                
                # Get maximum available data
                all_data = stock.history(period='max')
                
                if all_data.empty:
                    print(f"✗ {ticker}: No data available")
                    break
                
                # Get recent data for short-term calculations
                recent_data = stock.history(period='2mo')
                
                if recent_data.empty:
                    print(f"✗ {ticker}: No recent data")
                    break
                
                # Determine price column: Prefer Adj Close for Total Return calculation
                PRICE_COL = 'Adj Close' if 'Adj Close' in all_data.columns else 'Close'
                
                # Ensure the price column exists in recent data too
                if PRICE_COL not in recent_data.columns:
                    PRICE_COL = 'Close'  # Fallback
                
                current_price = recent_data[PRICE_COL].iloc[-1]
                
                # Find baseline price (earliest available or Aug 1, whichever is later)
                inception_date = pd.to_datetime(INCEPTION_DATE).tz_localize(all_data.index.tz)
                
                # Filter data from inception date or earliest available
                available_from_inception = all_data[all_data.index >= inception_date]
                
                if available_from_inception.empty:
                    # Use earliest available data as baseline
                    baseline_price = all_data[PRICE_COL].iloc[0]
                    baseline_date = all_data.index[0].strftime('%Y-%m-%d')
                    print(f"⚠ {ticker}: Using earliest data from {baseline_date}")
                else:
                    # Use inception date baseline
                    baseline_price = available_from_inception[PRICE_COL].iloc[0]
                    baseline_date = INCEPTION_DATE
                
                if pd.isna(baseline_price) or pd.isna(current_price) or baseline_price == 0:
                    break
                
                # Calculate cumulative return since baseline using Adj Close
                total_return = ((current_price - baseline_price) / baseline_price * 100)
                
                # Short-term returns using same price column
                recent_closes = recent_data[PRICE_COL].dropna()
                
                day_ago = recent_closes.iloc[-2] if len(recent_closes) > 1 else current_price
                week_ago = recent_closes.iloc[max(0, len(recent_closes) - 5)]
                month_ago = recent_closes.iloc[max(0, len(recent_closes) - 22)]
                
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
                
                price_type = "Adj Close" if PRICE_COL == 'Adj Close' else "Close"
                print(f"✓ {ticker}: {current_price:.2f} {currency} | {company_name} | Total: {total_return:.1f}% since {baseline_date} | Weight: {weight}% ({price_type})")
                success = True
                break
                
            except Exception as e:
                print(f"Attempt {attempt+1}/{MAX_RETRIES} failed for {ticker}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)  # Wait before retrying
                continue
        
        if not success:
            print(f"✗ {ticker}: Failed after {MAX_RETRIES} attempts")
            continue
    
    return results

def update_stock_data():
    global stock_data, last_updated, data_ready
    
    while True:
        print("Updating...")
        new_data = get_stock_data()
        
        # Thread-safe update using lock
        with data_lock:
            stock_data = new_data
            last_updated = datetime.now()
            data_ready = True
            
        print(f"Got {len(stock_data)} stocks")
        time.sleep(UPDATE_INTERVAL_SECONDS)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.get("/api/stocks")
async def get_stocks():
    with data_lock:
        current_stock_data = stock_data.copy()
        current_last_updated = last_updated
    
    return {
        'stocks': list(current_stock_data.values()),
        'last_updated': current_last_updated.isoformat() if current_last_updated else None,
        'total_stocks': len(current_stock_data)
    }

@app.get("/index")
@app.get("/api/index")
async def get_index():
    """Calculate weighted index performance"""
    with data_lock:
        current_stock_data = stock_data.copy()
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
        stock_count = len(stock_data)
    
    return {'ready': is_ready, 'stock_count': stock_count}

@app.get("/api/chart")
async def get_chart_data(period: str = '1M'):
    """Get historical chart data for the index - calculates real daily index values"""
    with data_lock:
        current_stock_data = stock_data.copy()
        is_ready = data_ready
    
    if not current_stock_data or not is_ready:
        return {'error': 'Data not ready yet', 'ready': False}
    
    try:
        # Define date ranges for each period
        end_date = pd.to_datetime('today')
        inception_date = pd.to_datetime(INCEPTION_DATE)
        
        # Determine start date based on period
        if period == '1D':
            start_date = end_date - pd.Timedelta(days=1)
        elif period == '1W':
            start_date = end_date - pd.Timedelta(days=7)
        elif period == '1M':
            start_date = end_date - pd.Timedelta(days=30)
        elif period == 'ITD':  # Inception-to-Date
            start_date = inception_date
        else:
            start_date = end_date - pd.Timedelta(days=30)  # Default to 1M
        
        # Never go before inception date
        if start_date < inception_date:
            start_date = inception_date
        
        print(f"Calculating chart data for {period}: {start_date.date()} to {end_date.date()}")
        
        # Get historical data for all tickers at once (more efficient)
        all_historical_data = {}
        for ticker in STOCK_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                # Get data from inception to now
                hist_data = stock.history(start=inception_date, end=end_date + pd.Timedelta(days=1))
                
                if not hist_data.empty:
                    # Use same price column logic
                    PRICE_COL = 'Adj Close' if 'Adj Close' in hist_data.columns else 'Close'
                    all_historical_data[ticker] = {
                        'data': hist_data,
                        'price_col': PRICE_COL,
                        'weight': PORTFOLIO_WEIGHTS.get(ticker, 0) / 100
                    }
                    print(f"✓ Got historical data for {ticker}: {len(hist_data)} days")
                else:
                    print(f"✗ No historical data for {ticker}")
                    
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
                continue
        
        if not all_historical_data:
            return {'error': 'No historical data available', 'ready': False}
        
        # Calculate daily index values
        chart_data = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_weighted_return = 0
            valid_stocks = 0
            
            for ticker, hist_info in all_historical_data.items():
                try:
                    hist_data = hist_info['data']
                    price_col = hist_info['price_col']
                    weight = hist_info['weight']
                    
                    # Get baseline price (inception date)
                    inception_data = hist_data[hist_data.index.date == inception_date.date()]
                    if inception_data.empty:
                        # Use first available data if inception date not available
                        baseline_price = hist_data[price_col].iloc[0]
                    else:
                        baseline_price = inception_data[price_col].iloc[0]
                    
                    # Get price for current date
                    current_data = hist_data[hist_data.index.date == current_date.date()]
                    if current_data.empty:
                        # Use last available price before this date
                        available_data = hist_data[hist_data.index.date <= current_date.date()]
                        if available_data.empty:
                            continue
                        current_price = available_data[price_col].iloc[-1]
                    else:
                        current_price = current_data[price_col].iloc[0]
                    
                    if pd.isna(baseline_price) or pd.isna(current_price) or baseline_price == 0:
                        continue
                    
                    # Calculate stock return since inception
                    stock_return = ((current_price - baseline_price) / baseline_price) * 100
                    
                    # Add weighted return to index
                    daily_weighted_return += stock_return * weight
                    valid_stocks += 1
                    
                except Exception as e:
                    print(f"Error calculating {ticker} for {current_date.date()}: {e}")
                    continue
            
            if valid_stocks > 0:
                # Index value starts at 100 on inception date
                index_value = 100 * (1 + daily_weighted_return / 100)
                
                chart_data.append({
                    'time': current_date.strftime('%Y-%m-%d'),
                    'value': round(index_value, 2)
                })
            
            current_date += pd.Timedelta(days=1)
        
        print(f"Generated {len(chart_data)} data points for {period}")
        
        return {
            'period': period,
            'data': chart_data,
            'inception_date': INCEPTION_DATE,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"Error generating chart data: {e}")
        return {'error': f'Error generating chart data: {str(e)}', 'ready': False}

if __name__ == "__main__":
    import uvicorn
    
    update_thread = threading.Thread(target=update_stock_data, daemon=True)
    update_thread.start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

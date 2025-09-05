STOCK_TICKERS = [
    'LLY',           # Eli Lilly (US)
    'WW',            # WW International (US) 
    'NVO',           # Novo Nordisk (US ADR)
    '2565.HK',       # PegBio (Hong Kong)
    'AIRS',          # AirSculpt Technologies (US)
    '1801.HK',       # Innovent Biologics (Hong Kong)
    '688166.SS',     # BrightGene Bio-Medical (Shanghai)
    'ROG.SW',        # Roche (Swiss)
    'SHAILY.NS',     # Shaily Engineering (India)
    'XPOF',          # Xponential Fitness (US)
    'NGVC',          # Natural Grocers (US)
    '4587.T',        # PeptiDream (Tokyo)
    'GALD.SW',       # Galderma (Swiss)
    'APYX',          # Apyx Medical (US)
    'ATR',           # AptarGroup (US)
    'HALO',          # Halozyme Therapeutics (US)
    'ONESOURCE.BO'   # OneSource Specialty (Bombay)
]

# Optional benchmarks for comparisons
BENCHMARK_TICKERS = ['SPY', 'QQQ', 'ACWI', 'EEM', 'XLV', 'IBB']

# Portfolio weights (must sum to 100)
PORTFOLIO_WEIGHTS = {
    'LLY': 13,
    'WW': 12,
    'NVO': 7,
    'XPOF': 4,
    'NGVC': 4,
    '2565.HK': 7,  # PegBio
    '688166.SS': 6,  # BrightGene
    'AIRS': 7,
    '4587.T': 4,  # PeptiDream
    'GALD.SW': 4,
    '1801.HK': 7,  # Innovent
    'APYX': 4,
    'ROG.SW': 5,
    'ATR': 4,  # AptarGroup
    'SHAILY.NS': 5,
    'ONESOURCE.BO': 3,
    'HALO': 4  # Halozyme
}

UPDATE_INTERVAL_SECONDS = 300  # 5 minutes

MAX_RETRIES = 3

TIMEOUT_SECONDS = 10

# Index inception date (when index = 100)
INCEPTION_DATE = '2025-08-01'

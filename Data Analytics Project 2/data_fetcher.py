"""
AlphaPulse - Data Fetcher Module
Week 1: Data Acquisition & Cleaning
Fetches historical OHLCV data for a diverse 10-stock portfolio + S&P 500
using yfinance with robust error handling and local caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Portfolio Definition
# ------------------------------------------------------------
PORTFOLIO_TICKERS = [
    "AAPL",   # Tech - Apple
    "MSFT",   # Tech - Microsoft
    "GOOGL",  # Tech - Alphabet
    "AMZN",   # Consumer Discretionary - Amazon
    "JPM",    # Financials - JPMorgan Chase
    "GS",     # Financials - Goldman Sachs
    "XOM",    # Energy - ExxonMobil
    "JNJ",    # Healthcare - Johnson & Johnson
    "TSLA",   # Industrials/EV - Tesla
    "NVDA",   # Semiconductors - NVIDIA
]

INDEX_TICKER = "^GSPC"   # S&P 500 benchmark

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "portfolio_data.csv")
PERIOD = "1y"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_portfolio_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for the portfolio + index.
    Uses local CSV cache to handle API rate limits.

    Returns:
        DataFrame indexed by Date, columns = ticker symbols.
    """
    _ensure_cache_dir()

    if not force_refresh and os.path.exists(CACHE_FILE):
        modified_time = os.path.getmtime(CACHE_FILE)
        age_hours = (time.time() - modified_time) / 3600
        if age_hours < 24:
            logger.info("Loading data from cache (age: %.1fh)", age_hours)
            df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            return df

    logger.info("Fetching fresh data from yfinance ...")
    all_tickers = PORTFOLIO_TICKERS + [INDEX_TICKER]
    frames = {}

    for ticker in all_tickers:
        try:
            logger.info("  Downloading %s ...", ticker)
            raw = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)
            if raw.empty:
                logger.warning("  No data for %s, skipping.", ticker)
                continue
            frames[ticker] = raw["Close"].rename(ticker)
            time.sleep(0.3)   # be polite with the API
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", ticker, exc)

    if not frames:
        raise RuntimeError("Could not fetch any ticker data. Check your internet connection.")

    df = pd.concat(frames.values(), axis=1).sort_index()
    # Forward-fill gaps from stock splits / holidays; drop rows still all-NaN
    df = df.ffill().dropna(how="all")

    df.to_csv(CACHE_FILE)
    logger.info("Data saved to cache: %s", CACHE_FILE)
    return df


def fetch_volume_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch daily trading volume data for the portfolio.

    Returns:
        DataFrame indexed by Date, columns = ticker symbols.
    """
    vol_cache = os.path.join(CACHE_DIR, "volume_data.csv")
    _ensure_cache_dir()

    if not force_refresh and os.path.exists(vol_cache):
        modified_time = os.path.getmtime(vol_cache)
        age_hours = (time.time() - modified_time) / 3600
        if age_hours < 24:
            df = pd.read_csv(vol_cache, index_col=0, parse_dates=True)
            return df

    logger.info("Fetching volume data from yfinance ...")
    frames = {}
    for ticker in PORTFOLIO_TICKERS:
        try:
            raw = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)
            if not raw.empty and "Volume" in raw.columns:
                frames[ticker] = raw["Volume"].rename(ticker)
            time.sleep(0.3)
        except Exception as exc:
            logger.warning("  Volume fetch failed for %s: %s", ticker, exc)

    df = pd.concat(frames.values(), axis=1).sort_index().ffill().dropna(how="all")
    df.to_csv(vol_cache)
    return df


if __name__ == "__main__":
    prices = fetch_portfolio_data()
    print(f"\nPrice data shape: {prices.shape}")
    print(prices.tail())
    volumes = fetch_volume_data()
    print(f"\nVolume data shape: {volumes.shape}")
    print(volumes.tail())

"""Data download and cleaning utilities."""

import pandas as pd
import yfinance as yf

from config import (
    BENCHMARK_TICKER,
    DATA_DIR,
    DATA_DOWNLOAD_START,
    END_DATE,
    FORWARD_FILL_LIMIT,
    FULL_START,
    MAX_MISSING_ASSET_FRACTION,
    STRICT_MISSING_DATA,
    TICKERS,
)


def _normalise_yfinance_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance for multiple tickers.")
    first_level = set(raw.columns.get_level_values(0))
    if {"Open", "High", "Low", "Close", "Adj Close", "Volume"}.intersection(first_level):
        raw = raw.swaplevel(axis=1)
    return raw.sort_index(axis=1)


def download_ohlcv(tickers=None, include_benchmark: bool = True) -> pd.DataFrame:
    """Download daily OHLCV data and return a cleaned long panel."""
    tickers = list(tickers or TICKERS)
    all_tickers = tickers + ([BENCHMARK_TICKER] if include_benchmark else [])
    print(f"Downloading {len(all_tickers)} tickers from yfinance...")
    raw = yf.download(
        all_tickers,
        start=DATA_DOWNLOAD_START,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    raw = _normalise_yfinance_columns(raw)

    adj_close = raw.xs("Adj Close", axis=1, level=1)
    if STRICT_MISSING_DATA:
        missing = raw.isna()
        if missing.any().any():
            missing_by_ticker = {
                ticker: int(missing[ticker].sum().sum())
                for ticker in raw.columns.get_level_values(0).unique()
            }
            missing_by_ticker = {
                ticker: count
                for ticker, count in sorted(
                    missing_by_ticker.items(), key=lambda item: item[1], reverse=True
                )
                if count > 0
            }
            missing_dates = raw.index[missing.any(axis=1)]
            preview = ", ".join(str(d.date()) for d in missing_dates[:10])
            details = "; ".join(f"{ticker}: {count} cells" for ticker, count in missing_by_ticker.items())
            raise ValueError(
                "Missing OHLCV values found in yfinance download. Strict mode is enabled, "
                "so the run has been stopped instead of filling or dropping data. "
                f"First missing dates: {preview}. Missing cells by ticker: {details}"
            )

    missing_fraction = adj_close.isna().mean(axis=1)
    keep_dates = missing_fraction <= MAX_MISSING_ASSET_FRACTION
    raw = raw.loc[keep_dates]

    cleaned = raw if FORWARD_FILL_LIMIT == 0 else raw.ffill(limit=FORWARD_FILL_LIMIT)
    adj_close = cleaned.xs("Adj Close", axis=1, level=1)
    complete_dates = adj_close.dropna().index
    cleaned = cleaned.loc[complete_dates]

    long = cleaned.stack(level=0).reset_index()
    long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "tic"})
    long = long.rename(
        columns={
            "Date": "date",
            "level_1": "tic",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    long["date"] = pd.to_datetime(long["date"])
    long = long[["date", "tic", "open", "high", "low", "close", "adj_close", "volume"]]
    long = long.sort_values(["date", "tic"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(DATA_DIR / "clean_prices.csv", index=False)
    print(f"Saved cleaned data to {DATA_DIR / 'clean_prices.csv'}")
    return long


def load_or_download_prices(force: bool = False) -> pd.DataFrame:
    path = DATA_DIR / "clean_prices.csv"
    if path.exists() and not force:
        cached = pd.read_csv(path, parse_dates=["date"])
        cached_start = cached["date"].min()
        if cached_start < pd.Timestamp(FULL_START):
            print(f"Loading cached cleaned prices from {path}")
            return cached
        print(
            "Cached cleaned prices do not include the required indicator warm-up "
            f"period from {DATA_DOWNLOAD_START}; redownloading."
        )
    return download_ohlcv()

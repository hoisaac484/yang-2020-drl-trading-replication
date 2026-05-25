"""Technical indicators used in Yang et al.'s state representation."""

import numpy as np
import pandas as pd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def rsi(close: pd.Series, window: int = 30) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window).mean()
    mean_dev = typical_price.rolling(window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma) / (0.015 * mean_dev.replace(0, np.nan))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
    """Return a DX-style ADX feature aligned with the author repo's dx_30."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr_components = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(window).mean()

    plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx


def add_indicators(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, RSI, CCI, and ADX to a long OHLCV panel."""
    frames = []
    for ticker, group in price_panel.groupby("tic", sort=False):
        g = group.sort_values("date").copy()
        g["macd"] = macd(g["adj_close"])
        g["rsi"] = rsi(g["adj_close"])
        g["cci"] = cci(g["high"], g["low"], g["close"])
        g["adx"] = adx(g["high"], g["low"], g["close"])
        frames.append(g)

    features = pd.concat(frames, ignore_index=True)
    features = features.dropna(subset=["macd", "rsi", "cci", "adx"])
    common_dates = features.groupby("date")["tic"].nunique()
    common_dates = common_dates[common_dates == features["tic"].nunique()].index
    return features[features["date"].isin(common_dates)].sort_values(["date", "tic"])


def add_turbulence(
    feature_panel: pd.DataFrame,
    lookback: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Add a past-looking turbulence index to each date.

    The turbulence value is the Mahalanobis distance of the current return vector
    from the trailing return distribution. Only observations before the current
    day are used for the trailing mean/covariance, which avoids look-ahead.
    """
    out = feature_panel.copy()
    prices = out.pivot(index="date", columns="tic", values="adj_close").sort_index()
    returns = prices.pct_change()
    values = []

    for i, date in enumerate(returns.index):
        if i < min_periods:
            values.append((date, 0.0))
            continue

        history = returns.iloc[max(1, i - lookback):i].dropna()
        current = returns.iloc[i]
        if history.empty or current.isna().any():
            values.append((date, 0.0))
            continue

        mean = history.mean().to_numpy()
        cov = history.cov().to_numpy()
        diff = current.to_numpy() - mean
        inv_cov = np.linalg.pinv(cov)
        turbulence = float(diff.T @ inv_cov @ diff)
        values.append((date, max(turbulence, 0.0)))

    turbulence = pd.DataFrame(values, columns=["date", "turbulence"])
    return out.merge(turbulence, on="date", how="left")

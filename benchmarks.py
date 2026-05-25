"""Buy-and-hold and minimum-variance benchmarks."""

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKER, INITIAL_CASH, TICKERS


def _price_pivot(df: pd.DataFrame, tickers) -> pd.DataFrame:
    return (
        df[df["tic"].isin(tickers)]
        .pivot(index="date", columns="tic", values="adj_close")
        .sort_index()
        .reindex(columns=tickers)
    )


def dia_buy_and_hold(price_df: pd.DataFrame) -> pd.DataFrame:
    prices = _price_pivot(price_df, [BENCHMARK_TICKER])[BENCHMARK_TICKER].dropna()
    shares = INITIAL_CASH / prices.iloc[0]
    return pd.DataFrame({"date": prices.index, "portfolio_value": shares * prices.values})


def equal_weight_buy_and_hold(price_df: pd.DataFrame, tickers=None) -> pd.DataFrame:
    tickers = tickers or TICKERS
    prices = _price_pivot(price_df, tickers).dropna()
    weights = np.repeat(1 / len(tickers), len(tickers))
    shares = INITIAL_CASH * weights / prices.iloc[0].values
    values = prices.to_numpy() @ shares
    return pd.DataFrame({"date": prices.index, "portfolio_value": values})


def minimum_variance_weights(train_price_df: pd.DataFrame, tickers=None) -> pd.Series:
    tickers = tickers or TICKERS
    prices = _price_pivot(train_price_df, tickers).dropna()
    returns = prices.pct_change().dropna()
    cov = returns.cov().to_numpy() * 252
    n = len(tickers)

    try:
        from scipy.optimize import minimize

        def objective(w):
            return float(w.T @ cov @ w)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n
        result = minimize(
            objective,
            np.repeat(1 / n, n),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success:
            return pd.Series(result.x, index=tickers, name="min_var_weight")
    except Exception as exc:
        print(f"Minimum-variance optimizer unavailable, using inverse-vol fallback: {exc}")

    inv_vol = 1 / returns.std().replace(0, np.nan)
    return (inv_vol / inv_vol.sum()).fillna(1 / n).rename("min_var_weight")


def minimum_variance_buy_and_hold(train_price_df: pd.DataFrame, test_price_df: pd.DataFrame, tickers=None):
    tickers = tickers or TICKERS
    weights = minimum_variance_weights(train_price_df, tickers)
    prices = _price_pivot(test_price_df, tickers).dropna()
    shares = INITIAL_CASH * weights.reindex(tickers).values / prices.iloc[0].values
    values = prices.to_numpy() @ shares
    return pd.DataFrame({"date": prices.index, "portfolio_value": values}), weights

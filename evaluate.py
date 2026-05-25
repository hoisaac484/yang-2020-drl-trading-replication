"""Evaluation helpers for RL strategies and benchmarks."""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE


def run_trading_episode(model, env):
    obs, _ = env.reset()
    done = False
    records = []
    action_records = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append(
            {
                "date": pd.to_datetime(info["date"]),
                "portfolio_value": info["portfolio_value"],
                "reward": reward,
                "unscaled_reward": reward / env.reward_scaling,
                "cash": info["cash"],
                "transaction_costs": info["transaction_costs"],
                "trades": info["trades"],
                "turbulence": info.get("turbulence", 0.0),
                "risk_off": info.get("risk_off", False),
            }
        )
        action_row = {"date": pd.to_datetime(info["date"])}
        for ticker, shares in zip(env.tickers, info["trade_shares"]):
            action_row[ticker] = shares
        action_records.append(action_row)

    values = pd.DataFrame(records)
    actions = pd.DataFrame(action_records)
    return values, actions


def performance_metrics(values: pd.Series, initial_value: float, rf: float = RISK_FREE_RATE) -> dict:
    values = values.dropna().astype(float)
    returns = values.pct_change().dropna()
    if values.empty:
        raise ValueError("Cannot compute metrics for an empty portfolio value series.")

    cumulative_return = values.iloc[-1] / initial_value - 1
    years = max(len(returns) / 252, 1 / 252)
    annual_return = (1 + cumulative_return) ** (1 / years) - 1
    annual_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = np.nan if annual_vol == 0 else (annual_return - rf) / annual_vol
    running_max = values.cummax()
    drawdown = values / running_max - 1

    return {
        "final_portfolio_value": values.iloc[-1],
        "cumulative_return": cumulative_return,
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
    }


def validation_sharpe(values: pd.Series, periods_per_window: float = 4.0) -> float:
    """Repo-style validation Sharpe used for quarterly model selection."""
    returns = values.dropna().astype(float).pct_change().dropna()
    vol = returns.std()
    if returns.empty or vol == 0:
        return np.nan
    return float(np.sqrt(periods_per_window) * returns.mean() / vol)


def values_to_returns(values_df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = values_df[["date", "portfolio_value"]].copy()
    out = out.rename(columns={"portfolio_value": name})
    return out

"""Run rolling ensemble robustness checks across multiple random seeds.

This script repeats the final rolling backtest for several seeds, but keeps
models in memory only. It writes summary CSV files and does not save model zips.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed

from benchmarks import (
    dia_buy_and_hold,
    equal_weight_buy_and_hold,
    minimum_variance_buy_and_hold,
)
from config import (
    HMAX,
    INITIAL_CASH,
    MODEL_KWARGS,
    ROBUSTNESS_OUTPUT_DIR,
    REWARD_SCALING,
    ROLLING_VALIDATION_DAYS,
    TEST_START,
    TICKERS,
    TRAIN_END,
    TRAIN_START,
    TRAIN_TIMESTEPS,
    TRANSACTION_COST_PCT,
)
from data import load_or_download_prices
from env import StockTradingEnv
from evaluate import performance_metrics, run_trading_episode, validation_sharpe
from main import prepare_features, split_by_date
from train import MODEL_CLASSES, _has_enough_dates, _quarter_windows, _split_by_date, turbulence_threshold


DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def make_env(df, tickers, initial_cash=INITIAL_CASH, initial_holdings=None, turbulence_limit=None):
    return StockTradingEnv(
        df=df,
        tickers=tickers,
        initial_cash=initial_cash,
        hmax=HMAX,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        reward_scaling=REWARD_SCALING,
        turbulence_threshold=turbulence_limit,
        initial_holdings=initial_holdings,
    )


def train_agent_no_save(name, train_df, tickers, seed, turbulence_limit=None):
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(train_df, tickers, turbulence_limit=turbulence_limit)
    kwargs = dict(MODEL_KWARGS.get(name, {}))
    if name == "ddpg":
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.5 * np.ones(env.action_space.shape[0]),
        )

    model = MODEL_CLASSES[name]("MlpPolicy", env, verbose=0, seed=seed, **kwargs)
    model.learn(total_timesteps=TRAIN_TIMESTEPS[name], progress_bar=False)
    return model


def rolling_ensemble_for_seed(features, tickers, seed, max_windows=None):
    all_dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    cash = INITIAL_CASH
    holdings = np.zeros(len(tickers), dtype=np.float64)
    value_frames = []
    selection_rows = []

    completed = 0
    for idx, (train_end, valid_start, valid_end, trade_start, trade_end) in enumerate(
        _quarter_windows(all_dates, TEST_START),
        start=1,
    ):
        train_df = _split_by_date(features, TRAIN_START, train_end)
        valid_df = _split_by_date(features, valid_start, valid_end)
        trade_df = _split_by_date(features, trade_start, trade_end)

        if train_df.empty or not _has_enough_dates(valid_df) or not _has_enough_dates(trade_df):
            continue

        recent_dates = train_df["date"].drop_duplicates().sort_values().tail(ROLLING_VALIDATION_DAYS)
        recent_df = train_df[train_df["date"].isin(recent_dates)]
        threshold = turbulence_threshold(train_df, recent_df=recent_df)

        print(
            f"Seed {seed}, window {idx}: train to {train_end.date()}, "
            f"validate {valid_start.date()}:{valid_end.date()}, "
            f"trade {trade_start.date()}:{trade_end.date()}"
        )

        quarter_results = {}
        for name in MODEL_CLASSES:
            model = train_agent_no_save(name, train_df, tickers, seed, turbulence_limit=threshold)
            valid_env = make_env(valid_df, tickers, turbulence_limit=threshold)
            valid_values, _ = run_trading_episode(model, valid_env)
            sharpe = validation_sharpe(valid_values["portfolio_value"])
            quarter_results[name] = {"model": model, "validation_sharpe": sharpe}
            print(f"  {name.upper()} validation Sharpe: {sharpe:.3f}")

        selected = max(
            quarter_results,
            key=lambda key: np.nan_to_num(quarter_results[key]["validation_sharpe"], nan=-np.inf),
        )
        selected_model = quarter_results[selected]["model"]
        trade_env = make_env(
            trade_df,
            tickers,
            initial_cash=cash,
            initial_holdings=holdings,
            turbulence_limit=threshold,
        )
        values, _ = run_trading_episode(selected_model, trade_env)
        values["seed"] = seed
        values["selected_model"] = selected
        value_frames.append(values)

        cash = trade_env.cash
        holdings = trade_env.holdings.copy()
        selection_rows.append(
            {
                "seed": seed,
                "quarter": idx,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "trade_start": trade_start,
                "trade_end": trade_end,
                "selected_model": selected,
                "a2c_validation_sharpe": quarter_results["a2c"]["validation_sharpe"],
                "ppo_validation_sharpe": quarter_results["ppo"]["validation_sharpe"],
                "ddpg_validation_sharpe": quarter_results["ddpg"]["validation_sharpe"],
                "ending_portfolio_value": float(values["portfolio_value"].iloc[-1]),
            }
        )

        completed += 1
        if max_windows is not None and completed >= max_windows:
            break

    if not value_frames:
        raise ValueError(f"No trading windows completed for seed {seed}.")

    return pd.concat(value_frames, ignore_index=True), pd.DataFrame(selection_rows)


def benchmark_rows(price_df):
    test_prices = split_by_date(price_df, TEST_START, None)
    train_prices = split_by_date(price_df, TRAIN_START, TRAIN_END)
    series = {
        "DIA buy-and-hold": dia_buy_and_hold(test_prices),
        "Equal-weight 30": equal_weight_buy_and_hold(test_prices, TICKERS),
        "Minimum-variance": minimum_variance_buy_and_hold(train_prices, test_prices, TICKERS)[0],
    }
    rows = []
    for name, values in series.items():
        metrics = performance_metrics(values["portfolio_value"], INITIAL_CASH)
        metrics["strategy"] = name
        rows.append(metrics)
    return rows


def summarise_seed_metrics(seed_metrics):
    metric_cols = [
        "final_portfolio_value",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    rl = seed_metrics[seed_metrics["strategy"] == "RL rolling ensemble"]
    summary = (
        rl[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .rename_axis("statistic")
        .reset_index()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed rolling ensemble robustness checks.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional debugging limit.")
    args = parser.parse_args()

    ROBUSTNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_df = load_or_download_prices(force=False)
    features = prepare_features(price_df)
    stock_features = features[features["tic"].isin(TICKERS)].copy()

    metrics_rows = []
    selection_frames = []

    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")
        rl_values, selections = rolling_ensemble_for_seed(
            stock_features,
            TICKERS,
            seed=seed,
            max_windows=args.max_windows,
        )
        metrics = performance_metrics(rl_values["portfolio_value"], INITIAL_CASH)
        metrics["strategy"] = "RL rolling ensemble"
        metrics["seed"] = seed
        metrics_rows.append(metrics)
        selection_frames.append(selections)
        print(
            f"Seed {seed} finished: final value={metrics['final_portfolio_value']:.2f}, "
            f"Sharpe={metrics['sharpe_ratio']:.3f}"
        )

    for row in benchmark_rows(price_df):
        row["seed"] = "deterministic"
        metrics_rows.append(row)

    seed_metrics = pd.DataFrame(metrics_rows)
    rl_summary = summarise_seed_metrics(seed_metrics)
    selections = pd.concat(selection_frames, ignore_index=True)

    suffix = "" if args.max_windows is None else f"_first_{args.max_windows}_windows"
    detail_path = ROBUSTNESS_OUTPUT_DIR / f"multi_seed_metrics{suffix}.csv"
    summary_path = ROBUSTNESS_OUTPUT_DIR / f"multi_seed_summary{suffix}.csv"
    selections_path = ROBUSTNESS_OUTPUT_DIR / f"multi_seed_selection_summary{suffix}.csv"

    seed_metrics.to_csv(detail_path, index=False)
    rl_summary.to_csv(summary_path, index=False)
    selections.to_csv(selections_path, index=False)

    print("\nRL multi-seed summary:")
    print(rl_summary.round(4).to_string(index=False))
    print(f"\nSaved seed metrics to {detail_path}")
    print(f"Saved RL summary to {summary_path}")
    print(f"Saved selection summary to {selections_path}")


if __name__ == "__main__":
    main()

"""Hyperparameter tuning over the full rolling validation design.

This script tunes PPO, A2C, and DDPG on the same 14 rolling validation
windows used by the final backtest. It does not save temporary model zip files;
it only writes tuning summaries to the outputs folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed

from config import (
    HMAX,
    INITIAL_CASH,
    MODEL_KWARGS,
    NESTED_TUNED_OUTPUT_DIR,
    REWARD_SCALING,
    ROLLING_VALIDATION_DAYS,
    SEED,
    TEST_START,
    TICKERS,
    TRAIN_START,
    TRAIN_TIMESTEPS,
    TRANSACTION_COST_PCT,
)
from data import load_or_download_prices
from env import StockTradingEnv
from evaluate import run_trading_episode, validation_sharpe
from main import prepare_features
from train import MODEL_CLASSES, _has_enough_dates, _quarter_windows, _split_by_date, turbulence_threshold


TUNING_GRIDS = {
    "a2c": [
        {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99},
        {"learning_rate": 7e-4, "n_steps": 10, "gamma": 0.99},
        {"learning_rate": 1e-3, "n_steps": 5, "gamma": 0.99},
        {"learning_rate": 1e-3, "n_steps": 10, "gamma": 0.99},
    ],
    "ppo": [
        {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 64, "gamma": 0.99, "ent_coef": 0.005},
        {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 128, "gamma": 0.99, "ent_coef": 0.005},
        {"learning_rate": 1e-4, "n_steps": 512, "batch_size": 64, "gamma": 0.99, "ent_coef": 0.005},
        {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "gamma": 0.99, "ent_coef": 0.005},
    ],
    "ddpg": [
        {"learning_rate": 1e-3, "buffer_size": 10_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5},
        {"learning_rate": 1e-3, "buffer_size": 50_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5},
        {"learning_rate": 3e-4, "buffer_size": 10_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5},
        {"learning_rate": 1e-3, "buffer_size": 10_000, "batch_size": 128, "gamma": 0.99, "noise_sigma": 0.3},
    ],
}


def make_env(df, tickers, turbulence_limit=None):
    return StockTradingEnv(
        df=df,
        tickers=tickers,
        initial_cash=INITIAL_CASH,
        hmax=HMAX,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        reward_scaling=REWARD_SCALING,
        turbulence_threshold=turbulence_limit,
    )


def stable_config_id(model_name: str, params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{model_name}_{digest}"


def build_model(model_name: str, train_df, tickers, params: dict, turbulence_limit=None):
    set_random_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env(train_df, tickers, turbulence_limit=turbulence_limit)
    kwargs = deepcopy(MODEL_KWARGS.get(model_name, {}))
    kwargs.update(params)

    if model_name == "ddpg":
        noise_sigma = kwargs.pop("noise_sigma", 0.5)
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=noise_sigma * np.ones(env.action_space.shape[0]),
        )

    model = MODEL_CLASSES[model_name]("MlpPolicy", env, verbose=0, seed=SEED, **kwargs)
    model.learn(total_timesteps=TRAIN_TIMESTEPS[model_name], progress_bar=False)
    return model


def rolling_validation_windows(features, max_windows=None):
    all_dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    windows = []
    for idx, (train_end, valid_start, valid_end, trade_start, trade_end) in enumerate(
        _quarter_windows(all_dates, TEST_START),
        start=1,
    ):
        train_df = _split_by_date(features, TRAIN_START, train_end)
        valid_df = _split_by_date(features, valid_start, valid_end)
        if train_df.empty or not _has_enough_dates(valid_df):
            continue
        windows.append(
            {
                "window": idx,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "trade_start": trade_start,
                "trade_end": trade_end,
                "train_df": train_df,
                "valid_df": valid_df,
            }
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def tune_model(model_name: str, candidate_grid: list[dict], windows: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate_num, params in enumerate(candidate_grid, start=1):
        config_id = stable_config_id(model_name, params)
        print(
            f"\nTuning {model_name.upper()} candidate {candidate_num}/{len(candidate_grid)} "
            f"({config_id}): {params}"
        )
        for window in windows:
            train_df = window["train_df"]
            valid_df = window["valid_df"]
            recent_dates = train_df["date"].drop_duplicates().sort_values().tail(ROLLING_VALIDATION_DAYS)
            recent_df = train_df[train_df["date"].isin(recent_dates)]
            threshold = turbulence_threshold(train_df, recent_df=recent_df)

            print(
                f"  Window {window['window']:02d}: train to {window['train_end'].date()}, "
                f"validate {window['valid_start'].date()} to {window['valid_end'].date()}"
            )
            model = build_model(model_name, train_df, TICKERS, params, turbulence_limit=threshold)
            valid_env = make_env(valid_df, TICKERS, turbulence_limit=threshold)
            valid_values, _ = run_trading_episode(model, valid_env)
            sharpe = validation_sharpe(valid_values["portfolio_value"])
            final_value = float(valid_values["portfolio_value"].iloc[-1])
            total_reward = float(valid_values["unscaled_reward"].sum())

            rows.append(
                {
                    "model": model_name,
                    "config_id": config_id,
                    "candidate_num": candidate_num,
                    "params": json.dumps(params, sort_keys=True),
                    "window": window["window"],
                    "train_end": window["train_end"],
                    "valid_start": window["valid_start"],
                    "valid_end": window["valid_end"],
                    "validation_sharpe": sharpe,
                    "validation_final_value": final_value,
                    "validation_total_reward": total_reward,
                }
            )
    return pd.DataFrame(rows)


def summarise_results(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(["model", "config_id", "params"], as_index=False)
        .agg(
            mean_validation_sharpe=("validation_sharpe", "mean"),
            median_validation_sharpe=("validation_sharpe", "median"),
            std_validation_sharpe=("validation_sharpe", "std"),
            mean_validation_final_value=("validation_final_value", "mean"),
            mean_validation_total_reward=("validation_total_reward", "mean"),
            windows=("window", "nunique"),
        )
        .sort_values(["model", "mean_validation_sharpe"], ascending=[True, False])
    )
    return summary


def write_best_configs(summary: pd.DataFrame, output_path: Path) -> dict:
    best = {}
    for model_name, group in summary.groupby("model"):
        row = group.sort_values("mean_validation_sharpe", ascending=False).iloc[0]
        best[model_name] = {
            "config_id": row["config_id"],
            "mean_validation_sharpe": float(row["mean_validation_sharpe"]),
            "params": json.loads(row["params"]),
        }
    output_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Tune PPO/A2C/DDPG over rolling validation windows.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(TUNING_GRIDS),
        default=sorted(TUNING_GRIDS),
        help="Models to tune. Defaults to all.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit for debugging. Omit to use all rolling windows.",
    )
    args = parser.parse_args()

    NESTED_TUNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_df = load_or_download_prices(force=False)
    features = prepare_features(price_df)
    stock_features = features[features["tic"].isin(TICKERS)].copy()
    windows = rolling_validation_windows(stock_features, max_windows=args.max_windows)
    if not windows:
        raise ValueError("No rolling validation windows were available.")

    print(f"Using {len(windows)} rolling validation windows.")
    if args.max_windows is None and len(windows) != 14:
        print("Note: latest available data produced a window count different from 14.")

    detail_frames = []
    for model_name in args.models:
        detail_frames.append(tune_model(model_name, TUNING_GRIDS[model_name], windows))

    detail = pd.concat(detail_frames, ignore_index=True)
    summary = summarise_results(detail)

    detail_path = NESTED_TUNED_OUTPUT_DIR / "hyperparam_tuning_detail.csv"
    summary_path = NESTED_TUNED_OUTPUT_DIR / "hyperparam_tuning_summary.csv"
    best_path = NESTED_TUNED_OUTPUT_DIR / "best_hyperparams.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    best = write_best_configs(summary, best_path)

    print("\nBest settings by average rolling validation Sharpe:")
    for model_name, info in best.items():
        print(
            f"{model_name.upper()}: Sharpe={info['mean_validation_sharpe']:.3f}, "
            f"params={info['params']}"
        )
    print(f"\nSaved detailed results to {detail_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved best configs to {best_path}")


if __name__ == "__main__":
    main()

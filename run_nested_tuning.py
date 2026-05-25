"""Experiment 2: nested walk-forward hyperparameter tuning.

This extension keeps the same trading environment as the author-style
replication, but chooses model settings using only validation folds that would
have been available before each trading window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from copy import deepcopy

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
    NESTED_TUNED_OUTPUT_DIR,
    REWARD_SCALING,
    ROLLING_TRADE_DAYS,
    SEED,
    TEST_START,
    TICKERS,
    TRAIN_END,
    TRAIN_START,
    TRANSACTION_COST_PCT,
)
from data import load_or_download_prices
from env import StockTradingEnv
from evaluate import performance_metrics, run_trading_episode, validation_sharpe
from main import prepare_features, split_by_date
import plots
from train import MODEL_CLASSES, _has_enough_dates, _split_by_date, turbulence_threshold


TUNING_SEEDS = [1, 2, 3, 4, 5]
VALIDATION_WINDOWS = [63, 126]
INSTABILITY_PENALTY = 0.5


AUTHOR_STYLE_CANDIDATES = {
    "a2c": {"timesteps": 30_000, "kwargs": {"gamma": 0.99}, "validation_days": 63},
    "ppo": {
        "timesteps": 100_000,
        "kwargs": {"ent_coef": 0.005, "n_steps": 512, "batch_size": 64, "gamma": 0.99},
        "validation_days": 63,
    },
    "ddpg": {
        "timesteps": 10_000,
        "kwargs": {"buffer_size": 10_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5},
        "validation_days": 63,
    },
}


TUNING_GRIDS = {
    "a2c": [
        {"timesteps": 10_000, "kwargs": {"gamma": 0.99}},
        {"timesteps": 30_000, "kwargs": {"gamma": 0.99}},
        {"timesteps": 10_000, "kwargs": {"learning_rate": 1e-3, "n_steps": 10, "gamma": 0.99}},
        {"timesteps": 30_000, "kwargs": {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99}},
    ],
    "ppo": [
        {"timesteps": 50_000, "kwargs": {"ent_coef": 0.005, "n_steps": 512, "batch_size": 64, "gamma": 0.99}},
        {"timesteps": 100_000, "kwargs": {"ent_coef": 0.005, "n_steps": 512, "batch_size": 64, "gamma": 0.99}},
        {"timesteps": 100_000, "kwargs": {"ent_coef": 0.005, "n_steps": 512, "batch_size": 128, "gamma": 0.99}},
        {"timesteps": 100_000, "kwargs": {"ent_coef": 0.005, "n_steps": 1024, "batch_size": 128, "gamma": 0.99}},
    ],
    "ddpg": [
        {"timesteps": 10_000, "kwargs": {"buffer_size": 10_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5}},
        {"timesteps": 10_000, "kwargs": {"buffer_size": 10_000, "batch_size": 128, "gamma": 0.99, "noise_sigma": 0.3}},
        {"timesteps": 10_000, "kwargs": {"learning_rate": 1e-3, "buffer_size": 10_000, "batch_size": 64, "gamma": 0.99, "noise_sigma": 0.5}},
        {"timesteps": 10_000, "kwargs": {"learning_rate": 1e-3, "buffer_size": 10_000, "batch_size": 128, "gamma": 0.99, "noise_sigma": 0.3}},
    ],
}


GRID_PRESETS = {
    "author": {
        model_name: [{"timesteps": cfg["timesteps"], "kwargs": cfg["kwargs"]}]
        for model_name, cfg in AUTHOR_STYLE_CANDIDATES.items()
    },
    "small": {
        "a2c": TUNING_GRIDS["a2c"][:2],
        "ppo": TUNING_GRIDS["ppo"][:2],
        "ddpg": TUNING_GRIDS["ddpg"][:2],
    },
    "full": TUNING_GRIDS,
}


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


def candidate_id(model_name: str, candidate: dict) -> str:
    payload = json.dumps(candidate, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{model_name}_{digest}"


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def candidate_grid(model_name: str, validation_windows=None, grid_preset: str = "full") -> list[dict]:
    validation_windows = validation_windows or VALIDATION_WINDOWS
    if grid_preset not in GRID_PRESETS:
        raise ValueError(f"Unknown grid preset '{grid_preset}'. Use one of {sorted(GRID_PRESETS)}.")

    candidates = []
    for base in GRID_PRESETS[grid_preset][model_name]:
        for validation_days in validation_windows:
            candidate = deepcopy(base)
            candidate["validation_days"] = validation_days
            candidate["candidate_id"] = candidate_id(model_name, candidate)
            candidates.append(candidate)
    return candidates


def set_seed(seed: int):
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model_name: str, train_df, tickers, candidate: dict, seed: int, turbulence_limit=None):
    set_seed(seed)
    env = make_env(train_df, tickers, turbulence_limit=turbulence_limit)
    kwargs = deepcopy(candidate["kwargs"])

    if model_name == "ddpg":
        noise_sigma = kwargs.pop("noise_sigma", 0.5)
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=noise_sigma * np.ones(env.action_space.shape[0]),
        )

    model = MODEL_CLASSES[model_name]("MlpPolicy", env, verbose=0, seed=seed, **kwargs)
    model.learn(total_timesteps=candidate["timesteps"], progress_bar=False)
    return model


def trade_windows(dates: pd.Series, test_start: str = TEST_START):
    dates = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values().reset_index(drop=True)
    start_positions = dates.index[dates >= pd.Timestamp(test_start)]
    if len(start_positions) == 0:
        return
    trade_start_pos = int(start_positions[0])

    while trade_start_pos < len(dates):
        trade_end_pos = min(trade_start_pos + ROLLING_TRADE_DAYS - 1, len(dates) - 1)
        yield trade_start_pos, dates.iloc[trade_start_pos], dates.iloc[trade_end_pos]
        trade_start_pos += ROLLING_TRADE_DAYS


def window_for_trade(dates: pd.Series, trade_start_pos: int, validation_days: int):
    valid_start_pos = trade_start_pos - validation_days
    valid_end_pos = trade_start_pos - 1
    train_end_pos = valid_start_pos - 1
    if train_end_pos < 0 or valid_start_pos < 0:
        return None
    return {
        "train_end": dates.iloc[train_end_pos],
        "valid_start": dates.iloc[valid_start_pos],
        "valid_end": dates.iloc[valid_end_pos],
    }


def prior_tuning_folds(dates: pd.Series, current_trade_start_pos: int, validation_days: int, max_folds=None):
    folds = []
    for prior_pos, _, _ in trade_windows(dates, TEST_START):
        if prior_pos >= current_trade_start_pos:
            break
        fold = window_for_trade(dates, prior_pos, validation_days)
        if fold is not None:
            folds.append({"trade_start_pos": prior_pos, **fold})
    if max_folds is not None and len(folds) > max_folds:
        folds = folds[-max_folds:]
    return folds


def score_candidate(model_name, candidate, folds, features, seeds, tickers):
    rows = []
    for fold_number, fold in enumerate(folds, start=1):
        train_df = _split_by_date(features, TRAIN_START, fold["train_end"])
        valid_df = _split_by_date(features, fold["valid_start"], fold["valid_end"])
        if train_df.empty or not _has_enough_dates(valid_df):
            continue

        recent_dates = train_df["date"].drop_duplicates().sort_values().tail(candidate["validation_days"])
        recent_df = train_df[train_df["date"].isin(recent_dates)]
        threshold = turbulence_threshold(train_df, recent_df=recent_df)

        for seed in seeds:
            log(
                "    scoring "
                f"model={model_name.upper()} candidate={candidate['candidate_id']} "
                f"valid_days={candidate['validation_days']} fold={fold_number}/{len(folds)} seed={seed}"
            )
            model = train_model(model_name, train_df, tickers, candidate, seed, turbulence_limit=threshold)
            valid_env = make_env(valid_df, tickers, turbulence_limit=threshold)
            valid_values, _ = run_trading_episode(model, valid_env)
            sharpe = validation_sharpe(valid_values["portfolio_value"])
            final_value = float(valid_values["portfolio_value"].iloc[-1])
            log(
                "      result "
                f"sharpe={sharpe:.4f} final_value={final_value:,.2f}"
            )
            rows.append(
                {
                    "model": model_name,
                    "candidate_id": candidate["candidate_id"],
                    "candidate": json.dumps(candidate, sort_keys=True),
                    "fold_trade_start_pos": fold["trade_start_pos"],
                    "validation_days": candidate["validation_days"],
                    "seed": seed,
                    "validation_sharpe": sharpe,
                    "validation_final_value": final_value,
                }
            )
    return rows


def select_candidates_for_window(
    model_names,
    features,
    dates,
    trade_start_pos,
    seeds,
    max_tuning_folds,
    validation_windows,
    grid_preset,
):
    selected = {}
    detail_rows = []
    choice_rows = []

    for model_name in model_names:
        best = None
        candidates = candidate_grid(model_name, validation_windows=validation_windows, grid_preset=grid_preset)
        log(f"  tuning {model_name.upper()} with {len(candidates)} candidates")
        for candidate_index, candidate in enumerate(candidates, start=1):
            folds = prior_tuning_folds(
                dates,
                current_trade_start_pos=trade_start_pos,
                validation_days=candidate["validation_days"],
                max_folds=max_tuning_folds,
            )
            if not folds:
                continue
            log(
                f"  candidate {candidate_index}/{len(candidates)} "
                f"{candidate['candidate_id']} valid_days={candidate['validation_days']} "
                f"timesteps={candidate['timesteps']} folds={len(folds)} seeds={len(seeds)}"
            )
            rows = score_candidate(model_name, candidate, folds, features, seeds, TICKERS)
            detail_rows.extend(rows)
            sharpes = pd.Series([row["validation_sharpe"] for row in rows], dtype=float).dropna()
            if sharpes.empty:
                continue
            score = float(sharpes.mean() - INSTABILITY_PENALTY * sharpes.std(ddof=0))
            summary = {
                "model": model_name,
                "candidate_id": candidate["candidate_id"],
                "candidate": candidate,
                "mean_validation_sharpe": float(sharpes.mean()),
                "std_validation_sharpe": float(sharpes.std(ddof=0)),
                "robust_score": score,
                "validation_days": candidate["validation_days"],
                "folds": len(folds),
                "seed_runs": int(sharpes.shape[0]),
            }
            log(
                "  candidate summary "
                f"{candidate['candidate_id']}: mean_sharpe={summary['mean_validation_sharpe']:.4f}, "
                f"std={summary['std_validation_sharpe']:.4f}, score={summary['robust_score']:.4f}"
            )
            if best is None or summary["robust_score"] > best["robust_score"]:
                best = summary

        if best is None:
            fallback = deepcopy(AUTHOR_STYLE_CANDIDATES[model_name])
            fallback["candidate_id"] = f"{model_name}_author_fallback"
            best = {
                "model": model_name,
                "candidate_id": fallback["candidate_id"],
                "candidate": fallback,
                "mean_validation_sharpe": np.nan,
                "std_validation_sharpe": np.nan,
                "robust_score": np.nan,
                "validation_days": fallback["validation_days"],
                "folds": 0,
                "seed_runs": 0,
            }

        selected[model_name] = best["candidate"]
        log(
            f"  selected tuning candidate for {model_name.upper()}: "
            f"{best['candidate_id']} valid_days={best['validation_days']} "
            f"score={best['robust_score']}"
        )
        choice_rows.append(
            {
                "model": model_name,
                "candidate_id": best["candidate_id"],
                "candidate": json.dumps(best["candidate"], sort_keys=True),
                "mean_validation_sharpe": best["mean_validation_sharpe"],
                "std_validation_sharpe": best["std_validation_sharpe"],
                "robust_score": best["robust_score"],
                "validation_days": best["validation_days"],
                "folds": best["folds"],
                "seed_runs": best["seed_runs"],
            }
        )

    return selected, detail_rows, choice_rows


def write_checkpoint(
    value_frames,
    action_frames,
    selection_rows,
    candidate_score_rows,
    tuning_choice_rows,
):
    if value_frames:
        pd.concat(value_frames, ignore_index=True).to_csv(
            NESTED_TUNED_OUTPUT_DIR / "rl_ensemble_values_checkpoint.csv", index=False
        )
    if action_frames:
        pd.concat(action_frames, ignore_index=True).to_csv(
            NESTED_TUNED_OUTPUT_DIR / "rl_ensemble_actions_checkpoint.csv", index=False
        )
    pd.DataFrame(selection_rows).to_csv(
        NESTED_TUNED_OUTPUT_DIR / "rolling_selection_summary_checkpoint.csv", index=False
    )
    pd.DataFrame(candidate_score_rows).to_csv(
        NESTED_TUNED_OUTPUT_DIR / "candidate_scores_checkpoint.csv", index=False
    )
    pd.DataFrame(tuning_choice_rows).to_csv(
        NESTED_TUNED_OUTPUT_DIR / "nested_tuning_choices_checkpoint.csv", index=False
    )


def read_checkpoint_csv(name: str) -> pd.DataFrame:
    path = NESTED_TUNED_OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def reconstruct_holdings_from_actions(actions: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    if actions.empty:
        return np.zeros(len(tickers), dtype=np.float64)
    missing = [ticker for ticker in tickers if ticker not in actions.columns]
    if missing:
        raise ValueError(
            "Cannot resume because the action checkpoint is missing ticker columns: "
            f"{missing}"
        )
    return actions[tickers].astype(float).sum(axis=0).to_numpy(dtype=np.float64)


def load_resume_state(tickers: list[str]):
    selections = read_checkpoint_csv("rolling_selection_summary_checkpoint.csv")
    if selections.empty:
        log("Resume requested, but no completed checkpoint was found. Starting from window 1.")
        return None

    values = read_checkpoint_csv("rl_ensemble_values_checkpoint.csv")
    actions = read_checkpoint_csv("rl_ensemble_actions_checkpoint.csv")
    candidate_scores = read_checkpoint_csv("candidate_scores_checkpoint.csv")
    tuning_choices = read_checkpoint_csv("nested_tuning_choices_checkpoint.csv")

    if values.empty or actions.empty:
        raise ValueError(
            "Cannot resume because selection checkpoint exists but values/actions checkpoints are missing."
        )

    completed_window = int(selections["quarter"].max())
    last_selection = selections.loc[selections["quarter"].idxmax()]
    cash = float(last_selection["ending_cash"])

    if "ending_holdings" in selections.columns and pd.notna(last_selection.get("ending_holdings")):
        holdings = np.asarray(json.loads(last_selection["ending_holdings"]), dtype=np.float64)
    else:
        holdings = reconstruct_holdings_from_actions(actions, tickers)

    if holdings.shape[0] != len(tickers):
        raise ValueError(
            f"Cannot resume because holdings length {holdings.shape[0]} does not match "
            f"ticker count {len(tickers)}."
        )

    for df in (values, actions):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    for col in ("trade_start", "trade_end"):
        if col in selections.columns:
            selections[col] = pd.to_datetime(selections[col])

    log(
        "Resuming from checkpoint "
        f"after window {completed_window}: cash={cash:,.2f}, "
        f"portfolio_value={float(last_selection['ending_portfolio_value']):,.2f}"
    )
    return {
        "completed_window": completed_window,
        "cash": cash,
        "holdings": holdings,
        "value_frames": [values],
        "action_frames": [actions],
        "selection_rows": selections.to_dict("records"),
        "candidate_score_rows": candidate_scores.to_dict("records"),
        "tuning_choice_rows": tuning_choices.to_dict("records"),
    }


def run_nested_tuned_ensemble(
    features,
    tickers,
    seeds,
    max_windows=None,
    max_tuning_folds=None,
    validation_windows=None,
    grid_preset="full",
    resume=False,
):
    validation_windows = validation_windows or VALIDATION_WINDOWS
    dates = pd.Series(pd.to_datetime(features["date"].unique())).drop_duplicates().sort_values().reset_index(drop=True)
    cash = INITIAL_CASH
    holdings = np.zeros(len(tickers), dtype=np.float64)
    value_frames = []
    action_frames = []
    selection_rows = []
    candidate_score_rows = []
    tuning_choice_rows = []
    completed_window = 0

    if resume:
        resume_state = load_resume_state(tickers)
        if resume_state is not None:
            completed_window = resume_state["completed_window"]
            cash = resume_state["cash"]
            holdings = resume_state["holdings"]
            value_frames = resume_state["value_frames"]
            action_frames = resume_state["action_frames"]
            selection_rows = resume_state["selection_rows"]
            candidate_score_rows = resume_state["candidate_score_rows"]
            tuning_choice_rows = resume_state["tuning_choice_rows"]

    for idx, (trade_start_pos, trade_start, trade_end) in enumerate(trade_windows(dates, TEST_START), start=1):
        if idx <= completed_window:
            continue
        if max_windows is not None and idx > max_windows:
            break

        log(f"\nNested tuned window {idx}: trade {trade_start.date()} to {trade_end.date()}")
        selected_candidates, detail_rows, choice_rows = select_candidates_for_window(
            MODEL_CLASSES.keys(),
            features,
            dates,
            trade_start_pos,
            seeds=seeds,
            max_tuning_folds=max_tuning_folds,
            validation_windows=validation_windows,
            grid_preset=grid_preset,
        )
        candidate_score_rows.extend({"trade_window": idx, **row} for row in detail_rows)
        tuning_choice_rows.extend({"trade_window": idx, **row} for row in choice_rows)

        quarter_results = {}
        for model_name, candidate in selected_candidates.items():
            latest = window_for_trade(dates, trade_start_pos, candidate["validation_days"])
            if latest is None:
                raise ValueError(f"No validation window available for {model_name} in trade window {idx}.")

            train_df = _split_by_date(features, TRAIN_START, latest["train_end"])
            valid_df = _split_by_date(features, latest["valid_start"], latest["valid_end"])
            trade_df = _split_by_date(features, trade_start, trade_end)
            if train_df.empty or not _has_enough_dates(valid_df) or not _has_enough_dates(trade_df):
                raise ValueError(f"Insufficient data for nested tuned trade window {idx}.")

            recent_dates = train_df["date"].drop_duplicates().sort_values().tail(candidate["validation_days"])
            recent_df = train_df[train_df["date"].isin(recent_dates)]
            threshold = turbulence_threshold(train_df, recent_df=recent_df)

            log(
                f"  {model_name.upper()} candidate {candidate['candidate_id']} "
                f"valid_days={candidate['validation_days']} timesteps={candidate['timesteps']}"
            )
            model = train_model(model_name, train_df, tickers, candidate, SEED, turbulence_limit=threshold)
            valid_env = make_env(valid_df, tickers, turbulence_limit=threshold)
            valid_values, _ = run_trading_episode(model, valid_env)
            sharpe = validation_sharpe(valid_values["portfolio_value"])
            quarter_results[model_name] = {
                "model": model,
                "candidate": candidate,
                "validation_sharpe": sharpe,
                "validation_window": latest,
                "turbulence_threshold": threshold,
            }
            log(f"    latest validation Sharpe: {sharpe:.3f}")

        selected_model_name = max(
            quarter_results,
            key=lambda key: np.nan_to_num(quarter_results[key]["validation_sharpe"], nan=-np.inf),
        )
        selected_result = quarter_results[selected_model_name]
        selected_model = selected_result["model"]
        trade_df = _split_by_date(features, trade_start, trade_end)
        trade_env = make_env(
            trade_df,
            tickers,
            initial_cash=cash,
            initial_holdings=holdings,
            turbulence_limit=selected_result["turbulence_threshold"],
        )
        values, actions = run_trading_episode(selected_model, trade_env)
        values["selected_model"] = selected_model_name
        actions["selected_model"] = selected_model_name
        value_frames.append(values)
        action_frames.append(actions)

        cash = trade_env.cash
        holdings = trade_env.holdings.copy()

        row = {
            "quarter": idx,
            "trade_start": trade_start,
            "trade_end": trade_end,
            "selected_model": selected_model_name,
            "selected_candidate_id": selected_result["candidate"]["candidate_id"],
            "selected_validation_days": selected_result["candidate"]["validation_days"],
            "selected_timesteps": selected_result["candidate"]["timesteps"],
            "selected_kwargs": json.dumps(selected_result["candidate"]["kwargs"], sort_keys=True),
            "ending_cash": cash,
            "ending_holdings": json.dumps(holdings.tolist()),
            "ending_portfolio_value": float(values["portfolio_value"].iloc[-1]),
        }
        for model_name, result in quarter_results.items():
            candidate = result["candidate"]
            row[f"{model_name}_candidate_id"] = candidate["candidate_id"]
            row[f"{model_name}_validation_days"] = candidate["validation_days"]
            row[f"{model_name}_timesteps"] = candidate["timesteps"]
            row[f"{model_name}_latest_validation_sharpe"] = result["validation_sharpe"]
        selection_rows.append(row)
        log(f"  Selected {selected_model_name.upper()} for trading.")
        write_checkpoint(
            value_frames,
            action_frames,
            selection_rows,
            candidate_score_rows,
            tuning_choice_rows,
        )
        log(f"  checkpoint saved after window {idx}")

    if not value_frames:
        raise ValueError("Nested tuned ensemble produced no trading windows.")

    return (
        pd.concat(value_frames, ignore_index=True),
        pd.concat(action_frames, ignore_index=True),
        pd.DataFrame(selection_rows),
        pd.DataFrame(candidate_score_rows),
        pd.DataFrame(tuning_choice_rows),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2 nested walk-forward tuning.")
    parser.add_argument("--seeds", nargs="+", type=int, default=TUNING_SEEDS)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional debug limit.")
    parser.add_argument(
        "--validation-windows",
        nargs="+",
        type=int,
        default=VALIDATION_WINDOWS,
        help="Validation-window candidates in trading days.",
    )
    parser.add_argument(
        "--grid-preset",
        choices=sorted(GRID_PRESETS),
        default="full",
        help="Candidate grid size. 'author' tunes only author-style settings plus validation-window choice.",
    )
    parser.add_argument(
        "--max-tuning-folds",
        type=int,
        default=None,
        help="Use only the latest N prior folds per trade window. Omit for all prior folds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from *_checkpoint.csv files in outputs/nested_tuned instead of starting at window 1.",
    )
    args = parser.parse_args()

    NESTED_TUNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(
        "Starting nested tuning run "
        f"seeds={args.seeds} validation_windows={args.validation_windows} "
        f"grid_preset={args.grid_preset} max_tuning_folds={args.max_tuning_folds} "
        f"max_windows={args.max_windows} resume={args.resume}"
    )
    price_df = load_or_download_prices(force=False)
    features = prepare_features(price_df)
    stock_features = features[features["tic"].isin(TICKERS)].copy()

    values, actions, selections, candidate_scores, tuning_choices = run_nested_tuned_ensemble(
        stock_features,
        TICKERS,
        seeds=args.seeds,
        max_windows=args.max_windows,
        max_tuning_folds=args.max_tuning_folds,
        validation_windows=args.validation_windows,
        grid_preset=args.grid_preset,
        resume=args.resume,
    )

    values.to_csv(NESTED_TUNED_OUTPUT_DIR / "rl_ensemble_values.csv", index=False)
    actions.to_csv(NESTED_TUNED_OUTPUT_DIR / "rl_ensemble_actions.csv", index=False)
    selections.to_csv(NESTED_TUNED_OUTPUT_DIR / "rolling_selection_summary.csv", index=False)
    candidate_scores.to_csv(NESTED_TUNED_OUTPUT_DIR / "candidate_scores.csv", index=False)
    tuning_choices.to_csv(NESTED_TUNED_OUTPUT_DIR / "nested_tuning_choices.csv", index=False)

    test_prices = split_by_date(price_df, TEST_START, None)
    train_prices = split_by_date(price_df, TRAIN_START, TRAIN_END)
    dia_values = dia_buy_and_hold(test_prices)
    equal_values = equal_weight_buy_and_hold(test_prices, TICKERS)
    minvar_values, minvar_weights = minimum_variance_buy_and_hold(train_prices, test_prices, TICKERS)
    minvar_weights.to_csv(NESTED_TUNED_OUTPUT_DIR / "minimum_variance_weights.csv")

    series = {
        "Nested tuned RL ensemble": values,
        "DIA buy-and-hold": dia_values,
        "Equal-weight 30": equal_values,
        "Minimum-variance": minvar_values,
    }
    rows = []
    for name, series_values in series.items():
        metrics = performance_metrics(series_values["portfolio_value"], INITIAL_CASH)
        metrics["strategy"] = name
        rows.append(metrics)
    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(NESTED_TUNED_OUTPUT_DIR / "performance_summary.csv")

    original_output_dir = plots.OUTPUT_DIR
    plots.OUTPUT_DIR = NESTED_TUNED_OUTPUT_DIR
    try:
        plots.make_all_plots(series, actions)
    finally:
        plots.OUTPUT_DIR = original_output_dir

    log("\nNested tuned performance summary:")
    print(summary.round(4))
    log(f"\nOutputs saved in {NESTED_TUNED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

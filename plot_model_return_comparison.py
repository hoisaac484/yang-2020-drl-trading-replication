"""Replay saved A2C/PPO/DDPG models and compare them with the ensemble."""

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO

from config import DATA_DIR, INITIAL_CASH, MODEL_DIR, OUTPUT_DIR, TEST_START, TICKERS, TRAIN_START
from evaluate import performance_metrics, run_trading_episode
from plots import plot_model_vs_ensemble_returns
from train import _quarter_windows, _split_by_date, make_env, turbulence_threshold


MODEL_CLASSES = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG}


def load_features():
    path = DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run main.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def run_single_model_strategy(features, model_name):
    cash = INITIAL_CASH
    holdings = np.zeros(len(TICKERS), dtype=np.float64)
    frames = []
    dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()

    for idx, (train_end, valid_start, valid_end, trade_start, trade_end) in enumerate(
        _quarter_windows(dates, TEST_START),
        start=1,
    ):
        train_df = _split_by_date(features, TRAIN_START, train_end)
        trade_df = _split_by_date(features, trade_start, trade_end)
        recent_dates = train_df["date"].drop_duplicates().sort_values().tail(63)
        recent_df = train_df[train_df["date"].isin(recent_dates)]
        threshold = turbulence_threshold(train_df, recent_df)
        run_label = f"q{idx:02d}_{trade_start.strftime('%Y%m%d')}_{trade_end.strftime('%Y%m%d')}"
        model_path = MODEL_DIR / f"{run_label}_{model_name}.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")

        model = MODEL_CLASSES[model_name].load(model_path)
        env = make_env(
            trade_df,
            TICKERS,
            initial_cash=cash,
            initial_holdings=holdings,
            turbulence_limit=threshold,
        )
        values, _ = run_trading_episode(model, env)
        values["strategy"] = model_name.upper()
        frames.append(values)
        cash = env.cash
        holdings = env.holdings.copy()

    return pd.concat(frames, ignore_index=True)


def main():
    features = load_features()
    series = {}
    for model_name in MODEL_CLASSES:
        values = run_single_model_strategy(features, model_name)
        values.to_csv(OUTPUT_DIR / f"{model_name}_trade_values.csv", index=False)
        series[model_name.upper()] = values.set_index("date")["portfolio_value"]

    ensemble = pd.read_csv(OUTPUT_DIR / "rl_ensemble_values.csv", parse_dates=["date"])
    series["Ensemble"] = ensemble.set_index("date")["portfolio_value"]
    wide = pd.concat(series, axis=1).sort_index()
    wide.to_csv(OUTPUT_DIR / "model_vs_ensemble_values.csv")
    plot_model_vs_ensemble_returns(wide)

    rows = []
    for name in wide.columns:
        metrics = performance_metrics(wide[name].dropna(), INITIAL_CASH)
        metrics["strategy"] = name
        rows.append(metrics)
    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(OUTPUT_DIR / "model_vs_ensemble_summary.csv")
    print(summary.round(4))
    print(f"Saved {OUTPUT_DIR / 'model_vs_ensemble_cumulative_returns.png'}")


if __name__ == "__main__":
    main()

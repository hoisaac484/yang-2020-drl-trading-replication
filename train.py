"""Model training and validation selection."""

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed

from config import (
    HMAX,
    ENABLE_TURBULENCE_CONTROL,
    INITIAL_CASH,
    MODEL_DIR,
    MODEL_KWARGS,
    ROLLING_TRADE_DAYS,
    ROLLING_VALIDATION_DAYS,
    SEED,
    SAVE_TRAINING_CURVES,
    SAVE_VALIDATION_DETAILS,
    TEST_START,
    TRAINING_CURVE_EVAL_FREQ,
    TRAIN_TIMESTEPS,
    TRANSACTION_COST_PCT,
    REWARD_SCALING,
    TURBULENCE_THRESHOLD_QUANTILE,
    USE_AUTHOR_TURBULENCE_LOGIC,
)
from env import StockTradingEnv
from evaluate import performance_metrics, run_trading_episode, validation_sharpe


MODEL_CLASSES = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG}


class ValidationRewardCallback(BaseCallback):
    """Evaluate validation reward at fixed training timesteps."""

    def __init__(self, valid_df, tickers, run_label, model_name, turbulence_limit=None, eval_freq=10_000):
        super().__init__(verbose=0)
        self.valid_df = valid_df
        self.tickers = tickers
        self.run_label = run_label
        self.model_name = model_name
        self.turbulence_limit = turbulence_limit
        self.eval_freq = eval_freq
        self.records = []

    def _evaluate(self):
        env = make_env(self.valid_df, self.tickers, turbulence_limit=self.turbulence_limit)
        values, _ = run_trading_episode(self.model, env)
        scaled_reward = values["reward"].sum() if "reward" in values.columns else np.nan
        unscaled_reward = (
            values["unscaled_reward"].sum()
            if "unscaled_reward" in values.columns
            else values["portfolio_value"].iloc[-1] - INITIAL_CASH
        )
        self.records.append(
            {
                "run_label": self.run_label,
                "model": self.model_name,
                "timesteps": self.num_timesteps,
                "validation_scaled_reward": scaled_reward,
                "validation_unscaled_reward": unscaled_reward,
                "validation_final_value": values["portfolio_value"].iloc[-1],
                "validation_sharpe": validation_sharpe(values["portfolio_value"]),
            }
        )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.records or self.records[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()
        if self.records:
            path = MODEL_DIR / f"{self.run_label}_{self.model_name}_training_curve.csv"
            pd.DataFrame(self.records).to_csv(path, index=False)


def turbulence_threshold(train_df, recent_df=None):
    if not ENABLE_TURBULENCE_CONTROL or "turbulence" not in train_df.columns:
        return None
    values = train_df.groupby("date")["turbulence"].first()
    values = values[values > 0]
    if values.empty:
        return None
    base_threshold = float(values.quantile(TURBULENCE_THRESHOLD_QUANTILE))
    if not USE_AUTHOR_TURBULENCE_LOGIC:
        return base_threshold

    recent = recent_df if recent_df is not None else train_df
    recent_values = recent.groupby("date")["turbulence"].first()
    recent_values = recent_values[recent_values > 0]
    if recent_values.empty:
        return float(values.max())
    if recent_values.mean() > base_threshold:
        return base_threshold
    return float(values.max())


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


def train_agent(
    name: str,
    train_df,
    tickers,
    run_label=None,
    turbulence_limit=None,
    valid_df=None,
    log_training_curve=False,
):
    """Train one SB3 agent with moderate defaults."""
    set_random_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env(train_df, tickers, turbulence_limit=turbulence_limit)
    cls = MODEL_CLASSES[name]
    kwargs = dict(MODEL_KWARGS.get(name, {}))
    if name == "ddpg":
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.5 * np.ones(env.action_space.shape[0]),
        )

    print(f"Training {name.upper()} for {TRAIN_TIMESTEPS[name]:,} timesteps...")
    model = cls("MlpPolicy", env, verbose=0, seed=SEED, **kwargs)
    callback = None
    if SAVE_TRAINING_CURVES and log_training_curve and valid_df is not None and run_label is not None:
        callback = ValidationRewardCallback(
            valid_df=valid_df,
            tickers=tickers,
            run_label=run_label,
            model_name=name,
            turbulence_limit=turbulence_limit,
            eval_freq=TRAINING_CURVE_EVAL_FREQ,
        )
    model.learn(total_timesteps=TRAIN_TIMESTEPS[name], progress_bar=False, callback=callback)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"{run_label}_{name}" if run_label else name
    model.save(MODEL_DIR / model_name)
    return model


def train_and_validate(train_df, valid_df, tickers):
    """Train PPO/A2C/DDPG and select the best validation Sharpe ratio."""
    results = {}
    threshold = turbulence_threshold(train_df)
    for name in MODEL_CLASSES:
        model = train_agent(name, train_df, tickers, turbulence_limit=threshold)
        valid_env = make_env(valid_df, tickers, turbulence_limit=threshold)
        valid_values, valid_actions = run_trading_episode(model, valid_env)
        metrics = performance_metrics(valid_values["portfolio_value"], INITIAL_CASH)
        metrics["repo_validation_sharpe"] = validation_sharpe(valid_values["portfolio_value"])
        if SAVE_VALIDATION_DETAILS:
            valid_values.to_csv(MODEL_DIR / f"{name}_validation_values.csv", index=False)
            valid_actions.to_csv(MODEL_DIR / f"{name}_validation_actions.csv", index=False)
        results[name] = {"model": model, "metrics": metrics}
        print(f"{name.upper()} validation Sharpe: {metrics['repo_validation_sharpe']:.3f}")

    selected = max(
        results,
        key=lambda key: np.nan_to_num(
            results[key]["metrics"]["repo_validation_sharpe"],
            nan=-np.inf,
        ),
    )
    print(f"Selected ensemble model: {selected.upper()}")
    return selected, results


def _split_by_date(df, start, end=None):
    out = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.copy()


def _has_enough_dates(df):
    return df["date"].nunique() >= 2


def _quarter_windows(dates, test_start):
    dates = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values().reset_index(drop=True)
    start_positions = dates.index[dates >= pd.Timestamp(test_start)]
    if len(start_positions) == 0:
        return
    trade_start_pos = int(start_positions[0])
    step = ROLLING_TRADE_DAYS

    while trade_start_pos < len(dates):
        valid_start_pos = trade_start_pos - ROLLING_VALIDATION_DAYS
        valid_end_pos = trade_start_pos - 1
        train_end_pos = valid_start_pos - 1
        trade_end_pos = min(trade_start_pos + ROLLING_TRADE_DAYS - 1, len(dates) - 1)

        if train_end_pos >= 0 and valid_start_pos >= 0:
            yield (
                dates.iloc[train_end_pos],
                dates.iloc[valid_start_pos],
                dates.iloc[valid_end_pos],
                dates.iloc[trade_start_pos],
                dates.iloc[trade_end_pos],
            )
        trade_start_pos += step


def rolling_quarterly_ensemble(features, tickers, train_start, test_start=TEST_START):
    """Run Yang-style quarterly retraining, validation selection, and trading.

    The window expands through time: each quarter trains on all available history
    before the latest 63-trading-day validation window, validates PPO/A2C/DDPG
    on that window, then trades the next 63 trading days with the best validation
    Sharpe model.
    """
    all_dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    cash = INITIAL_CASH
    holdings = np.zeros(len(tickers), dtype=np.float64)
    value_frames = []
    action_frames = []
    selection_rows = []

    for idx, (train_end, valid_start, valid_end, trade_start, trade_end) in enumerate(
        _quarter_windows(all_dates, test_start),
        start=1,
    ):
        train_df = _split_by_date(features, train_start, train_end)
        valid_df = _split_by_date(features, valid_start, valid_end)
        trade_df = _split_by_date(features, trade_start, trade_end)

        if (
            train_df.empty
            or not _has_enough_dates(valid_df)
            or not _has_enough_dates(trade_df)
        ):
            print(
                "Skipping rolling window with insufficient data: "
                f"train_end={train_end.date()}, valid={valid_start.date()}:{valid_end.date()}, "
                f"trade={trade_start.date()}:{trade_end.date()}"
            )
            continue

        recent_dates = train_df["date"].drop_duplicates().sort_values().tail(ROLLING_VALIDATION_DAYS)
        recent_df = train_df[train_df["date"].isin(recent_dates)]
        threshold = turbulence_threshold(train_df, recent_df=recent_df)
        run_label = f"q{idx:02d}_{trade_start.strftime('%Y%m%d')}_{trade_end.strftime('%Y%m%d')}"
        print(
            f"\nRolling window {idx}: train {pd.Timestamp(train_start).date()} to {train_end.date()}, "
            f"validate {valid_start.date()} to {valid_end.date()}, "
            f"trade {trade_start.date()} to {trade_end.date()}"
        )
        if threshold is not None:
            print(f"Turbulence threshold from training history: {threshold:.3f}")

        quarter_results = {}
        for name in MODEL_CLASSES:
            model = train_agent(
                name,
                train_df,
                tickers,
                run_label=run_label,
                turbulence_limit=threshold,
                valid_df=valid_df,
                log_training_curve=SAVE_TRAINING_CURVES,
            )
            valid_env = make_env(valid_df, tickers, turbulence_limit=threshold)
            valid_values, valid_actions = run_trading_episode(model, valid_env)
            metrics = performance_metrics(valid_values["portfolio_value"], INITIAL_CASH)
            metrics["repo_validation_sharpe"] = validation_sharpe(valid_values["portfolio_value"])
            if SAVE_VALIDATION_DETAILS:
                valid_values.to_csv(MODEL_DIR / f"{run_label}_{name}_validation_values.csv", index=False)
                valid_actions.to_csv(MODEL_DIR / f"{run_label}_{name}_validation_actions.csv", index=False)
            quarter_results[name] = {"model": model, "metrics": metrics}
            print(f"{name.upper()} validation Sharpe: {metrics['repo_validation_sharpe']:.3f}")

        selected = max(
            quarter_results,
            key=lambda key: np.nan_to_num(
                quarter_results[key]["metrics"]["repo_validation_sharpe"],
                nan=-np.inf,
            ),
        )
        selected_model = quarter_results[selected]["model"]
        print(f"Selected {selected.upper()} for {trade_start.date()} to {trade_end.date()}")

        trade_env = make_env(
            trade_df,
            tickers,
            initial_cash=cash,
            initial_holdings=holdings,
            turbulence_limit=threshold,
        )
        values, actions = run_trading_episode(selected_model, trade_env)
        values["selected_model"] = selected
        actions["selected_model"] = selected
        value_frames.append(values)
        action_frames.append(actions)

        cash = trade_env.cash
        holdings = trade_env.holdings.copy()

        row = {
            "quarter": idx,
            "train_start": pd.Timestamp(train_start),
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "trade_start": trade_start,
            "trade_end": trade_end,
            "selected_model": selected,
            "turbulence_threshold": threshold,
            "ending_cash": cash,
            "ending_portfolio_value": values["portfolio_value"].iloc[-1],
        }
        for name, result in quarter_results.items():
            row[f"{name}_validation_sharpe"] = result["metrics"]["repo_validation_sharpe"]
        selection_rows.append(row)

    if not value_frames:
        raise ValueError("Rolling ensemble produced no trading windows.")

    values = pd.concat(value_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    return values, actions, selections

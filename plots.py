"""Plotting helpers for coursework-ready figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR


def _wide_values(series_dict):
    frames = []
    for name, df in series_dict.items():
        tmp = df[["date", "portfolio_value"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.rename(columns={"portfolio_value": name})
        frames.append(tmp.set_index("date"))
    return pd.concat(frames, axis=1).dropna(how="all")


def plot_portfolio_values(series_dict):
    values = _wide_values(series_dict)
    ax = values.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("Portfolio Value")
    ax.set_xlabel("")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "portfolio_values.png", dpi=300)
    plt.close()


def plot_cumulative_returns(series_dict):
    values = _wide_values(series_dict)
    cumulative = values.apply(lambda s: s / s.dropna().iloc[0] - 1)
    ax = cumulative.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_returns.png", dpi=300)
    plt.close()


def plot_drawdowns(series_dict):
    values = _wide_values(series_dict)
    drawdowns = values / values.cummax() - 1
    ax = drawdowns.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("Drawdowns")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drawdowns.png", dpi=300)
    plt.close()


def plot_action_summary(actions: pd.DataFrame):
    if actions.empty or len(actions.columns) <= 1:
        return
    action_cols = [
        c for c in actions.columns
        if c not in {"date", "selected_model"} and pd.api.types.is_numeric_dtype(actions[c])
    ]
    if not action_cols:
        return
    summary = actions[action_cols].abs().sum().sort_values(ascending=False)
    ax = summary.plot(kind="bar", figsize=(10, 5))
    ax.set_title("RL Action Summary: Absolute Shares Traded")
    ax.set_xlabel("")
    ax.set_ylabel("Absolute shares")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "action_summary.png", dpi=300)
    plt.close()


def plot_rewards_over_time(values: pd.DataFrame):
    if values.empty:
        return
    rewards = values[["date", "reward"]].copy() if "reward" in values.columns else None
    if rewards is None:
        rewards = values[["date", "portfolio_value"]].copy()
        rewards["reward"] = rewards["portfolio_value"].diff().fillna(0) * 1e-4
    rewards["date"] = pd.to_datetime(rewards["date"])
    ax = rewards.set_index("date")["reward"].plot(figsize=(10, 5), linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("RL Reward Over Time")
    ax.set_xlabel("")
    ax.set_ylabel("Scaled reward")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rewards_over_time.png", dpi=300)
    plt.close()


def plot_training_reward_curves(curves: pd.DataFrame):
    if curves.empty:
        return
    metric = "validation_scaled_reward"
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in curves.groupby("model"):
        averaged = group.groupby("timesteps", as_index=True)[metric].mean().sort_index()
        ax.plot(averaged.index, averaged.values, marker="o", linewidth=1.8, label=model.upper())
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Validation Reward vs Training Timesteps")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Validation scaled reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_reward_curves.png", dpi=300)
    plt.close()


def plot_model_vs_ensemble_returns(wide_values: pd.DataFrame):
    if wide_values.empty:
        return
    cumulative = wide_values.apply(lambda s: s / s.dropna().iloc[0] - 1)
    ax = cumulative.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("Out-of-Sample Cumulative Returns: Models vs Ensemble")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_vs_ensemble_cumulative_returns.png", dpi=300)
    plt.close()


def make_all_plots(series_dict, actions):
    plot_portfolio_values(series_dict)
    plot_cumulative_returns(series_dict)
    plot_drawdowns(series_dict)
    rl_values = next(iter(series_dict.values()))
    plot_rewards_over_time(rl_values)
    plot_action_summary(actions)

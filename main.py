"""Run the coursework replication end to end."""

import pandas as pd

from benchmarks import (
    dia_buy_and_hold,
    equal_weight_buy_and_hold,
    minimum_variance_buy_and_hold,
)
from config import (
    ENABLE_TURBULENCE_CONTROL,
    INITIAL_CASH,
    DATA_DOWNLOAD_START,
    DATA_DIR,
    FULL_START,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_START,
    TICKERS,
    TRAIN_END,
    TRAIN_START,
    INDICATOR_WINDOW,
    ROLLING_TRADE_DAYS,
    ROLLING_VALIDATION_DAYS,
    TURBULENCE_LOOKBACK,
    TURBULENCE_THRESHOLD_QUANTILE,
    USE_AUTHOR_TURBULENCE_LOGIC,
)
from data import load_or_download_prices
from evaluate import performance_metrics
from indicators import add_indicators, add_turbulence
from plots import make_all_plots
from train import rolling_quarterly_ensemble


def split_by_date(df: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    out = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.copy()


def prepare_features(price_df: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = DATA_DIR / "features.csv"
    metadata_path = DATA_DIR / "features_metadata.txt"
    expected_metadata = (
        f"indicator_window={INDICATOR_WINDOW}\n"
        f"data_download_start={DATA_DOWNLOAD_START}\n"
        f"full_sample_start={FULL_START}\n"
        f"turbulence_lookback={TURBULENCE_LOOKBACK}\n"
        f"use_turbulence={ENABLE_TURBULENCE_CONTROL}\n"
    )
    if feature_path.exists():
        metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else ""
        if metadata == expected_metadata:
            print(f"Loading cached features from {feature_path}")
            features = pd.read_csv(feature_path, parse_dates=["date"])
            if "turbulence" in features.columns:
                return features
        print("Cached features use old settings; recomputing features.")
    print("Computing MACD, RSI(30), CCI(30), and ADX/DX-style trend indicator(30)...")
    stock_prices = price_df[price_df["tic"].isin(TICKERS)].copy()
    features = add_indicators(stock_prices)
    if ENABLE_TURBULENCE_CONTROL:
        print("Computing past-looking turbulence index...")
        features = add_turbulence(features, lookback=TURBULENCE_LOOKBACK)
    features = features[features["date"] >= pd.Timestamp(FULL_START)].copy()
    features.to_csv(feature_path, index=False)
    metadata_path.write_text(expected_metadata, encoding="utf-8")
    print(f"Saved features to {feature_path}")
    return features


def save_method_notes():
    note = (
        "Important limitation: this replication uses the current Dow 30 constituents as a "
        "fixed universe from 2015 onward. AMZN, NVDA, SHW, and some other names were not "
        "Dow constituents throughout the whole period. This follows the fixed-universe "
        "spirit of Yang et al. (2020), but it can introduce index-membership and "
        "survivorship-style bias.\n\n"
        "Rolling ensemble: PPO, A2C, and DDPG are retrained every quarter using an "
        f"expanding historical window. The latest {ROLLING_VALIDATION_DAYS} trading days "
        f"are used as validation, the best validation Sharpe model trades the next "
        f"{ROLLING_TRADE_DAYS} trading days, and the next window moves forward. Newly "
        "observed trading data enters later expanding windows, which implements "
        "continued training through the trading stage without using future data inside "
        "a trading window.\n\n"
        "Turbulence risk control: when enabled, the threshold follows the author repo's "
        f"adaptive rule using the {TURBULENCE_THRESHOLD_QUANTILE:.0%} in-sample quantile "
        "or the maximum in-sample turbulence depending on recent turbulence. If daily "
        "turbulence exceeds the threshold, the environment sells current holdings and "
        "blocks buys for that step.\n\n"
        f"Author-style turbulence logic enabled: {USE_AUTHOR_TURBULENCE_LOGIC}.\n"
    )
    (OUTPUT_DIR / "method_notes.txt").write_text(note, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_or_download_prices(force=False)
    features = prepare_features(price_df)

    stock_features = features[features["tic"].isin(TICKERS)].copy()
    initial_train_df = split_by_date(stock_features, TRAIN_START, TRAIN_END)
    test_df = split_by_date(stock_features, TEST_START, None)

    if initial_train_df.empty or test_df.empty:
        raise ValueError("One of the train/test splits is empty after cleaning.")

    print("Running rolling quarterly ensemble on the out-of-sample test period...")
    rl_values, rl_actions, rolling_summary = rolling_quarterly_ensemble(
        stock_features,
        TICKERS,
        train_start=TRAIN_START,
        test_start=TEST_START,
    )
    rl_values.to_csv(OUTPUT_DIR / "rl_ensemble_values.csv", index=False)
    rl_actions.to_csv(OUTPUT_DIR / "rl_ensemble_actions.csv", index=False)
    rolling_summary.to_csv(OUTPUT_DIR / "rolling_selection_summary.csv", index=False)

    test_prices = split_by_date(price_df, TEST_START, None)
    train_prices = split_by_date(price_df, TRAIN_START, TRAIN_END)
    dia_values = dia_buy_and_hold(test_prices)
    equal_values = equal_weight_buy_and_hold(test_prices, TICKERS)
    minvar_values, minvar_weights = minimum_variance_buy_and_hold(train_prices, test_prices, TICKERS)
    minvar_weights.to_csv(OUTPUT_DIR / "minimum_variance_weights.csv")

    series = {
        "RL rolling ensemble": rl_values,
        "DIA buy-and-hold": dia_values,
        "Equal-weight 30": equal_values,
        "Minimum-variance": minvar_values,
    }

    rows = []
    for name, values in series.items():
        metrics = performance_metrics(values["portfolio_value"], INITIAL_CASH)
        metrics["strategy"] = name
        rows.append(metrics)
    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(OUTPUT_DIR / "performance_summary.csv")

    make_all_plots(series, rl_actions)
    save_method_notes()

    print("\nPerformance summary:")
    print(summary.round(4))
    print(f"\nOutputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

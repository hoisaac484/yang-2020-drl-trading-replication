# Yang et al. (2020) DRL Trading Replication

Python coursework replication of Yang et al. (2020), **Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy**. The project keeps the paper's multi-stock reinforcement-learning trading environment and rolling ensemble idea, but moves the sample forward to recent data using `yfinance`.

The final coursework report is included at:

`reports/YingShekIsaacHo_IFTE0004_Personal_Coursework.pdf`

## Research Aim

The original paper frames stock trading as a sequential portfolio-control problem. Instead of forecasting returns first and then applying a separate allocation rule, PPO, A2C and DDPG agents observe the market state and choose trades directly. The ensemble selects the model with the strongest validation Sharpe ratio for the next trading period.

This replication asks whether the same general design remains effective on a more recent Dow 30 sample, and whether a small nested walk-forward tuning extension improves robustness.

## Data and Universe

Data are downloaded with `yfinance` using daily OHLCV prices. DIA is used as the Dow benchmark proxy.

The stock universe is fixed to the current Dow 30 list supplied in the coursework prompt:

`AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GS, HD, HON, IBM, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, NVDA, PG, SHW, TRV, UNH, V, VZ, WMT`

Date design:

- Data download starts: `2014-01-01`, only to warm up rolling indicators
- Main sample starts: `2015-01-01`
- Initial training period: `2015-01-01` to `2021-12-31`
- Out-of-sample trading period: `2023-01-01` to latest available data
- Rolling validation/trading windows: 63 trading days for the author-style experiment

The code runs in strict missing-data mode. If any OHLCV value is missing for the 30 stocks or DIA after alignment, the run stops instead of forward-filling. This keeps data-quality issues visible.

## Trading Environment

For 30 stocks, the state dimension is `1 + 30 * 6 = 181`:

- Cash balance
- Adjusted close prices
- Current holdings
- MACD
- RSI
- CCI
- ADX/DX-style trend strength

Actions are continuous values in `[-1, 1]` for each stock. Positive actions buy, negative actions sell, and near-zero actions hold. Actions are scaled by `HMAX = 100` shares. The environment enforces:

- Initial cash: `$1,000,000`
- No short selling
- No negative cash balance
- Transaction cost: `0.1%` per buy or sell trade
- Reward: change in total portfolio value, net of transaction costs, scaled for RL stability

Turbulence is used as an external risk-control rule. When turbulence is above the adaptive threshold, the environment liquidates current holdings and blocks new buys for that step. Turbulence is not added to the 181-dimensional state vector.

## Experiments

### Experiment 1: Author-Style Replication

`main.py` runs the author-style rolling ensemble:

1. Train PPO, A2C and DDPG on an expanding historical window.
2. Validate the three agents on the latest 63 trading days.
3. Select the agent with the highest validation Sharpe ratio.
4. Trade the next 63 trading days.
5. Carry cash and holdings into the next trading window.
6. Repeat until the end of the test sample.

Training lengths follow the author repository scale:

- A2C: `30,000` timesteps
- PPO: `100,000` timesteps
- DDPG: `10,000` timesteps

### Experiment 2: Nested Walk-Forward Tuning Extension

`run_nested_tuning.py` implements a small robustness extension. Before each trading window, candidate settings are evaluated only on prior validation folds, across five seeds. The scoring rule is:

```text
mean validation Sharpe - 0.5 * standard deviation of validation Sharpe
```

This tests whether a more robust nested selection rule improves the author-style setup without using future test-period information. The best reported nested configuration among the tested settings uses:

- 126-day validation window
- 5 validation seeds
- `max_tuning_folds = 3`

This is not a comprehensive hyperparameter search; it is a coursework-scale robustness extension.

## Benchmarks

The RL strategies are compared with:

- DIA buy-and-hold
- Equal-weight buy-and-hold portfolio of the same 30 stocks
- Long-only minimum-variance buy-and-hold portfolio, with covariance estimated using training-period returns only

## Main Results

Out-of-sample period: 2023 onward. Risk-free rate is set to zero.

| Strategy | Final value | Cumulative return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Author-style RL ensemble | `$1.417M` | `41.7%` | `0.739` | `-21.3%` |
| Nested tuned RL, 126 days, 5 seeds, folds=3 | `$1.865M` | `86.5%` | `1.254` | `-15.4%` |
| DIA buy-and-hold | `$1.584M` | `58.4%` | `1.092` | `-16.0%` |
| Equal-weight Dow 30 | `$2.169M` | `116.9%` | `1.679` | `-19.1%` |
| Minimum-variance | `$1.710M` | `71.0%` | `1.478` | `-9.9%` |

The nested tuned extension improves on the author-style RL replication and beats DIA in this sample, but it does not dominate the equal-weight or minimum-variance benchmarks. This is an important conclusion of the coursework: the DRL framework is implementable and extensible, but simple transparent portfolios remain very competitive.

Key output files:

- `outputs/author_style/performance_summary.csv`
- `outputs/nested_tuned/performance_summary.csv`
- `outputs/analysis/five_strategy_cumulative_returns_over_time.png`
- `outputs/analysis/bootstrap_metric_ci.csv`
- `outputs/analysis/paired_return_tests.csv`

## Project Structure

```text
main.py                         # Experiment 1 author-style rolling ensemble
run_nested_tuning.py            # Experiment 2 nested walk-forward tuning
run_multi_seed_robustness.py    # Multi-seed robustness check
config.py                       # Tickers, dates, costs, timesteps and model settings
data.py                         # yfinance download and cleaning
indicators.py                   # MACD, RSI, CCI, ADX/DX and turbulence
env.py                          # Multi-stock trading environment
train.py                        # PPO/A2C/DDPG training and rolling ensemble logic
evaluate.py                     # Performance metrics
benchmarks.py                   # DIA, equal-weight and minimum-variance benchmarks
plots.py                        # Standard result plots
outputs/author_style/           # Final Experiment 1 outputs
outputs/nested_tuned/           # Final Experiment 2 outputs
outputs/analysis/               # Report comparison and uncertainty tables/plots
outputs/robustness/             # Seed-sensitivity summaries
reports/                        # Final coursework report PDF
```

Large reproducible artifacts are intentionally excluded from Git: virtual environments, raw cached data/features, trained model zip files, and exploratory backup runs.

## How to Run

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the author-style replication:

```bash
python main.py
```

Run the nested tuning extension:

```bash
python run_nested_tuning.py --grid-preset small --validation-windows 126 --seeds 1 2 3 4 5 --max-tuning-folds 3
```

For a short smoke test:

```bash
python run_nested_tuning.py --max-windows 1 --seeds 1
```

## Limitations

- `yfinance` differs from institutional data sources used in academic studies.
- The current Dow 30 is fixed backward to 2015. AMZN, NVDA and SHW were not Dow members throughout the whole period, so this introduces index-membership bias and weakens direct comparability with the original paper.
- Transaction costs are simplified to a fixed 0.1% per trade.
- Market impact, bid-ask spread and liquidity limits are not modelled.
- The nested tuning grid is deliberately small because full rolling multi-seed tuning is computationally expensive.
- RL training is seed-sensitive; fixed-seed results should be interpreted as reproducible realizations, not guaranteed expected performance.
- Bootstrap and paired-return tests in the report suggest the nested RL improvement is economically interesting but statistically uncertain.

## Reference

Yang, H., Liu, X.-Y., Zhong, S. and Walid, A. (2020). *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy*. ICAIF 2020.

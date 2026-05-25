"""Gymnasium multi-stock trading environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        tickers,
        initial_cash=1_000_000.0,
        hmax=100,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        turbulence_threshold=None,
        initial_holdings=None,
    ):
        super().__init__()
        self.df = df.copy().sort_values(["date", "tic"])
        self.tickers = list(tickers)
        self.initial_cash = float(initial_cash)
        self.hmax = int(hmax)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_scaling = float(reward_scaling)
        self.turbulence_threshold = turbulence_threshold
        self.initial_holdings = (
            np.asarray(initial_holdings, dtype=np.float64)
            if initial_holdings is not None
            else np.zeros(len(self.tickers), dtype=np.float64)
        )

        self.dates = np.array(sorted(self.df["date"].unique()))
        self.n_stock = len(self.tickers)
        self.price_array = self._pivot("adj_close")
        self.tech_arrays = [self._pivot(col) for col in ["macd", "rsi", "cci", "adx"]]
        self.turbulence_array = self._date_values("turbulence")

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stock,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + self.n_stock * 6,), dtype=np.float32
        )
        self.reset()

    def _pivot(self, column):
        pivot = self.df.pivot(index="date", columns="tic", values=column).reindex(
            index=self.dates, columns=self.tickers
        )
        return pivot.astype(float).to_numpy()

    def _date_values(self, column):
        if column not in self.df.columns:
            return np.zeros(len(self.dates), dtype=np.float64)
        values = self.df.groupby("date")[column].first().reindex(self.dates).fillna(0.0)
        return values.astype(float).to_numpy()

    def _portfolio_value(self, prices):
        return float(self.cash + np.dot(self.holdings, prices))

    def _get_obs(self):
        obs = [np.array([self.cash]), self.price_array[self.day], self.holdings]
        obs.extend(arr[self.day] for arr in self.tech_arrays)
        return np.concatenate(obs).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_cash
        self.holdings = self.initial_holdings.copy()
        self.total_cost = 0.0
        self.trades = 0
        self.last_trade_shares = np.zeros(self.n_stock, dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        prices = self.price_array[self.day]
        previous_value = self._portfolio_value(prices)
        trade_shares = np.zeros(self.n_stock, dtype=np.float64)
        turbulence = float(self.turbulence_array[self.day])
        risk_off = (
            self.turbulence_threshold is not None
            and turbulence > float(self.turbulence_threshold)
        )

        if risk_off:
            action = -np.ones(self.n_stock, dtype=np.float64)

        sell_index = np.where(action < 0)[0]
        for idx in sell_index:
            shares = min(abs(action[idx]) * self.hmax, self.holdings[idx])
            if shares > 0:
                proceeds = shares * prices[idx]
                cost = proceeds * self.transaction_cost_pct
                self.cash += proceeds - cost
                self.holdings[idx] -= shares
                self.total_cost += cost
                self.trades += 1
                trade_shares[idx] = -shares

        if not risk_off:
            buy_index = np.where(action > 0)[0]
            for idx in buy_index[np.argsort(action[buy_index])[::-1]]:
                shares = action[idx] * self.hmax
                gross = shares * prices[idx]
                total = gross * (1 + self.transaction_cost_pct)
                if total > self.cash:
                    shares = self.cash / (prices[idx] * (1 + self.transaction_cost_pct))
                    gross = shares * prices[idx]
                    total = gross * (1 + self.transaction_cost_pct)
                if shares > 0 and total <= self.cash:
                    self.cash -= total
                    self.holdings[idx] += shares
                    self.total_cost += gross * self.transaction_cost_pct
                    self.trades += 1
                    trade_shares[idx] = shares

        self.last_trade_shares = trade_shares
        self.day += 1
        terminated = self.day >= len(self.dates) - 1
        new_value = self._portfolio_value(self.price_array[self.day])
        reward = (new_value - previous_value) * self.reward_scaling
        info = {
            "date": self.dates[self.day],
            "portfolio_value": new_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "transaction_costs": self.total_cost,
            "trades": self.trades,
            "trade_shares": trade_shares.copy(),
            "turbulence": turbulence,
            "risk_off": risk_off,
        }
        return self._get_obs(), float(reward), terminated, False, info

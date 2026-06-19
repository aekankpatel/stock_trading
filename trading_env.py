"""
Custom Gymnasium trading environment.

State  : market features + portfolio state (normalised)
Actions: 0=Hold, 1=Buy, 2=Sell
Reward : daily portfolio return minus transaction cost
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_loader import FEATURE_COLS

TRANSACTION_COST = 0.001  # 0.1% per trade


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance: float = 10_000.0, render_mode=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        # Market features + 3 portfolio state values: cash_ratio, shares_ratio, portfolio_return
        n_features = len(FEATURE_COLS) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        # 0=Hold, 1=Buy all-in, 2=Sell all
        self.action_space = spaces.Discrete(3)

        self._max_steps = len(self.df) - 1

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.prev_portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.trade_log = []
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        price = float(row["close"])
        portfolio_value = self.cash + self.shares * price

        market_feats = np.array(
            [float(row[c]) for c in FEATURE_COLS], dtype=np.float32
        )

        # Portfolio state (normalised so scale is comparable to market feats)
        cash_ratio = self.cash / self.initial_balance
        shares_value_ratio = (self.shares * price) / self.initial_balance
        portfolio_return = (portfolio_value / self.initial_balance) - 1.0

        portfolio_feats = np.array(
            [cash_ratio, shares_value_ratio, portfolio_return], dtype=np.float32
        )

        return np.concatenate([market_feats, portfolio_feats])

    # ------------------------------------------------------------------
    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = float(row["close"])

        prev_value = self.cash + self.shares * price
        cost = 0.0

        if action == 1:  # Buy: spend all cash
            if self.cash > price:
                shares_to_buy = self.cash / (price * (1 + TRANSACTION_COST))
                cost = shares_to_buy * price * TRANSACTION_COST
                self.shares += shares_to_buy
                self.cash -= shares_to_buy * price + cost
                self.trade_log.append(("BUY", self.current_step, price))

        elif action == 2:  # Sell: liquidate all shares
            if self.shares > 0:
                proceeds = self.shares * price
                cost = proceeds * TRANSACTION_COST
                self.cash += proceeds - cost
                self.trade_log.append(("SELL", self.current_step, price))
                self.shares = 0.0

        self.current_step += 1
        new_price = float(self.df.iloc[self.current_step]["close"])
        new_value = self.cash + self.shares * new_price

        # Reward: excess return over the market.
        # portfolio_return - market_return means:
        #   - holding cash when market rises  → negative reward (opportunity cost)
        #   - holding cash when market falls  → positive reward (capital protection)
        #   - being invested when market rises → near-zero reward (matched market)
        # This gives a strong directional signal instead of always rewarding "do nothing".
        market_return = (new_price - price) / price
        portfolio_return = (new_value - prev_value) / prev_value
        reward = portfolio_return - market_return

        self.portfolio_history.append(new_value)
        self.prev_portfolio_value = new_value

        terminated = self.current_step >= self._max_steps
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "portfolio_value": new_value,
            "cash": self.cash,
            "shares": self.shares,
            "price": new_price,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        row = self.df.iloc[self.current_step]
        price = float(row["close"])
        portfolio_value = self.cash + self.shares * price
        print(
            f"Step {self.current_step:4d} | Price: ${price:8.2f} | "
            f"Cash: ${self.cash:10.2f} | Shares: {self.shares:8.4f} | "
            f"Portfolio: ${portfolio_value:10.2f}"
        )

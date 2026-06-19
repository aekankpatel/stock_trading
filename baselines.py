"""
Three simple baselines: Buy-and-Hold, Random, Moving-Average crossover.
"""

import numpy as np
import pandas as pd

TRANSACTION_COST = 0.001


def run_buy_and_hold(df: pd.DataFrame, initial_balance: float = 10_000.0) -> dict:
    price_series = df["close"].values
    # Buy on day 0, hold until end
    shares = initial_balance / (price_series[0] * (1 + TRANSACTION_COST))
    cash = initial_balance - shares * price_series[0] * (1 + TRANSACTION_COST)
    portfolio = [cash + shares * p for p in price_series]
    trade_log = [("BUY", 0, price_series[0])]
    return {"portfolio_history": portfolio, "trade_log": trade_log}


def run_random(df: pd.DataFrame, initial_balance: float = 10_000.0, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    price_series = df["close"].values
    cash = initial_balance
    shares = 0.0
    portfolio = [initial_balance]
    trade_log = []

    for i, price in enumerate(price_series[:-1]):
        action = rng.integers(0, 3)  # 0=hold, 1=buy, 2=sell
        cost = 0.0
        if action == 1 and cash > price:
            shares_to_buy = cash / (price * (1 + TRANSACTION_COST))
            cost = shares_to_buy * price * TRANSACTION_COST
            shares += shares_to_buy
            cash -= shares_to_buy * price + cost
            trade_log.append(("BUY", i, price))
        elif action == 2 and shares > 0:
            proceeds = shares * price
            cost = proceeds * TRANSACTION_COST
            cash += proceeds - cost
            trade_log.append(("SELL", i, price))
            shares = 0.0
        next_price = price_series[i + 1]
        portfolio.append(cash + shares * next_price)

    return {"portfolio_history": portfolio, "trade_log": trade_log}


def run_moving_average(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    short_window: int = 5,
    long_window: int = 20,
) -> dict:
    """
    Classic MA crossover: buy when short MA crosses above long MA, sell when it crosses below.
    """
    price_series = df["close"].values
    ma_short = pd.Series(price_series).rolling(short_window).mean().values
    ma_long = pd.Series(price_series).rolling(long_window).mean().values

    cash = initial_balance
    shares = 0.0
    portfolio = [initial_balance]
    trade_log = []

    for i in range(len(price_series) - 1):
        price = price_series[i]
        cost = 0.0

        # Signal only valid after enough data
        if np.isnan(ma_short[i]) or np.isnan(ma_long[i]):
            portfolio.append(cash + shares * price)
            continue

        if i > 0 and not np.isnan(ma_short[i - 1]) and not np.isnan(ma_long[i - 1]):
            cross_up = (ma_short[i - 1] < ma_long[i - 1]) and (ma_short[i] >= ma_long[i])
            cross_down = (ma_short[i - 1] > ma_long[i - 1]) and (ma_short[i] <= ma_long[i])

            if cross_up and cash > price:
                shares_to_buy = cash / (price * (1 + TRANSACTION_COST))
                cost = shares_to_buy * price * TRANSACTION_COST
                shares += shares_to_buy
                cash -= shares_to_buy * price + cost
                trade_log.append(("BUY", i, price))

            elif cross_down and shares > 0:
                proceeds = shares * price
                cost = proceeds * TRANSACTION_COST
                cash += proceeds - cost
                trade_log.append(("SELL", i, price))
                shares = 0.0

        next_price = price_series[i + 1]
        portfolio.append(cash + shares * next_price)

    return {"portfolio_history": portfolio, "trade_log": trade_log}

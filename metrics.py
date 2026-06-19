"""
Portfolio performance metrics.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.15 = -15%)."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return float(np.min(drawdown))


def total_return(portfolio_values: np.ndarray) -> float:
    return float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0])


def win_rate(trade_log: list) -> float:
    """Fraction of round-trip trades that were profitable."""
    buys, sells = [], []
    for action, step, price in trade_log:
        if action == "BUY":
            buys.append(price)
        elif action == "SELL":
            sells.append(price)
    pairs = list(zip(buys, sells))
    if not pairs:
        return 0.0
    wins = sum(1 for b, s in pairs if s > b)
    return wins / len(pairs)


def compute_all(portfolio_values: list, trade_log: list, initial_balance: float) -> dict:
    arr = np.array(portfolio_values, dtype=float)
    daily_rets = np.diff(arr) / arr[:-1]
    return {
        "final_value": arr[-1],
        "total_return_pct": total_return(arr) * 100,
        "sharpe_ratio": sharpe_ratio(daily_rets),
        "max_drawdown_pct": max_drawdown(arr) * 100,
        "num_trades": len(trade_log),
        "win_rate_pct": win_rate(trade_log) * 100,
    }

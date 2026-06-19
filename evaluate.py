"""
Backtest trained DQN and PPO agents on the held-out test set.
Compares against Buy-and-Hold, Random, and Moving-Average baselines.

Usage:
    python evaluate.py
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from data_loader import load_data
from trading_env import StockTradingEnv
from baselines import run_buy_and_hold, run_random, run_moving_average
from metrics import compute_all

MODELS_DIR = "models"
INITIAL_BALANCE = 10_000.0


def run_agent(model, df) -> dict:
    env = StockTradingEnv(df, initial_balance=INITIAL_BALANCE)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return {
        "portfolio_history": env.portfolio_history,
        "trade_log": env.trade_log,
    }


def load_model(algo: str, model_cls):
    """
    Always load the final model (end of training).
    The 'best' checkpoint is selected by a single noisy eval episode and
    frequently picks a degenerate hold-all policy on financial data.
    """
    final_path = os.path.join(MODELS_DIR, f"{algo}_final.zip")
    best_path  = os.path.join(MODELS_DIR, f"{algo}_best", "best_model.zip")
    if os.path.exists(final_path):
        print(f"  Loading {algo.upper()} final model from {final_path}")
        return model_cls.load(final_path)
    elif os.path.exists(best_path):
        print(f"  Loading {algo.upper()} best model from {best_path}")
        return model_cls.load(best_path)
    else:
        raise FileNotFoundError(f"No saved model found for {algo}. Run train.py first.")


def evaluate(test_df: pd.DataFrame) -> pd.DataFrame:
    results = {}

    print("\nRunning baselines...")
    results["Buy & Hold"] = run_buy_and_hold(test_df, INITIAL_BALANCE)
    results["Random"] = run_random(test_df, INITIAL_BALANCE, seed=42)
    results["MA Crossover"] = run_moving_average(test_df, INITIAL_BALANCE)

    print("Running RL agents...")
    dqn = load_model("dqn", DQN)
    results["DQN"] = run_agent(dqn, test_df)

    ppo = load_model("ppo", PPO)
    results["PPO"] = run_agent(ppo, test_df)

    # Compute metrics for each strategy
    rows = []
    for name, data in results.items():
        m = compute_all(
            data["portfolio_history"],
            data["trade_log"],
            INITIAL_BALANCE,
        )
        m["Strategy"] = name
        rows.append(m)

    metrics_df = pd.DataFrame(rows).set_index("Strategy")
    metrics_df = metrics_df[[
        "final_value",
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "num_trades",
        "win_rate_pct",
    ]]
    metrics_df.columns = [
        "Final Value ($)",
        "Total Return (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "# Trades",
        "Win Rate (%)",
    ]

    # Save raw portfolio histories for plotting
    os.makedirs("results", exist_ok=True)
    for name, data in results.items():
        pd.Series(data["portfolio_history"]).to_csv(
            f"results/{name.lower().replace(' ', '_').replace('&', 'and')}_portfolio.csv",
            index=False,
        )

    return metrics_df, results


if __name__ == "__main__":
    _, test_df = load_data(ticker="SPY", start="2015-01-01", end="2024-12-31")

    metrics_df, results = evaluate(test_df)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.2f}"))
    metrics_df.to_csv("results/metrics_summary.csv")
    print("\nMetrics saved to results/metrics_summary.csv")
    print("Run plot_results.py to generate figures.")

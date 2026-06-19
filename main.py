"""
Single entry point: data download → train → evaluate → plot.

Usage:
    python main.py
    python main.py --skip-train          # re-use saved models
    python main.py --timesteps 500000    # longer training
"""

import argparse
import os

from data_loader import load_data
from train import train_dqn, train_ppo
from evaluate import evaluate
from plot_results import (
    plot_equity_curves,
    plot_drawdown_curves,
    plot_bar_comparison,
    plot_training_curves,
    plot_action_distribution,
)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training, load saved models")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps per agent")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_df, test_df = load_data(ticker=args.ticker, start=args.start, end=args.end)

    val_split = int(len(train_df) * 0.9)
    val_df = train_df.iloc[val_split:].reset_index(drop=True)
    pure_train_df = train_df.iloc[:val_split].reset_index(drop=True)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    if not args.skip_train:
        import train as train_module
        train_module.TRAIN_TIMESTEPS = args.timesteps
        train_dqn(pure_train_df, val_df)
        train_ppo(pure_train_df, val_df)
    else:
        print("Skipping training — loading saved models.")

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    metrics_df, results = evaluate(test_df)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.2f}"))

    # ── 4. Plot ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_equity_curves(results, test_df)
    plot_drawdown_curves(results, test_df)
    plot_bar_comparison(metrics_df)
    plot_training_curves()
    plot_action_distribution(results)

    print(f"\nDone. Figures are in ./{FIGURES_DIR}/")


if __name__ == "__main__":
    main()

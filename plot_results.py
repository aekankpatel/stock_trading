"""
Generate all figures for the report:
  1. Equity curves
  2. Drawdown curves
  3. Bar charts: Total Return, Sharpe, Max Drawdown
  4. Training reward curves (smoothed)
  5. Action distribution (stacked horizontal bar)

Usage:
    python plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns

from data_loader import load_data
from evaluate import evaluate

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "Buy & Hold":  "#2196F3",
    "Random":      "#9E9E9E",
    "MA Crossover":"#FF9800",
    "DQN":         "#F44336",
    "PPO":         "#4CAF50",
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def _smooth(y, window=7):
    """Simple rolling-mean smoother; returns same length as input."""
    s = pd.Series(y).rolling(window, min_periods=1, center=True).mean()
    return s.values


# ---------------------------------------------------------------------------
# 1. Equity Curves
# ---------------------------------------------------------------------------
def plot_equity_curves(results: dict, test_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))
    dates = pd.to_datetime(test_df["Date"].values)

    for name, data in results.items():
        hist = np.array(data["portfolio_history"])
        n = min(len(hist), len(dates))
        lw = 2.5 if name in ("DQN", "PPO") else 1.8
        ax.plot(dates[:n], hist[:n], label=name, color=COLORS[name],
                linewidth=lw, zorder=3 if name in ("DQN", "PPO") else 2)

    ax.axhline(10_000, color="#555555", linestyle="--", linewidth=1,
               alpha=0.6, label="Initial $10,000", zorder=1)

    ax.set_title("Portfolio Value — Test Period (2023–2024)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Portfolio Value ($)", labelpad=8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: pd.Timestamp(x, unit="D").strftime("%Y-%m") if x > 0 else ""))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.margins(x=0.01)
    plt.tight_layout()
    _save(fig, "equity_curves.png")


# ---------------------------------------------------------------------------
# 2. Drawdown Curves
# ---------------------------------------------------------------------------
def _drawdown(values: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(values)
    return (values - peak) / peak * 100


def plot_drawdown_curves(results: dict, test_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    dates = pd.to_datetime(test_df["Date"].values)

    for name, data in results.items():
        hist = np.array(data["portfolio_history"])
        dd   = _drawdown(hist)
        n    = min(len(dd), len(dates))
        lw   = 2.5 if name in ("DQN", "PPO") else 1.8
        ax.plot(dates[:n], dd[:n], label=name, color=COLORS[name],
                linewidth=lw, zorder=3 if name in ("DQN", "PPO") else 2)

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.5)
    ax.fill_between(dates[:n], 0, np.full(n, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -20),
                    alpha=0.03, color="red")

    ax.set_title("Portfolio Drawdown — Test Period (2023–2024)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Drawdown from Peak (%)", labelpad=8)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10)
    ax.margins(x=0.01)
    # Keep y-axis from expanding to -100 just because of fill_between
    worst = min(
        _drawdown(np.array(d["portfolio_history"])).min()
        for d in results.values()
    )
    ax.set_ylim(worst * 1.4, 1)
    plt.tight_layout()
    _save(fig, "drawdown_curves.png")


# ---------------------------------------------------------------------------
# 3. Bar chart comparison (3 panels)
# ---------------------------------------------------------------------------
def plot_bar_comparison(metrics_df: pd.DataFrame):
    strategies = metrics_df.index.tolist()
    bar_colors = [COLORS[s] for s in strategies]
    x          = np.arange(len(strategies))
    width      = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Total Return (%)", "Total Return (%)"),
        ("Sharpe Ratio",     "Sharpe Ratio"),
        ("Max Drawdown (%)", "Max Drawdown (%)"),
    ]

    for ax, (col, title) in zip(axes, panels):
        vals = metrics_df[col].values
        bars = ax.bar(x, vals, width=width, color=bar_colors,
                      edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=10)
        ax.axhline(0, color="#444444", linewidth=0.7, alpha=0.5, zorder=2)

        # Compute axis range first so offset is proportional
        lo, hi = ax.get_ylim()
        span   = hi - lo
        pad    = span * 0.15

        for bar, v in zip(bars, vals):
            if v >= 0:
                # Positive bar: label just above the top edge
                y_pos = v + span * 0.012
                va    = "bottom"
            else:
                # Negative bar: label just below the bottom edge
                y_pos = v - span * 0.012
                va    = "top"

            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_pos, f"{v:.1f}",
                    ha="center", va=va, fontsize=9, fontweight="bold",
                    color="#222222")

        # Expand limits so labels are never clipped
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("Strategy Comparison — Test Period (2023–2024)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "bar_comparison.png")


# ---------------------------------------------------------------------------
# 4. Training curves (raw + smoothed trend)
# ---------------------------------------------------------------------------
def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (algo, color) in zip(axes, [("dqn", COLORS["DQN"]), ("ppo", COLORS["PPO"])]):
        path = f"logs/{algo}_eval/evaluations.npz"
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "No eval data found\n(run train.py first)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{algo.upper()} — Eval Reward", fontweight="bold")
            continue

        data       = np.load(path)
        timesteps  = data["timesteps"]
        raw        = data["results"].mean(axis=1)
        smoothed   = _smooth(raw, window=9)

        # Raw (faint) + smoothed trend (bold)
        ax.plot(timesteps, raw,      color=color, linewidth=1,   alpha=0.35, label="Raw")
        ax.plot(timesteps, smoothed, color=color, linewidth=2.5, alpha=0.95, label="Smoothed (window=9)")
        ax.fill_between(timesteps,
                        data["results"].min(axis=1),
                        data["results"].max(axis=1),
                        alpha=0.12, color=color)
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_title(f"{algo.upper()} — Evaluation Reward During Training",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Timesteps", labelpad=6)
        ax.set_ylabel("Episode Reward", labelpad=6)
        ax.legend(fontsize=9, loc="lower right")
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    plt.tight_layout()
    _save(fig, "training_curves.png")


# ---------------------------------------------------------------------------
# 5. Action distribution — stacked horizontal bar (replaces pie charts)
# ---------------------------------------------------------------------------
def plot_action_distribution(results: dict):
    rl_names = [n for n in ("DQN", "PPO") if n in results]

    # Compute percentages
    rows = {}
    for name in rl_names:
        log   = results[name]["trade_log"]
        total = len(results[name]["portfolio_history"]) - 1
        buys  = sum(1 for a, *_ in log if a == "BUY")
        sells = sum(1 for a, *_ in log if a == "SELL")
        holds = total - buys - sells
        rows[name] = {
            "Hold":  holds / total * 100,
            "Buy":   buys  / total * 100,
            "Sell":  sells / total * 100,
        }

    fig, ax = plt.subplots(figsize=(9, 3.2))

    action_colors = {"Hold": "#78909C", "Buy": "#4CAF50", "Sell": "#F44336"}
    actions       = ["Hold", "Buy", "Sell"]
    y_pos         = np.arange(len(rl_names))
    bar_h         = 0.45
    lefts         = np.zeros(len(rl_names))

    for action in actions:
        vals = np.array([rows[n][action] for n in rl_names])
        bars = ax.barh(y_pos, vals, left=lefts, height=bar_h,
                       color=action_colors[action], label=action, edgecolor="white")

        # Label each segment if wide enough to fit text
        for bar, v, left in zip(bars, vals, lefts):
            if v >= 4:   # only label if segment is at least 4%
                ax.text(left + v / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rl_names, fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage of Trading Days (%)", labelpad=8)
    ax.set_xlim(0, 100)
    ax.set_title("Action Distribution During Test Period", fontsize=13, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Annotate raw trade counts on the right margin
    for i, name in enumerate(rl_names):
        log    = results[name]["trade_log"]
        buys   = sum(1 for a, *_ in log if a == "BUY")
        sells  = sum(1 for a, *_ in log if a == "SELL")
        ax.text(101, i, f"  {buys}B / {sells}S",
                va="center", fontsize=9, color="#333333")

    legend_patches = [mpatches.Patch(color=action_colors[a], label=a) for a in actions]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.9, fontsize=10, ncol=3)

    plt.tight_layout()
    _save(fig, "action_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _, test_df = load_data(ticker="SPY", start="2015-01-01", end="2024-12-31")

    metrics_df, results = evaluate(test_df)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nGenerating figures...")
    plot_equity_curves(results, test_df)
    plot_drawdown_curves(results, test_df)
    plot_bar_comparison(metrics_df)
    plot_training_curves()
    plot_action_distribution(results)

    print(f"\nAll figures saved to ./{FIGURES_DIR}/")

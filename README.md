# Reinforcement Learning for Stock Trading

This project implements a reinforcement learning-based stock trading system using **DQN** and **PPO** agents. The agents are trained on historical SPY market data and evaluated against traditional baseline strategies such as Buy & Hold, Random Trading, and Moving Average Crossover.

The goal of the project is to test whether reinforcement learning agents can learn profitable trading behavior from technical indicators and portfolio-state information. The project also highlights an important practical limitation: the model selected as the “best” during validation does not always produce the best trading behavior on the final test set, especially in noisy financial environments.

---

## Project Overview

The system follows a complete trading experimentation pipeline:

1. Download historical stock data using `yfinance`
2. Engineer technical indicators such as moving averages, RSI, MACD, volatility, and volume change
3. Train DQN and PPO agents in a custom Gymnasium trading environment
4. Evaluate RL agents against baseline strategies
5. Compute portfolio performance metrics
6. Generate visualizations for equity curves, drawdowns, returns, Sharpe ratio, and action distributions

---

## Models and Strategies Compared

The project compares the following strategies:

| Strategy | Description |
|---|---|
| Buy & Hold | Buys SPY at the beginning of the test period and holds until the end |
| Random | Randomly chooses between hold, buy, and sell actions |
| Moving Average Crossover | Buys/sells based on short-term and long-term moving average crossover signals |
| DQN | Deep Q-Network reinforcement learning agent |
| PPO | Proximal Policy Optimization reinforcement learning agent |

---

## Trading Environment

The project uses a custom Gymnasium environment called `StockTradingEnv`.

### Action Space

The agent can choose one of three discrete actions:

| Action | Meaning |
|---|---|
| 0 | Hold |
| 1 | Buy all-in |
| 2 | Sell all holdings |

### Observation Space

Each observation contains market features and portfolio-state features.

Market features include:

- Daily return
- 5-day / 20-day moving average ratio
- 20-day volatility
- RSI
- MACD histogram
- Volume change

Portfolio-state features include:

- Cash ratio
- Share value ratio
- Portfolio return

### Reward Function

The reward is based on the agent’s portfolio return relative to the market return. This helps discourage the agent from simply doing nothing and gives stronger feedback when the agent protects capital during market declines or participates during upward movement.

Transaction costs are included at **0.1% per trade**.

---

## Project Structure

```text
fe529/
│
├── main.py              # Single entry point: data loading, training, evaluation, and plotting
├── data_loader.py       # Downloads SPY data and creates technical indicators
├── trading_env.py       # Custom Gymnasium stock trading environment
├── train.py             # Trains DQN and PPO agents
├── evaluate.py          # Backtests trained agents against baselines
├── baselines.py         # Buy & Hold, Random, and Moving Average strategies
├── metrics.py           # Portfolio performance metrics
├── plot_results.py      # Generates result visualizations
├── requirements.txt     # Python dependencies
│
├── models/              # Saved DQN/PPO models
├── logs/                # Training and evaluation logs
├── results/             # CSV results and portfolio histories
└── figures/             # Generated plots
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aekankpatel/stock_trading.git
cd stock_trading
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Run the full pipeline

This downloads data, trains the agents, evaluates strategies, and generates figures.

```bash
python main.py
```

### Option 2: Train for more timesteps

```bash
python main.py --timesteps 500000
```

### Option 3: Skip training and reuse saved models

```bash
python main.py --skip-train
```

### Option 4: Run individual scripts

Train models:

```bash
python train.py
```

Evaluate models:

```bash
python evaluate.py
```

Generate figures:

```bash
python plot_results.py
```

---

## Default Configuration

The default experiment uses:

| Setting | Value |
|---|---|
| Ticker | SPY |
| Start Date | 2015-01-01 |
| End Date | 2024-12-31 |
| Initial Balance | $10,000 |
| Transaction Cost | 0.1% |
| Training Timesteps | 200,000 per agent |
| Train/Test Split | 80/20 |

---

## Evaluation Metrics

The following metrics are computed for each strategy:

- Final Portfolio Value
- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown (%)
- Number of Trades
- Win Rate (%)

The results are saved in:

```text
results/metrics_summary.csv
```

Portfolio histories for each strategy are also saved in the `results/` folder.

---

## Generated Figures

The project generates the following plots:

| Figure | Description |
|---|---|
| `equity_curves.png` | Portfolio value over the test period |
| `drawdown_curves.png` | Drawdown comparison across strategies |
| `bar_comparison.png` | Return, Sharpe ratio, and drawdown comparison |
| `training_curves.png` | Smoothed training reward curves |
| `action_distribution.png` | Buy/Hold/Sell action distribution |

All figures are saved in:

```text
figures/
```

---

## Important Result Note

A key observation from this project is that the reinforcement learning agents did not always perform as expected. In particular, the checkpoint selected as the “best” model during validation could sometimes produce poor trading behavior, such as a degenerate hold-heavy policy.

Because financial data is noisy and validation episodes can be unstable, the evaluation script prioritizes loading the final trained model over the single-episode best checkpoint. This makes the final comparison more consistent and avoids over-relying on a noisy validation snapshot.

This result is important because it shows that reinforcement learning in trading is not only about training a model, but also about careful evaluation, reward design, validation strategy, and robustness testing.

---

## Key Takeaways

- Built a complete reinforcement learning trading pipeline using DQN and PPO
- Created a custom Gymnasium environment for stock trading
- Engineered technical indicators from historical SPY data
- Compared RL agents with traditional trading baselines
- Evaluated models using portfolio-level financial metrics
- Identified that the better validation checkpoint did not necessarily lead to better test performance
- Demonstrated the difficulty of applying reinforcement learning to noisy financial markets

---

## Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- yfinance
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

---

## Limitations

This project is intended for academic and experimental purposes only. It does not represent financial advice or a production-ready trading system.

Some limitations include:

- Only one ticker is used by default
- No live trading or broker integration
- No walk-forward validation
- Transaction costs are simplified
- Slippage and liquidity constraints are not fully modeled
- RL performance is sensitive to reward design and training stability

---

## Future Improvements

Possible extensions include:

- Add walk-forward validation
- Test on multiple stocks and ETFs
- Improve reward shaping
- Add position sizing instead of all-in/all-out trades
- Include transaction slippage
- Tune DQN and PPO hyperparameters more extensively
- Add recurrent policies for sequence modeling
- Compare against additional baselines
- Use risk-adjusted reward functions

---

## Disclaimer

This project is for educational purposes only. It is not intended for real-money trading or investment decision-making.

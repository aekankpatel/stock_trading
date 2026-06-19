"""
Train DQN and PPO agents on the training split and save models.

Usage:
    python train.py
"""

import os
import numpy as np
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from data_loader import load_data
from trading_env import StockTradingEnv

MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

INITIAL_BALANCE = 10_000.0
TRAIN_TIMESTEPS = 200_000  # increase for better convergence


class PortfolioRewardCallback(BaseCallback):
    """Prints mean portfolio value every N steps during training."""

    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            infos = self.locals.get("infos", [{}])
            values = [i.get("portfolio_value", 0) for i in infos if i]
            if values:
                print(f"  Step {self.n_calls:>7d} | mean portfolio: ${np.mean(values):,.2f}")
        return True


def make_env(df):
    def _init():
        env = StockTradingEnv(df, initial_balance=INITIAL_BALANCE)
        env = Monitor(env)
        return env
    return _init


def train_dqn(train_df, val_df):
    print("\n" + "=" * 60)
    print("Training DQN")
    print("=" * 60)

    train_env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(val_df)])

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=0.005,             # soft target update
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.5,   # explore for first 50% of training
        exploration_final_eps=0.15, # keep 15% random at the end
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOGS_DIR,
        verbose=0,
        seed=42,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "dqn_best"),
        log_path=os.path.join(LOGS_DIR, "dqn_eval"),
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    reward_cb = PortfolioRewardCallback(check_freq=10_000)

    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=[eval_cb, reward_cb],
        tb_log_name="DQN",
        progress_bar=True,
    )
    model.save(os.path.join(MODELS_DIR, "dqn_final"))
    print("DQN saved.")
    return model


def train_ppo(train_df, val_df):
    print("\n" + "=" * 60)
    print("Training PPO")
    print("=" * 60)

    train_env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(val_df)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,        # higher entropy → forces more action diversity
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=LOGS_DIR,
        verbose=0,
        seed=42,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "ppo_best"),
        log_path=os.path.join(LOGS_DIR, "ppo_eval"),
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    reward_cb = PortfolioRewardCallback(check_freq=10_000)

    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=[eval_cb, reward_cb],
        tb_log_name="PPO",
        progress_bar=True,
    )
    model.save(os.path.join(MODELS_DIR, "ppo_final"))
    print("PPO saved.")
    return model


if __name__ == "__main__":
    train_df, test_df = load_data(ticker="SPY", start="2015-01-01", end="2024-12-31")

    # Use last 10% of training split as validation during training
    val_split = int(len(train_df) * 0.9)
    val_df = train_df.iloc[val_split:].reset_index(drop=True)
    pure_train_df = train_df.iloc[:val_split].reset_index(drop=True)

    dqn_model = train_dqn(pure_train_df, val_df)
    ppo_model = train_ppo(pure_train_df, val_df)

    print("\nTraining complete. Run evaluate.py to backtest on test data.")

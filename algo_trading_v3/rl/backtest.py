# rl/backtest.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent.parent))

from rl.environment import TradingEnv
from rl.agent import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest(df: pd.DataFrame, model_path: str = "models/final_model.zip", config: dict = None):
    """
    Uruchamia backtest na wytrenowanym modelu
    """
    if config is None:
        config = {}

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model nie znaleziony: {model_path}")

    logger.info(f"Wczytywanie modelu z: {model_path}")

    env = TradingEnv(df, config=config)
    agent = TradingAgent(env, algorithm="PPO")
    agent.load(model_path)

    logger.info("Rozpoczynanie backtestu...")

    obs, _ = env.reset()
    done = False
    equity_curve = [env.equity]
    steps = [0]

    while not done:
        action = agent.predict(obs)
        obs, reward, done, _, info = env.step(action)
        equity_curve.append(env.equity)
        steps.append(env.current_step)

    stats = env.get_stats()

    logger.info("✅ Backtest zakończony!")

    return {
        "final_equity": stats["final_equity"],
        "total_return": stats["total_return"],
        "total_profit": stats["total_profit"],
        "total_trades": stats["total_trades"],
        "winning_trades": stats["winning_trades"],
        "losing_trades": stats["losing_trades"],
        "win_rate": stats["win_rate"],
        "max_drawdown": stats["max_drawdown"],
        "equity_curve": equity_curve,
        "steps": steps
    }
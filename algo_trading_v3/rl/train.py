# rl/train.py
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent.parent))

from rl.environment import TradingEnv
from rl.agent import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_agent(df: pd.DataFrame, config: dict = None,
                total_timesteps: int = 100_000,
                save_path: str = "models/"):
    if config is None:
        config = {}

    logger.info("Tworzenie środowiska TradingEnv...")
    env = TradingEnv(df, config=config)

    logger.info("Tworzenie agenta PPO...")
    agent = TradingAgent(env, algorithm="PPO")
    agent.create_model()

    logger.info(f"Rozpoczynanie treningu na {total_timesteps} krokach...")

    agent.train(
        total_timesteps=total_timesteps,
        save_path=save_path,
        save_freq=20000
    )

    logger.info("✅ Trening zakończony!")

    # Zbieramy statystyki
    stats = env.get_stats()

    # === ZAPIS DO PLIKU TXT ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, f"training_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"TRAINING RESULTS - {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== USTAWIENIA NAGRÓD I KAR ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== WYNIKI TRENINGU ===\n")
        f.write(f"Final Equity: ${stats['final_equity']:,.2f}\n")
        f.write(f"Total Return: {stats['total_return']:.2f}%\n")
        f.write(f"Total Profit: ${stats['total_profit']:,.2f}\n")
        f.write(f"Max Drawdown: {stats['max_drawdown']:.2f}%\n\n")

        f.write(f"Total Trades: {stats['total_trades']}\n")
        f.write(f"Winning Trades: {stats['winning_trades']}\n")
        f.write(f"Losing Trades: {stats['losing_trades']}\n")
        f.write(f"Win Rate: {stats['win_rate']:.1f}%\n\n")

        avg_profit = stats['total_profit'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        f.write(f"Average Profit per Trade: ${avg_profit:,.2f}\n\n")

        f.write(f"Model saved to: {save_path}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"📄 Wyniki zapisane do: {filename}")

    return {
        "agent": agent,
        "final_equity": stats["final_equity"],
        "total_return": stats["total_return"],
        "total_profit": stats["total_profit"],
        "total_trades": stats["total_trades"],
        "winning_trades": stats["winning_trades"],
        "losing_trades": stats["losing_trades"],
        "win_rate": stats["win_rate"],
        "max_drawdown": stats["max_drawdown"],
        "results_file": filename
    }
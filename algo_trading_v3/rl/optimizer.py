# rl/optimizer.py
import optuna
import os
import json
from datetime import datetime
from rl.environment import TradingEnv
from rl.agent import TradingAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLOptimizer:
    def __init__(self, df, base_config, n_trials=25, study_name=None):
        self.df = df
        self.base_config = base_config or {}
        self.n_trials = n_trials
        self.study_name = study_name or f"rl_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        self.best_score = -999
        self.best_model_path = None
        self.checkpoint_interval = 5
        self.storage_path = "optuna_studies.db"

    def objective(self, trial):
        config = self.base_config.copy()

        # === WSZYSTKIE PARAMETRY Z ZAKŁADKI REWARD & RISK ===
        config["reward_profit"] = trial.suggest_float("reward_profit", 1.5, 5.5)
        config["reward_unrealized"] = trial.suggest_float("reward_unrealized", -1.0, 2.0)
        config["reward_sharpe"] = trial.suggest_float("reward_sharpe", 0.3, 2.5)
        config["reward_holding"] = trial.suggest_float("reward_holding", -0.2, 0.4)

        config["penalty_drawdown"] = trial.suggest_float("penalty_drawdown", 0.8, 3.0)
        config["penalty_transaction"] = trial.suggest_float("penalty_transaction", 0.05, 0.4)
        config["penalty_holding_time"] = trial.suggest_float("penalty_holding_time", 0.02, 0.15)
        config["penalty_large_position"] = trial.suggest_float("penalty_large_position", 0.5, 3.0)
        config["penalty_inactivity"] = trial.suggest_float("penalty_inactivity", 0.02, 0.15)

        config["stop_loss_pct"] = trial.suggest_float("stop_loss_pct", 1.5, 5.0)
        config["take_profit_pct"] = trial.suggest_float("take_profit_pct", 8.0, 100.0)
        config["max_drawdown_limit"] = trial.suggest_float("max_drawdown_limit", 10.0, 30.0)
        config["risk_aversion"] = trial.suggest_float("risk_aversion", 0.4, 2.2)

        config["position_size_pct"] = trial.suggest_int("position_size_pct", 4, 20)
        config["max_position_pct"] = trial.suggest_int("max_position_pct", 15, 40)
        config["leverage"] = 5  # ← STAŁE 5x

        config["win_rate_bonus"] = trial.suggest_categorical("win_rate_bonus", [True, False])
        config["use_sortino"] = trial.suggest_categorical("use_sortino", [True, False])

        # === UŻYWAJ TYLKO STAŁYCH CECH ===
        fixed_features = ["open", "high", "low", "close", "volume",
                          "buy_volume", "sell_volume", "volume_delta", "cvd", "buy_ratio"]
        available_cols = [col for col in fixed_features if col in self.df.columns]
        df_for_env = self.df[available_cols].copy()

        env = TradingEnv(df_for_env, config=config)
        agent = TradingAgent(env, algorithm="PPO")
        agent.create_model()

        agent.train(total_timesteps=80000, save_path="models/")

        stats = env.get_stats()
        sharpe = stats.get("sharpe_ratio", 0.0)
        total_return = stats.get("total_return", 0.0)

        score = (sharpe * 0.65) + (total_return * 0.0035)

        trial_result = {
            "trial": trial.number,
            "score": round(score, 4),
            "sharpe": round(sharpe, 4),
            "total_return": round(total_return, 2),
            "win_rate": round(stats.get("win_rate", 0), 1),
            "max_drawdown": round(stats.get("max_drawdown", 0), 2),
            "params": config
        }
        self.results.append(trial_result)

        if (trial.number + 1) % self.checkpoint_interval == 0:
            self._save_checkpoint(trial.number + 1)

        if score > self.best_score:
            self.best_score = score
            self.best_model_path = f"models/best_model_trial_{trial.number}.zip"
            agent.model.save(self.best_model_path)

        return score

    def _save_checkpoint(self, trial_num):
        checkpoint = {
            "trial": trial_num,
            "best_score": self.best_score,
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/checkpoint_trial_{trial_num}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"💾 Checkpoint zapisany: {path}")

    def run(self):
        storage = f"sqlite:///{self.storage_path}"
        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True
        )

        completed = len(study.trials)
        remaining = max(0, self.n_trials - completed)

        if remaining > 0:
            logger.info(f"📌 Kontynuacja. Pozostało prób: {remaining}")
            study.optimize(self.objective, n_trials=remaining, show_progress_bar=True)
        else:
            logger.info("✅ Optymalizacja już zakończona.")

        logger.info(f"✅ Zakończono! Najlepszy wynik: {study.best_value:.4f}")

        # === ZAPIS TOP 3 TRIALI DO PLIKU TEKSTOWEGO ===
        self._save_top3_results(study)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "best_model_path": self.best_model_path,
            "all_results": self.results
        }

    def _save_top3_results(self, study):
        sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"results/optimization_top3_{timestamp}.txt"
        os.makedirs("results", exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"NAJLEPSZE 3 TRIALE Z OPTYMALIZACJI - {timestamp}\n")
            f.write("=" * 70 + "\n\n")

            for i, trial in enumerate(sorted_trials, 1):
                f.write(f"### TRIAL {trial.number} (Score: {trial.value:.4f})\n")
                f.write("=== USTAWIENIA NAGRÓD I KAR ===\n")
                for key, value in trial.params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n=== WYNIKI TRENINGU ===\n")
                f.write(f"Final Equity: ${trial.user_attrs.get('final_equity', 0):,.2f}\n")
                f.write(f"Total Return: {trial.user_attrs.get('total_return', 0):.2f}%\n")
                f.write(f"Sharpe Ratio: {trial.user_attrs.get('sharpe', 0):.4f}\n")
                f.write(f"Win Rate: {trial.user_attrs.get('win_rate', 0):.1f}%\n")
                f.write(f"Max Drawdown: {trial.user_attrs.get('max_drawdown', 0):.2f}%\n")
                f.write(f"Total Trades: {trial.user_attrs.get('total_trades', 0)}\n\n")
                f.write("-" * 70 + "\n\n")

        logger.info(f"📄 Plik z Top 3 trialami zapisany: {filename}")
"""
trader/checkpoint_manager.py
------------------------------
Zarządza zapisem checkpointów, historią sesji i top 3 wyników.

Struktura plików:
    checkpoints/
        best_<score>_ep<N>_<timestamp>.json   ← checkpoint gdy nowy best
        session_history.json                   ← pełna historia sesji
        top3.txt                               ← 3 najlepsze wyniki z parametrami
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Optional


CHECKPOINT_DIR = "checkpoints"
TOP3_FILE      = os.path.join(CHECKPOINT_DIR, "top3.txt")
HISTORY_FILE   = os.path.join(CHECKPOINT_DIR, "session_history.json")


class CheckpointManager:

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)
        self._top3: list = self._load_top3_internal()
        self._history: list = self._load_history()

    # ── Publiczny interfejs ───────────────────────────────────────────

    def save_if_best(
        self,
        episode:       int,
        trial_number:  int,
        metrics:       dict,
        reward_config: dict,
        feature_config: dict,
        optuna_params: dict,
    ) -> bool:
        """
        Zapisuje checkpoint jeśli wynik jest lepszy niż dotychczasowy best.
        Zwraca True jeśli zapisano.
        """
        score = self._composite_score(metrics)

        # Sprawdź czy lepszy niż najgorszy z top 3
        if len(self._top3) >= 3:
            worst_top3 = min(e["composite_score"] for e in self._top3)
            if score <= worst_top3:
                return False

        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"best_{score:.4f}_ep{episode}_trial{trial_number}_{ts}.json"
        filepath = os.path.join(self.dir, filename)

        data = {
            "timestamp":      ts,
            "episode":        episode,
            "trial_number":   trial_number,
            "composite_score": round(score, 4),
            "metrics":        metrics,
            "reward_config":  reward_config,
            "feature_config": feature_config,
            "optuna_params":  optuna_params,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Zaktualizuj top 3
        self._top3.append({**data, "filepath": filepath})
        self._top3.sort(key=lambda x: x["composite_score"], reverse=True)
        self._top3 = self._top3[:3]
        self._save_top3_txt()
        self._save_top3_internal()

        return True

    def add_to_history(self, episode: int, trial: int, metrics: dict, params: dict):
        """Dodaje wpis do historii sesji."""
        self._history.append({
            "episode":   episode,
            "trial":     trial,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics":   metrics,
            "params":    params,
        })
        # Zapisuj historię co 10 wpisów
        if len(self._history) % 10 == 0:
            self._save_history()

    def get_top3(self) -> list:
        return self._top3.copy()

    def get_history(self) -> list:
        return self._history.copy()

    def get_best_score(self) -> float:
        if not self._top3:
            return -999.0
        return self._top3[0]["composite_score"]

    def load_checkpoint(self, filepath: str) -> Optional[dict]:
        """Ładuje checkpoint z pliku — do wznowienia treningu."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return None

    def list_checkpoints(self) -> list[str]:
        """Lista wszystkich plików checkpoint."""
        try:
            files = [f for f in os.listdir(self.dir) if f.endswith(".json") and f.startswith("best_")]
            return sorted(files, reverse=True)
        except Exception:
            return []

    def finalize(self):
        """Wywołaj na końcu sesji — zapisuje historię i top 3."""
        self._save_history()
        self._save_top3_txt()

    # ── Prywatne ─────────────────────────────────────────────────────

    def _composite_score(self, metrics: dict) -> float:
        """
        Łączny score z metryk — wyższy = lepszy.
        Główny priorytet: dodatni return i win rate > 50%.
        Sharpe jako bonus, nie dominanta.
        """
        ret      = metrics.get("total_return_pct", 0.0)
        sharpe   = metrics.get("sharpe", 0.0)
        drawdown = metrics.get("max_drawdown_pct", 100.0)
        win_rate = metrics.get("win_rate", 0.0)
        trades   = metrics.get("total_trades", 0)

        # Kara za brak transakcji
        if trades < 5:
            return -999.0

        # Kara za ujemny return — bez zysku nie ma dobrego modelu
        if ret < 0:
            return ret * 2.0   # negatywny score proporcjonalny do straty

        # Clamp Sharpe żeby nie dominował
        sharpe_clamped = max(-3.0, min(sharpe, 5.0))

        score = (
            ret      * 0.50 +           # return najważniejszy
            sharpe_clamped * 3.0 * 0.20 +  # Sharpe skalowany rozsądnie
            (100 - drawdown) * 0.15 +   # niski drawdown
            win_rate * 0.15             # win rate
        )
        return round(score, 4)

    def _save_top3_txt(self):
        """Zapisuje top 3 do czytelnego pliku tekstowego."""
        lines = [
            "=" * 60,
            "  TOP 3 NAJLEPSZE WYNIKI TRENINGU",
            f"  Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        for i, entry in enumerate(self._top3, 1):
            m  = entry.get("metrics", {})
            rp = entry.get("optuna_params", {})
            lines += [
                f"{'─'*60}",
                f"  #{i}  Score: {entry['composite_score']:.4f}",
                f"{'─'*60}",
                f"  Epizod:         {entry.get('episode', '?')}",
                f"  Trial Optuna:   {entry.get('trial_number', '?')}",
                f"  Timestamp:      {entry.get('timestamp', '?')}",
                "",
                "  WYNIKI:",
                f"    Return:        {m.get('total_return_pct', 0):+.2f}%",
                f"    Sharpe:        {m.get('sharpe', 0):.3f}",
                f"    Max Drawdown:  {m.get('max_drawdown_pct', 0):.2f}%",
                f"    Win Rate:      {m.get('win_rate', 0):.1f}%",
                f"    Total Trades:  {m.get('total_trades', 0)}",
                f"    Final Balance: ${m.get('final_balance', 0):,.2f}",
                "",
                "  PARAMETRY OPTUNA:",
            ]
            for k, v in rp.items():
                lines.append(f"    {k:<35} {v}")

            rc = entry.get("reward_config", {})
            if rc:
                lines += ["", "  REWARD CONFIG:"]
                for k, v in rc.items():
                    lines.append(f"    {k:<35} {v}")

            lines.append("")

        lines += ["=" * 60, ""]

        try:
            with open(TOP3_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            pass

    def _save_top3_internal(self):
        path = os.path.join(self.dir, "top3_internal.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._top3, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_top3_internal(self) -> list:
        path = os.path.join(self.dir, "top3_internal.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_history(self) -> list:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
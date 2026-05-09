"""
trader/hitl_controller.py
---------------------------
Kontroler HITL/HOTL.

HOTL: Optuna automatycznie stroi parametry.
HITL: Co X epizodów system zatrzymuje się, prezentuje wyniki
      i czeka na decyzję użytkownika (akceptuj / modyfikuj / kontynuuj).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class HITLConfig:
    """Konfiguracja trybu HITL/HOTL."""
    hitl_interval:      int   = 25      # Co ile epizodów włącza się HITL
    hitl_enabled:       bool  = True    # Czy HITL jest aktywny
    plateau_threshold:  int   = 20      # Po ilu epizodach bez poprawy = plateau
    plateau_hitl:       bool  = True    # HITL przy plateau
    auto_accept_better: bool  = False   # Auto-akceptuj jeśli wynik lepszy
    hitl_timeout_sec:   int   = 300     # Timeout oczekiwania na HITL (sekundy)


@dataclass
class HITLState:
    """Aktualny stan kontrolera HITL."""
    mode:                  str   = "hotl"       # "hotl" | "hitl_waiting" | "hitl_accepted" | "hitl_modified"
    waiting_since:         Optional[float] = None
    last_hitl_episode:     int   = 0
    episodes_without_improvement: int = 0
    plateau_detected:      bool  = False
    user_message:          str   = ""
    user_modifications:    dict  = field(default_factory=dict)
    recommendation:        dict  = field(default_factory=dict)


class HITLController:
    """
    Kontroluje przepływ HITL/HOTL podczas treningu.

    Callback do UI: on_hitl_trigger(state, metrics, recommendation)
    Callback od UI: user_response(action, modifications)
    """

    def __init__(
        self,
        config:             HITLConfig,
        on_hitl_trigger:    Optional[Callable] = None,
    ):
        self.cfg      = config
        self._trigger = on_hitl_trigger
        self.state    = HITLState()
        self._best_score_at_last_hitl = -999.0

    # ── Publiczny interfejs ───────────────────────────────────────────

    def should_pause(self, episode: int, current_score: float, best_score: float) -> bool:
        """
        Sprawdza czy należy zatrzymać trening i włączyć HITL.
        Wywołuj po każdym epizodzie.
        """
        if not self.cfg.hitl_enabled:
            return False

        # Już czekamy
        if self.state.mode == "hitl_waiting":
            return True

        # Aktualizuj licznik plateau
        if current_score >= best_score * 0.99:
            self.state.episodes_without_improvement = 0
        else:
            self.state.episodes_without_improvement += 1

        # Plateau
        if (self.cfg.plateau_hitl and
                self.state.episodes_without_improvement >= self.cfg.plateau_threshold):
            self.state.plateau_detected = True
            logger.info(f"HITLController: plateau wykryte po {self.state.episodes_without_improvement} epizodach")
            return True

        # Regularny interwał
        if episode > 0 and episode % self.cfg.hitl_interval == 0:
            return True

        return False

    def trigger_hitl(
        self,
        episode:      int,
        metrics:      dict,
        best_metrics: dict,
        optuna_suggestion: dict,
    ):
        """Włącza tryb HITL — informuje UI i czeka na odpowiedź."""
        import time

        recommendation = self._build_recommendation(metrics, best_metrics, optuna_suggestion)

        self.state.mode            = "hitl_waiting"
        self.state.waiting_since   = time.time()
        self.state.last_hitl_episode = episode
        self.state.recommendation  = recommendation
        self.state.plateau_detected = False
        self.state.user_message    = ""

        logger.info(f"HITLController: HITL aktywny (epizod {episode})")

        if self._trigger:
            self._trigger(self.state, metrics, recommendation)

    def user_accept(self):
        """Użytkownik akceptuje rekomendacje Optuna — kontynuuj HOTL."""
        self.state.mode             = "hitl_accepted"
        self.state.user_modifications = {}
        self.state.episodes_without_improvement = 0
        logger.info("HITLController: użytkownik zaakceptował — kontynuuję HOTL")

    def user_modify(self, modifications: dict):
        """Użytkownik wprowadził własne modyfikacje parametrów."""
        self.state.mode              = "hitl_modified"
        self.state.user_modifications = modifications
        self.state.episodes_without_improvement = 0
        logger.info(f"HITLController: użytkownik zmodyfikował: {list(modifications.keys())}")

    def user_skip(self):
        """Użytkownik pomija HITL — kontynuuj bez zmian."""
        self.state.mode              = "hitl_accepted"
        self.state.user_modifications = {}
        logger.info("HITLController: użytkownik pominął HITL")

    def is_waiting(self) -> bool:
        return self.state.mode == "hitl_waiting"

    def get_user_modifications(self) -> dict:
        return self.state.user_modifications.copy()

    def reset_after_hitl(self):
        """Reset stanu po obsłużeniu HITL."""
        self.state.mode             = "hotl"
        self.state.user_modifications = {}
        self.state.waiting_since    = None

    # ── Generowanie rekomendacji ─────────────────────────────────────

    def _build_recommendation(
        self,
        current_metrics: dict,
        best_metrics:    dict,
        optuna_suggestion: dict,
    ) -> dict:
        """
        Buduje rekomendacje dla użytkownika na podstawie analizy wyników.
        """
        recommendations = []
        issues          = []

        ret      = current_metrics.get("total_return_pct", 0)
        sharpe   = current_metrics.get("sharpe", 0)
        drawdown = current_metrics.get("max_drawdown_pct", 0)
        win_rate = current_metrics.get("win_rate", 0)
        trades   = current_metrics.get("total_trades", 0)

        # Analiza wyników i sugestie
        if drawdown > 20:
            issues.append("Wysoki drawdown (>20%) — agent zbyt ryzykownie zarządza pozycjami")
            recommendations.append("Zwiększ loss_multiplier lub zmniejsz stop_loss_pct")

        if win_rate < 40:
            issues.append(f"Niska skuteczność ({win_rate:.0f}%) — agent za często przegrywa")
            recommendations.append("Zwiększ trend_confirm_bonus lub divergence_confirm_bonus")

        if trades < 10:
            issues.append("Za mało transakcji — agent zbyt pasywny")
            recommendations.append("Zmniejsz missed_opportunity_penalty lub idle_penalty")

        if trades > 200:
            issues.append("Za dużo transakcji — overtrading")
            recommendations.append("Zwiększ overtrading_penalty lub min_candles_between_trades")

        if sharpe < 0.5 and ret > 0:
            issues.append("Niski Sharpe — zyski nieregularne")
            recommendations.append("Zwiększ hold_profit_bonus żeby agent trzymał zyskowne pozycje dłużej")

        trend = "poprawa" if ret > best_metrics.get("total_return_pct", 0) else "brak poprawy"

        return {
            "trend":           trend,
            "issues":          issues,
            "recommendations": recommendations,
            "optuna_next":     optuna_suggestion,
            "summary": (
                f"Return: {ret:+.1f}% | Sharpe: {sharpe:.2f} | "
                f"Drawdown: {drawdown:.1f}% | WR: {win_rate:.0f}% | Trades: {trades}"
            ),
        }

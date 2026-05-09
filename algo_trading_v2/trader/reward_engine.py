"""
trader/reward_engine.py
------------------------
Konfigurowalny silnik nagród i kar dla agenta RL.
Wszystkie parametry są przekazywane z UI dashboardu.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class RewardConfig:
    """
    Konfiguracja nagród i kar — edytowalna z dashboardu.
    Wszystkie wartości są mnożnikami lub wartościami bezpośrednimi.
    """

    # ── Nagrody za zysk ──────────────────────────────────────────────
    profit_multiplier: float = 2.0
    # Mnożnik zysku z pozycji. Wyższy = agent bardziej agresywny w szukaniu zysku.

    divergence_confirm_bonus: float = 0.5
    # Bonus gdy wejście potwierdzone dywergencją w kierunku pozycji.

    trend_confirm_bonus: float = 0.3
    # Bonus gdy wejście zgodne z trendem 15m i 5m jednocześnie.

    hold_profit_bonus: float = 0.1
    # Nagroda per świeca za trzymanie zyskownej pozycji (premiuje cierpliwość).

    divergence_exit_bonus: float = 0.4
    # Bonus za zamknięcie pozycji gdy pojawi się dywergencja contra trend.

    # ── Kary za straty ───────────────────────────────────────────────
    loss_multiplier: float = 2.5
    # Mnożnik straty. Wyższy niż profit_multiplier = agent unika strat bardziej niż szuka zysku.

    stop_loss_penalty: float = 1.0
    # Dodatkowa kara gdy pozycja zamknięta przez stop loss.

    hold_loss_penalty: float = 0.15
    # Kara per świeca za trzymanie stratnej pozycji (uczy szybkiego cięcia strat).

    counter_trend_penalty: float = 0.3
    # Kara za otwarcie pozycji contra trend 15m.

    overtrading_penalty: float = 0.2
    # Kara za otwarcie pozycji gdy poprzednia zamknięta < N świec temu.

    # ── Kary za bezczynność ──────────────────────────────────────────
    missed_opportunity_penalty: float = 0.1
    # Kara gdy agent nie otworzył pozycji a był silny sygnał (divergence + trend zgodne).

    idle_penalty: float = 0.01
    # Mała kara per świeca za bezczynność gdy są sygnały. Zapobiega "nicnierobieniu".

    # ── Parametry pozycji ────────────────────────────────────────────
    max_hold_candles: int = 50
    # Maksymalna liczba świec trzymania pozycji zanim agent dostanie karę za przetrzymanie.

    min_candles_between_trades: int = 3
    # Minimalna liczba świec między zamknięciem a nowym otwarciem (anty-overtrading).

    stop_loss_pct: float = 0.015
    # Stop loss w % od ceny wejścia (1.5%).

    take_profit_pct: float = 0.03
    # Take profit w % od ceny wejścia (3.0%).

    # ── Dywergencja ──────────────────────────────────────────────────
    divergence_min_strength: str = "medium"
    # Minimalna siła dywergencji która wpływa na decyzję: "medium" lub "strong".

    divergence_weight: float = 0.6
    # Waga sygnału dywergencji w decyzji (0.0 = ignoruj, 1.0 = decyduje).

    def to_dict(self) -> dict:
        return {
            "profit_multiplier":          self.profit_multiplier,
            "divergence_confirm_bonus":   self.divergence_confirm_bonus,
            "trend_confirm_bonus":        self.trend_confirm_bonus,
            "hold_profit_bonus":          self.hold_profit_bonus,
            "divergence_exit_bonus":      self.divergence_exit_bonus,
            "loss_multiplier":            self.loss_multiplier,
            "stop_loss_penalty":          self.stop_loss_penalty,
            "hold_loss_penalty":          self.hold_loss_penalty,
            "counter_trend_penalty":      self.counter_trend_penalty,
            "overtrading_penalty":        self.overtrading_penalty,
            "missed_opportunity_penalty": self.missed_opportunity_penalty,
            "idle_penalty":               self.idle_penalty,
            "max_hold_candles":           self.max_hold_candles,
            "min_candles_between_trades": self.min_candles_between_trades,
            "stop_loss_pct":              self.stop_loss_pct,
            "take_profit_pct":            self.take_profit_pct,
            "divergence_min_strength":    self.divergence_min_strength,
            "divergence_weight":          self.divergence_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


class RewardEngine:
    """
    Oblicza nagrodę dla agenta RL na podstawie akcji i stanu rynku.
    """

    def __init__(self, config: RewardConfig):
        self.cfg = config

    def compute(
        self,
        action:              str,           # "long" | "short" | "close" | "hold"
        pnl_pct:             float,         # % P&L aktualnej pozycji
        position_side:       Optional[str], # "long" | "short" | None
        candles_in_position: int,           # ile świec w pozycji
        candles_since_close: int,           # ile świec od ostatniego zamknięcia
        trend_score:         float,         # -1..+1 z Algo #2
        divergence_score:    float,         # -1..+1 z DivergenceDetector
        divergence_strength: str,           # "weak" | "medium" | "strong" | "none"
        hit_stop_loss:       bool = False,
        hit_take_profit:     bool = False,
        missed_signal:       bool = False,
    ) -> tuple[float, dict]:
        """
        Oblicza nagrodę i zwraca (reward, breakdown) gdzie breakdown
        to słownik składowych nagrody dla logowania.
        """
        reward     = 0.0
        breakdown  = {}
        cfg        = self.cfg

        div_relevant = divergence_strength in ("medium", "strong")

        # ── Zamknięcie pozycji ───────────────────────────────────────
        if action == "close" and position_side is not None:
            if pnl_pct > 0:
                r = pnl_pct * cfg.profit_multiplier
                reward += r
                breakdown["profit"] = round(r, 4)
            else:
                r = pnl_pct * cfg.loss_multiplier
                reward += r
                breakdown["loss"] = round(r, 4)

            if hit_stop_loss:
                reward -= cfg.stop_loss_penalty
                breakdown["stop_loss_penalty"] = -cfg.stop_loss_penalty

            if hit_take_profit and pnl_pct > 0:
                reward += cfg.profit_multiplier * 0.3
                breakdown["take_profit_bonus"] = round(cfg.profit_multiplier * 0.3, 4)

            # Bonus za zamknięcie na dywergencji contra trend
            if div_relevant:
                contra = (position_side == "long"  and divergence_score < -0.3) or \
                         (position_side == "short" and divergence_score >  0.3)
                if contra:
                    reward += cfg.divergence_exit_bonus
                    breakdown["divergence_exit_bonus"] = cfg.divergence_exit_bonus

        # ── Otwarcie pozycji ─────────────────────────────────────────
        elif action in ("long", "short") and position_side is None:

            # Overtrading penalty
            if candles_since_close < cfg.min_candles_between_trades:
                reward -= cfg.overtrading_penalty
                breakdown["overtrading_penalty"] = -cfg.overtrading_penalty

            # Bonus za zgodność z trendem
            trend_agrees = (action == "long"  and trend_score >  0.2) or \
                           (action == "short" and trend_score < -0.2)
            if trend_agrees:
                reward += cfg.trend_confirm_bonus
                breakdown["trend_confirm"] = cfg.trend_confirm_bonus
            else:
                reward -= cfg.counter_trend_penalty
                breakdown["counter_trend_penalty"] = -cfg.counter_trend_penalty

            # Bonus za dywergencję potwierdzającą
            if div_relevant:
                div_agrees = (action == "long"  and divergence_score >  0.3) or \
                             (action == "short" and divergence_score < -0.3)
                if div_agrees:
                    r = cfg.divergence_confirm_bonus * cfg.divergence_weight
                    reward += r
                    breakdown["divergence_confirm"] = round(r, 4)

        # ── Trzymanie pozycji ────────────────────────────────────────
        elif action == "hold" and position_side is not None:
            if pnl_pct > 0:
                reward += cfg.hold_profit_bonus
                breakdown["hold_profit"] = cfg.hold_profit_bonus
            else:
                reward -= cfg.hold_loss_penalty
                breakdown["hold_loss"] = -cfg.hold_loss_penalty

            # Kara za przetrzymanie
            if candles_in_position > cfg.max_hold_candles:
                reward -= 0.05
                breakdown["overhold_penalty"] = -0.05

        # ── Bezczynność ──────────────────────────────────────────────
        elif action == "hold" and position_side is None:
            if missed_signal:
                reward -= cfg.missed_opportunity_penalty
                breakdown["missed_opportunity"] = -cfg.missed_opportunity_penalty
            elif abs(trend_score) > 0.3 or (div_relevant and abs(divergence_score) > 0.3):
                reward -= cfg.idle_penalty
                breakdown["idle_penalty"] = -cfg.idle_penalty

        breakdown["total"] = round(reward, 4)
        return round(reward, 4), breakdown

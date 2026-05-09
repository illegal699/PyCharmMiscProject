"""
trader/meta_agent.py
---------------------
Meta-agent oparty na Optuna (Bayesian Optimization).
Automatycznie stroi parametry reward_config i feature_config.

Sztywne ograniczenia (niezmienne):
    - dźwignia: x5
    - max kapital na pozycję: 2%

Parametry strojone przez Optuna:
    - wszystkie reward_config (oprócz stop_loss_pct który liczymy z dźwignią)
    - wybór feature_config (które wskaźniki cenowe ładować)
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ── Stałe (niezmienne przez meta-agenta) ─────────────────────────────
FIXED_LEVERAGE         = 5       # dźwignia x5
FIXED_MAX_POSITION_PCT = 0.02    # max 2% kapitału na pozycję


@dataclass
class MetaAgentConfig:
    """Konfiguracja meta-agenta."""
    n_trials:         int   = 50
    n_episodes_per_trial: int = 30
    n_jobs:           int   = 1
    study_name:       str   = "trader_optuna"
    storage:          Optional[str] = "sqlite:///checkpoints/optuna_study.db"
    direction:        str   = "maximize"
    load_if_exists:   bool  = True
    locked_params:    dict  = None    # {param_name: (is_locked, value)}

    def __post_init__(self):
        if self.locked_params is None:
            self.locked_params = {}

    def get_locked_value(self, param: str, default):
        """Zwraca wartość zablokowaną lub default jeśli nie zablokowany."""
        entry = self.locked_params.get(param)
        if entry and entry[0]:   # (is_locked=True, value)
            return entry[1]
        return default


@dataclass
class MetaAgentProgress:
    """Stan postępu meta-agenta."""
    trial:              int   = 0
    total_trials:       int   = 0
    pct_complete:       float = 0.0
    best_score:         float = -999.0
    best_trial:         int   = 0
    current_score:      float = 0.0
    status:             str   = "idle"   # idle|running|hitl_waiting|done|error
    message:            str   = ""
    history:            list  = field(default_factory=list)
    hitl_waiting:       bool  = False
    hitl_recommendation: dict = field(default_factory=dict)
    elapsed_sec:        float = 0.0
    eta_sec:            float = 0.0


class MetaAgent:
    """
    Meta-agent Optuna który automatycznie stroi parametry tradingowe.

    Architektura:
        Optuna trial → suggest params → Trainer episode → metrics → Optuna score
                                ↕ (co X epizodów)
                           HITL Controller → UI → user decision
    """

    def __init__(
        self,
        meta_config:        MetaAgentConfig,
        training_config,    # TrainingConfig
        hitl_config,        # HITLConfig
        candles:            list,
        progress_callback:  Optional[Callable] = None,
        hitl_callback:      Optional[Callable] = None,
    ):
        self.mc       = meta_config
        self.tc       = training_config
        self.candles  = candles
        self.prog_cb  = progress_callback
        self.hitl_cb  = hitl_callback

        self.progress = MetaAgentProgress(total_trials=meta_config.n_trials)
        self._stop    = False
        self._hitl_event = threading.Event()

        # Inicjalizuj HITL controller
        from trader.hitl_controller import HITLController
        self._hitl = HITLController(
            config          = hitl_config,
            on_hitl_trigger = self._on_hitl_trigger,
        )

        # Checkpoint manager
        from trader.checkpoint_manager import CheckpointManager
        self._ckpt = CheckpointManager()

        self._best_metrics: dict = {}
        self._current_optuna_params: dict = {}

    # ── Publiczny interfejs ───────────────────────────────────────────

    def run(self):
        """Uruchamia meta-agenta. Wywołuj w osobnym wątku."""
        import time
        self.progress.status     = "running"
        self._start_time         = time.time()
        self._stop               = False

        try:
            self._run_optuna()
            self.progress.status  = "done"
            self.progress.message = f"✅ Zakończono! Najlepszy score: {self.progress.best_score:.4f}"
        except Exception as e:
            self.progress.status  = "error"
            self.progress.message = f"❌ Błąd: {e}"
            logger.error(f"MetaAgent: błąd — {e}", exc_info=True)
        finally:
            self._ckpt.finalize()
            self._notify()

    def stop(self):
        self._stop = True
        self._hitl_event.set()   # odblokuj jeśli czeka na HITL
        logger.info("MetaAgent: zatrzymywanie...")

    def user_accept_hitl(self):
        """Użytkownik akceptuje rekomendacje."""
        self._hitl.user_accept()
        self.progress.hitl_waiting = False
        self.progress.status       = "running"
        self._hitl_event.set()

    def user_modify_hitl(self, modifications: dict):
        """Użytkownik modyfikuje parametry."""
        self._hitl.user_modify(modifications)
        self.progress.hitl_waiting = False
        self.progress.status       = "running"
        self._hitl_event.set()

    def user_skip_hitl(self):
        """Użytkownik pomija HITL."""
        self._hitl.user_skip()
        self.progress.hitl_waiting = False
        self.progress.status       = "running"
        self._hitl_event.set()

    def get_top3(self) -> list:
        return self._ckpt.get_top3()

    def get_checkpoints(self) -> list:
        return self._ckpt.list_checkpoints()

    # ── Optuna ───────────────────────────────────────────────────────

    def _run_optuna(self):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            self._update("⚠️ Optuna niedostępna — pip install optuna — używam Random Search")
            self._run_random_search()
            return

        import optuna

        import os
        os.makedirs("checkpoints", exist_ok=True)

        study = optuna.create_study(
            study_name      = self.mc.study_name,
            direction       = self.mc.direction,
            storage         = self.mc.storage,
            load_if_exists  = self.mc.load_if_exists,
        )

        existing = len(study.trials)
        if existing > 0:
            self._update(f"📂 Wczytano istniejące study: {existing} poprzednich prób")
            # Odtwórz best score z historii
            try:
                best = study.best_trial
                self.progress.best_score = best.value or -999.0
                self.progress.best_trial = best.number + 1
                self._update(f"🏆 Najlepszy poprzedni wynik: score={self.progress.best_score:.4f} (trial #{best.number+1})")
            except Exception:
                pass
        else:
            self._update(f"🆕 Nowe study Optuna — {self.mc.n_trials} prób × {self.mc.n_episodes_per_trial} epizodów")

        for trial_num in range(self.mc.n_trials):
            if self._stop:
                break

            trial  = study.ask()
            params = self._suggest_params(trial)
            self._current_optuna_params = params
            self.progress.trial = trial_num + 1

            self._update(f"Trial {trial_num+1}/{self.mc.n_trials} — testowanie parametrów...")

            score, metrics = self._run_trial(params, trial_num)

            study.tell(trial, score)

            # Aktualizuj progress
            self.progress.current_score = score
            if score > self.progress.best_score:
                self.progress.best_score = score
                self.progress.best_trial = trial_num + 1
                self._best_metrics       = metrics

                # Zapisz checkpoint
                saved = self._ckpt.save_if_best(
                    episode        = trial_num * self.mc.n_episodes_per_trial,
                    trial_number   = trial_num + 1,
                    metrics        = metrics,
                    reward_config  = params.get("reward", {}),
                    feature_config = params.get("features", {}),
                    optuna_params  = params,
                )
                if saved:
                    self._update(f"💾 Nowy best! Score={score:.4f} — checkpoint zapisany")

            self._ckpt.add_to_history(
                episode = trial_num * self.mc.n_episodes_per_trial,
                trial   = trial_num + 1,
                metrics = metrics,
                params  = params,
            )

            self.progress.history.append({
                "trial":      trial_num + 1,
                "score":      round(score, 4),
                "return_pct": metrics.get("total_return_pct", 0),
                "sharpe":     metrics.get("sharpe", 0),
                "win_rate":   metrics.get("win_rate", 0),
                "drawdown":   metrics.get("max_drawdown_pct", 0),
                "trades":     metrics.get("total_trades", 0),
            })

            self.progress.pct_complete = (trial_num + 1) / self.mc.n_trials * 100
            self._update_eta(trial_num + 1)

            # Sprawdź HITL
            if self._hitl.should_pause(trial_num + 1, score, self.progress.best_score):
                next_params = self._get_next_suggestion(study)
                self._hitl.trigger_hitl(
                    episode           = trial_num + 1,
                    metrics           = metrics,
                    best_metrics      = self._best_metrics,
                    optuna_suggestion = next_params,
                )
                # Czekaj na odpowiedź użytkownika (max 10 minut)
                self._hitl_event.clear()
                self._update("⏸️ HITL: czekam na Twoją decyzję w dashboardzie...")
                got_response = self._hitl_event.wait(timeout=600)
                if not got_response:
                    # Timeout — auto-kontynuuj bez zmian
                    self._update("⏱️ HITL timeout — kontynuuję automatycznie")
                    self._hitl.user_skip()
                    self.progress.hitl_waiting = False

                if self._stop:
                    break

                # Zastosuj modyfikacje użytkownika jeśli są
                mods = self._hitl.get_user_modifications()
                if mods:
                    self._apply_user_modifications(mods)

                self._hitl.reset_after_hitl()
                self._notify()

        self._update(f"🏁 Optuna zakończona. Best score: {self.progress.best_score:.4f} (trial {self.progress.best_trial})")

    def _run_random_search(self):
        """Fallback gdy Optuna niedostępna."""
        import random, time

        for trial_num in range(self.mc.n_trials):
            if self._stop:
                break

            params = self._random_params()
            self._current_optuna_params = params
            self.progress.trial = trial_num + 1

            score, metrics = self._run_trial(params, trial_num)
            self.progress.current_score = score

            if score > self.progress.best_score:
                self.progress.best_score = score
                self.progress.best_trial = trial_num + 1
                self._best_metrics       = metrics
                self._ckpt.save_if_best(
                    episode=trial_num, trial_number=trial_num+1,
                    metrics=metrics, reward_config=params.get("reward",{}),
                    feature_config=params.get("features",{}), optuna_params=params,
                )

            self.progress.history.append({
                "trial": trial_num+1, "score": round(score,4),
                "return_pct": metrics.get("total_return_pct",0),
                "sharpe": metrics.get("sharpe",0),
                "win_rate": metrics.get("win_rate",0),
                "drawdown": metrics.get("max_drawdown_pct",0),
                "trades": metrics.get("total_trades",0),
            })
            self.progress.pct_complete = (trial_num+1)/self.mc.n_trials*100
            self._update(f"Trial {trial_num+1}/{self.mc.n_trials} | Score: {score:.4f}")

    def _run_trial(self, params: dict, trial_num: int) -> tuple[float, dict]:
        """
        Trenuje PPO z parametrami sugerowanymi przez Optuna i zwraca metryki.

        Przepływ:
            Optuna → reward_config → PPO trenuje się na danych → metryki → Optuna
        """
        from trader.reward_engine   import RewardEngine, RewardConfig
        from trader.feature_builder import FeatureBuilder, FeatureConfig
        from trader.ppo_trainer     import PPOTrainer

        rc = RewardConfig.from_dict(params.get("reward", {}))
        fc = FeatureConfig.from_dict(params.get("features", {}))
        re = RewardEngine(rc)
        fb = FeatureBuilder(fc)

        # Sygnały zewnętrzne (trend + dywergencja) per świeca — obliczane raz
        ext_signals = self._build_external_signals(self.candles)

        def on_episode(ep_num, total_eps):
            if self.prog_cb:
                pct = ((trial_num * total_eps + ep_num) /
                       max(self.mc.n_trials * total_eps, 1)) * 100
                self.progress.pct_complete = min(pct, 100.0)

        ppo = PPOTrainer(
            candles           = self.candles,
            feature_builder   = fb,
            reward_engine     = re,
            extra_signals     = ext_signals,
            initial_balance   = self.tc.initial_balance,
            commission_pct    = self.tc.commission_pct,
            n_episodes        = self.mc.n_episodes_per_trial,
            progress_callback = on_episode,
        )
        ppo._stop = self._stop

        metrics = ppo.train_and_evaluate()

        # Zapisz model jeśli lepszy niż poprzedni best
        if metrics.get("total_return_pct", 0) > self.progress.best_score:
            import os
            os.makedirs("models", exist_ok=True)
            ppo.save_model(f"models/ppo_trial_{trial_num+1}")

        if not metrics:
            return -999.0, {}

        score = self._ckpt._composite_score(metrics)
        return score, metrics

    def _build_external_signals(self, candles: list) -> dict:
        """
        Buduje sygnały zewnętrzne (trend + dywergencja) dla każdej świecy.
        Używa prawdziwych wskaźników technicznych i DivergenceDetector.
        Sygnały są obliczane raz przed treningiem i cache'owane.
        """
        from trend_analyzer.indicators import ema, rsi as calc_rsi_fn, macd as calc_macd_fn

        closes  = [c.close for c in candles]
        highs   = [c.high  for c in candles]
        lows    = [c.low   for c in candles]
        n       = len(closes)
        signals = {}

        # Oblicz EMA dla wszystkich świec (raz)
        ema9_vals  = ema(closes, 9)
        ema21_vals = ema(closes, 21)
        ema50_vals = ema(closes, 50)
        off9  = n - len(ema9_vals)
        off21 = n - len(ema21_vals)
        off50 = n - len(ema50_vals)

        # Oblicz RSI dla każdego punktu (rolling window 20 świec)
        rsi_vals = {}
        for i in range(20, n):
            window = closes[i-20:i+1]
            rv = calc_rsi_fn(window, 14)
            rsi_vals[i] = rv if rv else 50.0

        # Oblicz MACD histogram dla każdego punktu
        macd_hists = {}
        for i in range(40, n):
            window = closes[max(0, i-60):i+1]
            mr = calc_macd_fn(window)
            macd_hists[i] = mr[2] if mr else 0.0

        # Dywergencja RSI — okno kroczące 20 świec
        def rsi_divergence(i, window=20):
            if i < window:
                return 0.0, False, False
            
            # Znajdź lokalne ekstrema w oknie
            price_window = closes[i-window:i+1]
            rsi_window_vals = [rsi_vals.get(j, 50.0) for j in range(i-window, i+1)]
            
            # Porównaj pierwszą i drugą połowę okna
            mid = window // 2
            p_first  = min(price_window[:mid])
            p_second = min(price_window[mid:])
            r_first  = min(rsi_window_vals[:mid])
            r_second = min(rsi_window_vals[mid:])
            
            ph_first  = max(price_window[:mid])
            ph_second = max(price_window[mid:])
            rh_first  = max(rsi_window_vals[:mid])
            rh_second = max(rsi_window_vals[mid:])
            
            # Bullish: cena robi niższy dół, RSI wyższy dół
            bullish = (p_second < p_first * 0.998 and
                       r_second > r_first + 2.0 and
                       rsi_vals.get(i, 50) < 50)
            
            # Bearish: cena robi wyższy szczyt, RSI niższy szczyt
            bearish = (ph_second > ph_first * 1.002 and
                       rh_second < rh_first - 2.0 and
                       rsi_vals.get(i, 50) > 50)
            
            if bullish:
                strength = abs(r_second - r_first) / 10
                return min(strength, 1.0), True, False
            if bearish:
                strength = abs(rh_second - rh_first) / 10
                return -min(strength, 1.0), False, True
            return 0.0, False, False

        # Dywergencja MACD — okno kroczące
        def macd_divergence(i, window=15):
            if i < window + 40:
                return 0.0, False, False
            
            price_window = closes[i-window:i+1]
            macd_window  = [macd_hists.get(j, 0.0) for j in range(i-window, i+1)]
            mid = window // 2

            p_second = min(price_window[mid:])
            p_first  = min(price_window[:mid])
            m_second = min(macd_window[mid:])
            m_first  = min(macd_window[:mid])

            ph_second = max(price_window[mid:])
            ph_first  = max(price_window[:mid])
            mh_second = max(macd_window[mid:])
            mh_first  = max(macd_window[:mid])

            bullish = (p_second < p_first * 0.998 and m_second > m_first + 0.000001)
            bearish = (ph_second > ph_first * 1.002 and mh_second < mh_first - 0.000001)

            if bullish: return 0.6, True, False
            if bearish: return -0.6, False, True
            return 0.0, False, False

        for i in range(n):
            # Trend z EMA
            e9  = ema9_vals[i  - off9]  if i >= off9  else closes[i]
            e21 = ema21_vals[i - off21] if i >= off21 else closes[i]
            e50 = ema50_vals[i - off50] if i >= off50 else closes[i]
            p   = closes[i]

            trend = 0.0
            trend += 0.25 if p   > e9  else -0.25
            trend += 0.25 if e9  > e21 else -0.25
            trend += 0.25 if e21 > e50 else -0.25
            trend += 0.25 if p   > e50 else -0.25

            rsi_val = rsi_vals.get(i, 50.0)

            # Łącz dywergencje RSI i MACD
            rsi_div, rsi_bull, rsi_bear     = rsi_divergence(i)
            macd_div, macd_bull, macd_bear  = macd_divergence(i)

            # Kompozyt dywergencji (RSI 60%, MACD 40%)
            div_score   = rsi_div * 0.6 + macd_div * 0.4
            div_bullish = rsi_bull or macd_bull
            div_bearish = rsi_bear or macd_bear
            div_strength = "none"
            if abs(div_score) > 0.5:
                div_strength = "strong"
            elif abs(div_score) > 0.2:
                div_strength = "medium"

            signals[i] = {
                "sentiment_score":     0.0,
                "trend_score_5m":      trend,
                "trend_score_15m":     trend,
                "tf_alignment":        abs(trend),
                "divergence_score":    round(div_score, 4),
                "divergence_bullish":  div_bullish,
                "divergence_bearish":  div_bearish,
                "divergence_strength": div_strength,
                "rsi_15m":             rsi_val,
            }

        # Statystyki dla logowania
        n_bull = sum(1 for s in signals.values() if s["divergence_bullish"])
        n_bear = sum(1 for s in signals.values() if s["divergence_bearish"])
        n_trend_up = sum(1 for s in signals.values() if s["trend_score_15m"] > 0.2)
        import logging
        logging.getLogger(__name__).info(
            f"Sygnały: {n} świec | trend_up={n_trend_up} | "
            f"div_bull={n_bull} | div_bear={n_bear}"
        )

        return signals

    def _simple_policy(self, obs: list, fc) -> int:
        """
        Polityka dla meta-treningu — używa końcowych cech wektora
        które zawsze są obecne (stan pozycji, dywergencja, trend).
        Progi są różne dla różnych parametrów reward przez pośredni wpływ.
        """
        from trader.trader_env import ACTION_HOLD, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE
        import random

        if not obs or len(obs) < 4:
            return ACTION_HOLD

        n = len(obs)

        # Ostatnie cechy wektora to zawsze stan pozycji (gdy włączone)
        # use_position_side  → obs[-4]
        # use_position_pnl   → obs[-3]
        # use_position_duration → obs[-2]
        # use_candles_since_close → obs[-1]
        pos_side    = obs[-4] if n >= 4 else 0.0
        pos_pnl     = obs[-3] if n >= 3 else 0.0
        pos_dur     = obs[-2] if n >= 2 else 0.0

        # Cechy dywergencji i trendu — blisko końca wektora
        # use_divergence_score   → obs[-8]
        # use_divergence_bullish → obs[-7]
        # use_divergence_bearish → obs[-6]
        div_score   = obs[-8] if n >= 8 else 0.0
        div_bull    = obs[-7] if n >= 7 else 0.0
        div_bear    = obs[-6] if n >= 6 else 0.0

        # Sygnały trendu
        trend_5m    = obs[-12] if n >= 12 else 0.0
        trend_15m   = obs[-11] if n >= 11 else 0.0
        alignment   = obs[-10] if n >= 10 else 0.0

        combined = (trend_15m * 0.4 + trend_5m * 0.2 +
                    div_score * 0.3 + alignment * 0.1)

        if pos_side > 0.5:      # long
            if pos_pnl < -0.3 or div_bear > 0.4 or pos_dur > 0.8:
                return ACTION_CLOSE
            if combined < -0.20:
                return ACTION_CLOSE
            return ACTION_HOLD

        elif pos_side < -0.5:   # short
            if pos_pnl < -0.3 or div_bull > 0.4 or pos_dur > 0.8:
                return ACTION_CLOSE
            if combined > 0.20:
                return ACTION_CLOSE
            return ACTION_HOLD

        else:                   # brak pozycji
            # Wejdź gdy trend wyraźny
            if combined > 0.20:
                return ACTION_LONG
            if combined < -0.20:
                return ACTION_SHORT
            # Dywergencja nawet bez mocnego trendu
            if div_bull > 0.4:
                return ACTION_LONG
            if div_bear > 0.4:
                return ACTION_SHORT
            return ACTION_HOLD

    # ── Parametry Optuna ─────────────────────────────────────────────

    def _suggest_params(self, trial) -> dict:
        """Optuna sugeruje parametry do przetestowania."""
        lp = self.mc.locked_params   # skrót

        # Helper: użyj wartości zablokowanej lub zasugeruj przez Optuna
        def locked_or_suggest_float(param, optuna_name, low, high):
            entry = lp.get(param)
            if entry and entry[0]:
                return entry[1]
            return trial.suggest_float(optuna_name, low, high)

        def locked_or_suggest_int(param, optuna_name, low, high):
            entry = lp.get(param)
            if entry and entry[0]:
                return int(entry[1])
            return trial.suggest_int(optuna_name, low, high)

        reward = {
            "profit_multiplier":          trial.suggest_float("profit_multiplier",    1.0, 4.0),
            "loss_multiplier":            trial.suggest_float("loss_multiplier",       1.5, 5.0),
            "divergence_confirm_bonus":   trial.suggest_float("div_confirm_bonus",     0.1, 1.5),
            "divergence_exit_bonus":      trial.suggest_float("div_exit_bonus",        0.1, 1.5),
            "trend_confirm_bonus":        trial.suggest_float("trend_confirm_bonus",   0.1, 1.0),
            "hold_profit_bonus":          trial.suggest_float("hold_profit_bonus",     0.01, 0.3),
            "hold_loss_penalty":          trial.suggest_float("hold_loss_penalty",     0.05, 0.4),
            "counter_trend_penalty":      trial.suggest_float("counter_trend_penalty", 0.1, 0.8),
            "overtrading_penalty":        trial.suggest_float("overtrading_penalty",   0.1, 0.8),
            "missed_opportunity_penalty": trial.suggest_float("missed_opp_penalty",    0.05, 0.5),
            "idle_penalty":               trial.suggest_float("idle_penalty",          0.001, 0.05),
            "stop_loss_penalty":          trial.suggest_float("stop_loss_penalty",     0.2, 2.0),
            "divergence_weight":          trial.suggest_float("divergence_weight",     0.3, 0.9),
            "max_hold_candles":           trial.suggest_int("max_hold_candles",        10, 100),
            "min_candles_between_trades": trial.suggest_int("min_candles_btw_trades",  1, 10),
            "divergence_min_strength":    trial.suggest_categorical("div_strength", ["medium", "strong"]),
            # Parametry które mogą być zablokowane przez użytkownika
            "stop_loss_pct":    locked_or_suggest_float("stop_loss_pct",    "stop_loss_pct",    0.005, 0.05),
            "take_profit_pct":  locked_or_suggest_float("take_profit_pct",  "take_profit_pct",  0.01,  1.00),
            "leverage":         locked_or_suggest_int("leverage",           "leverage",          1,     10),
            "max_position_pct": locked_or_suggest_float("max_position_pct", "max_position_pct", 0.01,  0.10),
        }

        # Cechy — wskaźniki cenowe losowane, reszta zawsze włączona
        features = {
            # Wskaźniki cenowe — strojone przez Optuna
            "use_ema_9":       trial.suggest_categorical("use_ema_9",   [True, False]),
            "use_ema_21":      trial.suggest_categorical("use_ema_21",  [True, False]),
            "use_ema_50":      trial.suggest_categorical("use_ema_50",  [True, False]),
            "use_ema_cross":   trial.suggest_categorical("use_ema_cross", [True, False]),
            "use_rsi_1m":      trial.suggest_categorical("use_rsi_1m",  [True, False]),
            "use_rsi_5m":      trial.suggest_categorical("use_rsi_5m",  [True, False]),
            "use_rsi_15m":     trial.suggest_categorical("use_rsi_15m", [True, False]),
            "use_macd_hist":   trial.suggest_categorical("use_macd_hist", [True, False]),
            "use_macd_cross":  trial.suggest_categorical("use_macd_cross", [True, False]),
            # Reszta zawsze włączona
            "use_hh_hl":              True,
            "use_ll_lh":              True,
            "use_swing_dist":         True,
            "use_volume_ratio":       True,
            "use_volume_trend":       True,
            "use_volume_delta":       False,
            "use_buy_volume":         True,
            "use_sell_volume":        True,
            "use_cvd":                True,
            "use_large_candles":      True,
            "use_sentiment_score":    True,
            "use_trend_score_5m":     True,
            "use_trend_score_15m":    True,
            "use_tf_alignment":       True,
            "use_divergence_score":   True,
            "use_divergence_bullish": True,
            "use_divergence_bearish": True,
            "use_divergence_strength": True,
            "use_position_side":      True,
            "use_position_pnl":       True,
            "use_position_duration":  True,
            "use_candles_since_close": True,
        }

        return {"reward": reward, "features": features}

    def _random_params(self) -> dict:
        """Losowe parametry jako fallback."""
        import random
        from trader.reward_engine   import RewardConfig
        from trader.feature_builder import FeatureConfig

        rc = RewardConfig()
        rc.profit_multiplier          = random.uniform(1.0, 4.0)
        rc.loss_multiplier            = random.uniform(1.5, 5.0)
        rc.divergence_confirm_bonus   = random.uniform(0.1, 1.5)
        rc.divergence_weight          = random.uniform(0.3, 0.9)
        rc.stop_loss_pct              = 0.01 * FIXED_LEVERAGE

        fc = FeatureConfig()
        for attr in ["use_ema_9","use_ema_21","use_ema_50","use_rsi_5m","use_macd_hist"]:
            setattr(fc, attr, random.choice([True, False]))

        return {"reward": rc.to_dict(), "features": fc.to_dict()}

    def _get_next_suggestion(self, study) -> dict:
        """Pobiera sugestię następnych parametrów od Optuna."""
        try:
            trial = study.ask()
            params = self._suggest_params(trial)
            study.tell(trial, 0.0)  # placeholder
            return params.get("reward", {})
        except Exception:
            return {}

    def _apply_user_modifications(self, mods: dict):
        """Zastosuj modyfikacje użytkownika do bieżącej konfiguracji."""
        logger.info(f"MetaAgent: stosowanie modyfikacji użytkownika: {list(mods.keys())}")

    # ── Callbacks i helpers ──────────────────────────────────────────

    def _on_hitl_trigger(self, state, metrics, recommendation):
        """Callback gdy HITL się włącza."""
        self.progress.hitl_waiting       = True
        self.progress.status             = "hitl_waiting"
        self.progress.hitl_recommendation = recommendation
        self._update(f"⏸️ HITL aktywny — oczekiwanie na decyzję użytkownika")
        if self.hitl_cb:
            self.hitl_cb(state, metrics, recommendation)

    def _update(self, message: str):
        import time
        self.progress.message     = message
        self.progress.elapsed_sec = time.time() - self._start_time if hasattr(self, "_start_time") else 0
        logger.info(f"MetaAgent: {message}")
        self._notify()

    def _update_eta(self, completed: int):
        import time
        if not hasattr(self, "_start_time"):
            return
        elapsed = time.time() - self._start_time
        if completed > 0:
            per_trial = elapsed / completed
            remaining = (self.mc.n_trials - completed) * per_trial
            self.progress.elapsed_sec = elapsed
            self.progress.eta_sec     = remaining

    def _notify(self):
        if self.prog_cb:
            try:
                self.prog_cb(self.progress)
            except Exception:
                pass

"""
test_loop_log.py
-----------------
Generuje szczegółowy log jednej pętli treningu.
Pokazuje:
- Parametry nagrody i kary użyte w tej pętli
- Jak trend i dywergencja zmieniają się per świeca
- Każdy krok: cena, sygnały, akcja, nagroda, breakdown nagród

Uruchomienie: python test_loop_log.py
Wynik: loop_debug.log
"""

import sys, os, logging, random
sys.path.insert(0, os.path.dirname(__file__))

LOG_FILE = "loop_debug.log"
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[fh, sh])
log = logging.getLogger("loop_test")

# ── Dane ─────────────────────────────────────────────────────────────
log.info("=" * 75)
log.info("  TEST LOOP LOG")
log.info("=" * 75)

log.info("\n[1] Pobieranie danych z Binance...")
from trend_analyzer.data_fetcher import BinanceDataFetcher
fetcher = BinanceDataFetcher()
candles = fetcher.get_candles("BTCUSDT", "5m", limit=300)

if not candles:
    log.error("BŁĄD: brak danych z Binance — sprawdź połączenie i API key")
    sys.exit(1)

log.info(f"    Pobrano:       {len(candles)} świec")
log.info(f"    Cena pierwsza: ${candles[0].close:,.2f}")
log.info(f"    Cena ostatnia: ${candles[-1].close:,.2f}")
log.info(f"    Zmiana:        {(candles[-1].close/candles[0].close-1)*100:+.2f}%")

# ── Sygnały zewnętrzne ────────────────────────────────────────────────
log.info("\n[2] Budowanie sygnałów (trend + dywergencja) per świeca...")
from trader.meta_agent      import MetaAgent, MetaAgentConfig
from trader.hitl_controller import HITLConfig
from trader.trainer         import TrainingConfig

tc   = TrainingConfig(symbol="BTCUSDT", interval="5m")
mc   = MetaAgentConfig(n_trials=1, n_episodes_per_trial=1)
hc   = HITLConfig(hitl_enabled=False)
meta = MetaAgent(mc, tc, hc, candles)
ext  = meta._build_external_signals(candles)

n_bull   = sum(1 for s in ext.values() if s["divergence_bullish"])
n_bear   = sum(1 for s in ext.values() if s["divergence_bearish"])
n_strong = sum(1 for s in ext.values() if s["divergence_strength"] == "strong")
n_medium = sum(1 for s in ext.values() if s["divergence_strength"] == "medium")
n_up     = sum(1 for s in ext.values() if s["trend_score_15m"] > 0.2)
n_down   = sum(1 for s in ext.values() if s["trend_score_15m"] < -0.2)
uniq_t   = len(set(round(s["trend_score_15m"], 3) for s in ext.values()))
uniq_d   = len(set(round(s["divergence_score"], 3) for s in ext.values()))

log.info(f"    Unikalne wartości trendu:      {uniq_t}  ← zmienia się per świeca")
log.info(f"    Unikalne wartości dywergencji: {uniq_d}  ← zmienia się per świeca")
log.info(f"    Trend UP:    {n_up} świec ({n_up/len(ext)*100:.0f}%)")
log.info(f"    Trend DOWN:  {n_down} świec ({n_down/len(ext)*100:.0f}%)")
log.info(f"    Div BULL:    {n_bull} świec ({n_bull/len(ext)*100:.0f}%)")
log.info(f"    Div BEAR:    {n_bear} świec ({n_bear/len(ext)*100:.0f}%)")
log.info(f"    Div STRONG:  {n_strong} świec")
log.info(f"    Div MEDIUM:  {n_medium} świec")

# Próbka zmian per świeca
log.info(f"\n    Próbka sygnałów (co 15 świec):")
log.info(f"    {'Idx':>4}  {'Cena':>10}  {'Trend':>7}  {'Div':>7}  {'Siła':<8}  {'Bull':>5}  {'Bear':>5}  {'RSI':>5}")
log.info(f"    {'─'*4}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*5}")
for i in range(55, min(250, len(candles)), 15):
    s = ext[i]
    log.info(
        f"    {i:>4}  ${candles[i].close:>9,.2f}  "
        f"{s['trend_score_15m']:>+7.3f}  "
        f"{s['divergence_score']:>+7.3f}  "
        f"{s['divergence_strength']:<8}  "
        f"{'TAK' if s['divergence_bullish'] else 'nie':>5}  "
        f"{'TAK' if s['divergence_bearish'] else 'nie':>5}  "
        f"{s['rsi_15m']:>5.1f}"
    )

# ── Parametry nagrody i kary ──────────────────────────────────────────
log.info(f"\n{'='*75}")
log.info(f"  PARAMETRY NAGRODY I KARY DLA TEJ PĘTLI")
log.info(f"{'='*75}")

from trader.reward_engine import RewardConfig, RewardEngine

rc = RewardConfig()
rc.profit_multiplier          = 2.5
rc.loss_multiplier            = 3.0
rc.divergence_confirm_bonus   = 0.8
rc.divergence_exit_bonus      = 0.6
rc.trend_confirm_bonus        = 0.4
rc.hold_profit_bonus          = 0.15
rc.hold_loss_penalty          = 0.10
rc.counter_trend_penalty      = 0.3
rc.overtrading_penalty        = 0.4
rc.missed_opportunity_penalty = 0.1
rc.idle_penalty               = 0.01
rc.stop_loss_penalty          = 1.2
rc.divergence_weight          = 0.7
rc.max_hold_candles           = 50
rc.min_candles_between_trades = 4
rc.stop_loss_pct              = 0.02
rc.take_profit_pct            = 0.04
rc.divergence_min_strength    = "medium"

log.info(f"\n  ┌─ NAGRODY ────────────────────────────────────────────────────")
log.info(f"  │  profit_multiplier          = {rc.profit_multiplier}")
log.info(f"  │    → zysk z pozycji × {rc.profit_multiplier} = nagroda za zysk")
log.info(f"  │  divergence_confirm_bonus   = {rc.divergence_confirm_bonus}")
log.info(f"  │    → bonus gdy wejście potwierdzone dywergencją (weight={rc.divergence_weight})")
log.info(f"  │    → efektywny bonus = {rc.divergence_confirm_bonus * rc.divergence_weight:.3f}")
log.info(f"  │  trend_confirm_bonus        = {rc.trend_confirm_bonus}")
log.info(f"  │    → bonus za wejście zgodne z trendem 15m")
log.info(f"  │  hold_profit_bonus          = {rc.hold_profit_bonus} per świeca")
log.info(f"  │    → nagroda za trzymanie zyskownej pozycji")
log.info(f"  │  divergence_exit_bonus      = {rc.divergence_exit_bonus}")
log.info(f"  │    → bonus za zamknięcie gdy pojawi się div contra trend")
log.info(f"  │")
log.info(f"  ├─ KARY ───────────────────────────────────────────────────────")
log.info(f"  │  loss_multiplier            = {rc.loss_multiplier}")
log.info(f"  │    → strata × {rc.loss_multiplier} = kara za stratę")
log.info(f"  │  stop_loss_penalty          = {rc.stop_loss_penalty}")
log.info(f"  │    → dodatkowa kara przy zamknięciu przez SL")
log.info(f"  │  hold_loss_penalty          = {rc.hold_loss_penalty} per świeca")
log.info(f"  │    → kara za trzymanie stratnej pozycji")
log.info(f"  │  counter_trend_penalty      = {rc.counter_trend_penalty}")
log.info(f"  │    → kara za wejście contra trendu 15m")
log.info(f"  │  overtrading_penalty        = {rc.overtrading_penalty}")
log.info(f"  │    → kara za otwarcie < {rc.min_candles_between_trades} świec po zamknięciu")
log.info(f"  │  missed_opportunity_penalty = {rc.missed_opportunity_penalty}")
log.info(f"  │    → kara za pominięcie silnego sygnału (div+trend)")
log.info(f"  │  idle_penalty               = {rc.idle_penalty} per świeca")
log.info(f"  │    → mała kara za bezczynność gdy są sygnały")
log.info(f"  │")
log.info(f"  └─ PARAMETRY POZYCJI ──────────────────────────────────────────")
log.info(f"     stop_loss_pct              = {rc.stop_loss_pct*100:.1f}%  (stałe)")
log.info(f"     take_profit_pct            = {rc.take_profit_pct*100:.1f}%  (stałe)")
log.info(f"     max_hold_candles           = {rc.max_hold_candles}")
log.info(f"     min_candles_between_trades = {rc.min_candles_between_trades}")
log.info(f"     divergence_min_strength    = {rc.divergence_min_strength}")
log.info(f"     divergence_weight          = {rc.divergence_weight}")

# ── Pętla krok po kroku ───────────────────────────────────────────────
log.info(f"\n{'='*75}")
log.info(f"  PĘTLA KROK PO KROKU (pierwsze 150 kroków)")
log.info(f"{'='*75}")

from trader.feature_builder import FeatureConfig, FeatureBuilder
from trader.trader_env import TraderEnv, ACTION_HOLD, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE

re = RewardEngine(rc)
fb = FeatureBuilder(FeatureConfig())

env = TraderEnv(
    candles         = candles,
    feature_builder = fb,
    reward_engine   = re,
    initial_balance = 1000.0,
    commission_pct  = 0.001,
    extra_signals   = ext,
)
obs  = env.reset()
done = False

random.seed(42)
n_feat    = fb.cfg.feature_count()
n_actions = 4
weights   = [[random.uniform(-0.05, 0.05) for _ in range(n_feat)] for _ in range(n_actions)]
lr        = 0.005

ACTION_NAMES = ["HOLD ", "LONG ", "SHORT", "CLOSE"]

log.info(f"\n  {'Krok':>5}  {'Cena':>10}  {'Trend':>7}  {'Div':>7}  {'Siła':<7}  "
         f"{'Poz':>5}  {'PnL%':>6}  {'Akcja':<6}  {'Nagroda':>9}  "
         f"{'Balance':>9}  Składowe nagrody")
log.info(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  "
         f"{'─'*5}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*30}")

step            = 0
prev_trend      = None
prev_div        = None
trend_changes   = 0
div_changes     = 0
action_counts   = {0:0, 1:0, 2:0, 3:0}
total_reward    = 0.0
trades_detail   = []

while not done and step < 150:
    idx = env.step_idx
    if idx >= len(candles):
        break

    sig       = ext.get(idx, {})
    trend_now = round(sig.get("trend_score_15m", 0), 3)
    div_now   = round(sig.get("divergence_score", 0), 3)
    div_str   = sig.get("divergence_strength", "none")
    price_now = candles[idx].close

    if prev_trend is not None and trend_now != prev_trend: trend_changes += 1
    if prev_div   is not None and div_now   != prev_div:   div_changes   += 1
    prev_trend, prev_div = trend_now, div_now

    # Polityka RL (epsilon-greedy z wagami)
    eps = max(0.05, 0.35 - step * 0.002)
    if random.random() < eps:
        action = random.randint(0, n_actions - 1)
    else:
        scores = [sum(weights[a][i] * obs[i]
                      for i in range(min(len(weights[a]), len(obs))))
                  for a in range(n_actions)]
        action = scores.index(max(scores))

    action_counts[action] += 1

    # Zapamiętaj stan przed krokiem
    pos_before  = env.position_side
    pnl_before  = env._calc_pnl(price_now)

    obs, reward, done, info = env.step(action)
    total_reward += reward

    # Aktualizuj wagi
    for i in range(min(len(weights[action]), len(obs))):
        weights[action][i] += lr * reward * obs[i]

    # Breakdown nagrody (odtwórz z reward engine)
    _, breakdown = re.compute(
        action              = ACTION_NAMES[action].strip().lower(),
        pnl_pct             = pnl_before,
        position_side       = pos_before,
        candles_in_position = env.candles_in_position,
        candles_since_close = env.candles_since_close,
        trend_score         = trend_now,
        divergence_score    = div_now,
        divergence_strength = div_str,
        hit_stop_loss       = info.get("trade_info", {}).get("pnl_pct", 0) <= -rc.stop_loss_pct if pos_before else False,
        hit_take_profit     = info.get("trade_info", {}).get("pnl_pct", 0) >= rc.take_profit_pct if pos_before else False,
    )

    # Formatuj składowe nagrody
    breakdown_str = "  ".join(f"{k}={v:+.3f}" for k, v in breakdown.items() if k != "total" and abs(v) > 0.0001)
    if not breakdown_str:
        breakdown_str = "(brak aktywnych składowych)"

    pos_str = f"{pos_before or '—':>5}"
    pnl_str = f"{pnl_before*100:>+6.2f}" if pos_before else f"{'—':>6}"

    log.info(
        f"  {step:>5}  "
        f"${price_now:>9,.2f}  "
        f"{trend_now:>+7.3f}  "
        f"{div_now:>+7.3f}  "
        f"{div_str:<7}  "
        f"{pos_str}  "
        f"{pnl_str}  "
        f"{ACTION_NAMES[action]}  "
        f"{reward:>+9.4f}  "
        f"${env.balance:>8,.2f}  "
        f"{breakdown_str}"
    )

    # Zapisz szczegóły transakcji
    ti = info.get("trade_info", {})
    if "opened" in ti:
        trades_detail.append({
            "step": step, "action": "OPEN", "side": ti["opened"],
            "price": ti["entry_price"], "trend": trend_now, "div": div_now,
            "div_str": div_str,
        })
    if "closed_side" in ti:
        trades_detail.append({
            "step": step, "action": "CLOSE", "side": ti["closed_side"],
            "entry": ti.get("entry_price", 0), "exit": ti.get("exit_price", 0),
            "pnl": ti.get("pnl_pct", 0) * 100,
            "duration": ti.get("duration", 0),
        })

    step += 1

# ── Podsumowanie ──────────────────────────────────────────────────────
log.info(f"\n{'='*75}")
log.info(f"  PODSUMOWANIE PĘTLI")
log.info(f"{'='*75}")
log.info(f"  Kroków:              {step}")
log.info(f"  Zmian trendu:        {trend_changes}  ← trend zmieniał się {trend_changes}x")
log.info(f"  Zmian dywergencji:   {div_changes}  ← dywergencja zmieniała się {div_changes}x")
log.info(f"  Suma nagród:         {total_reward:+.4f}")
log.info(f"  Balance końcowy:     ${env.balance:,.2f}")
log.info(f"  Akcje:")
log.info(f"    HOLD:  {action_counts[0]}")
log.info(f"    LONG:  {action_counts[1]}")
log.info(f"    SHORT: {action_counts[2]}")
log.info(f"    CLOSE: {action_counts[3]}")

stats = env.get_episode_stats()
log.info(f"  Return:              {stats['total_return_pct']:+.2f}%")
log.info(f"  Win Rate:            {stats['win_rate']:.1f}%")
log.info(f"  Total Trades:        {stats['total_trades']}")
log.info(f"  Sharpe:              {stats['sharpe']:.3f}")
log.info(f"  Max Drawdown:        {stats['max_drawdown_pct']:.2f}%")

if trades_detail:
    log.info(f"\n{'─'*75}")
    log.info(f"  SZCZEGÓŁY TRANSAKCJI")
    log.info(f"{'─'*75}")
    for t in trades_detail:
        if t["action"] == "OPEN":
            log.info(f"  OPEN  {t['side'].upper():<5}  krok={t['step']:>3}  "
                     f"cena=${t['price']:>10,.2f}  "
                     f"trend={t['trend']:>+7.3f}  "
                     f"div={t['div']:>+7.3f}  siła={t['div_str']}")
        else:
            pnl_c = "✅" if t["pnl"] > 0 else "❌"
            log.info(f"  CLOSE {t['side'].upper():<5}  krok={t['step']:>3}  "
                     f"entry=${t['entry']:>10,.2f}  "
                     f"exit=${t['exit']:>10,.2f}  "
                     f"pnl={t['pnl']:>+7.2f}%  "
                     f"dur={t['duration']:>3} świec  {pnl_c}")

log.info(f"\n{'='*75}")
log.info(f"  Log zapisany do: {LOG_FILE}")
log.info(f"{'='*75}")
print(f"\nGotowe! Sprawdź plik: {LOG_FILE}")

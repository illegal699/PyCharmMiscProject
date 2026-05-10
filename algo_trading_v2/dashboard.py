"""
dashboard.py - Algo Trading Dashboard z zakładkami
Uruchomienie: streamlit run dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import asyncio, time, threading, json
from datetime import datetime, date
import streamlit as st

st.set_page_config(page_title="Algo Trading", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--text:#e6edf3;--muted:#8b949e;--font-mono:'Space Mono',monospace;--font-body:'Inter',sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin-bottom:16px;}
.card-title{font-family:var(--font-mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}
.card-value{font-family:var(--font-mono);font-size:32px;font-weight:700;line-height:1;}
.card-sub{font-family:var(--font-body);font-size:13px;color:var(--muted);margin-top:6px;}
.bullish{color:var(--green)!important;}.bearish{color:var(--red)!important;}.neutral{color:var(--yellow)!important;}
.strong-bullish{color:#56d364!important;}.strong-bearish{color:#ff7b72!important;}
.section-header{font-family:var(--font-mono);font-size:13px;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:8px;margin-bottom:16px;text-transform:uppercase;letter-spacing:1px;}
.indicator-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #21262d;font-size:13px;}
.indicator-label{font-family:var(--font-mono);color:var(--muted);font-size:11px;}
.indicator-value{font-family:var(--font-mono);font-weight:700;}
.param-group{font-family:var(--font-mono);font-size:11px;color:var(--blue);text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;}
.train-log{background:#010409;border:1px solid var(--border);border-radius:6px;padding:12px 16px;font-family:var(--font-mono);font-size:12px;color:#7ee787;height:220px;overflow-y:auto;white-space:pre-wrap;}
.progress-bar-wrap{background:#21262d;border-radius:4px;height:10px;width:100%;margin:8px 0;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
.stButton button{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:var(--font-mono)!important;font-size:12px!important;border-radius:6px!important;}
.stButton button:hover{border-color:var(--blue)!important;color:var(--blue)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface);border-bottom:1px solid var(--border);gap:0;}
.stTabs [data-baseweb="tab"]{font-family:var(--font-mono)!important;font-size:12px!important;color:var(--muted)!important;background:transparent!important;border:none!important;padding:10px 20px!important;}
.stTabs [aria-selected="true"]{color:var(--text)!important;border-bottom:2px solid var(--blue)!important;}
</style>""", unsafe_allow_html=True)

# helpers
def dc(d): return {"strongly_bullish":"strong-bullish","bullish":"bullish","neutral":"neutral","bearish":"bearish","strongly_bearish":"strong-bearish","strong_up":"strong-bullish","up":"bullish","balance":"neutral","down":"bearish","strong_down":"strong-bearish"}.get(d,"neutral")
def de(d): return {"strongly_bullish":"🟢🟢","bullish":"🟢","neutral":"⚪","bearish":"🔴","strongly_bearish":"🔴🔴","strong_up":"🟢🟢","up":"🟢","balance":"⚪","down":"🔴","strong_down":"🔴🔴"}.get(d,"❓")
def sbar(score):
    pct=((score+1)/2)*100; color="#3fb950" if score>=0 else "#f85149"
    if score>=0: s=f"position:absolute;height:100%;background:{color};border-radius:4px;left:50%;width:{abs(pct-50):.1f}%;"
    else: s=f"position:absolute;height:100%;background:{color};border-radius:4px;left:{min(50,pct):.1f}%;width:{abs(pct-50):.1f}%;"
    return f'<div style="background:#21262d;border-radius:4px;height:8px;width:100%;margin:8px 0;position:relative;"><div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#30363d;"></div><div style="{s}"></div></div>'
def cbar(c):
    color="#3fb950" if c>0.65 else "#d29922" if c>0.40 else "#f85149"
    return f'<div style="background:#21262d;border-radius:4px;height:6px;width:100%;margin:4px 0;"><div style="width:{c*100:.0f}%;height:100%;background:{color};border-radius:4px;"></div></div>'
def run_async(coro):
    loop=asyncio.new_event_loop()
    try: return loop.run_until_complete(coro)
    finally: loop.close()

# session state
for k,v in [("training_running",False),("training_progress",None),("trainer_instance",None)]:
    if k not in st.session_state: st.session_state[k]=v
if "reward_config" not in st.session_state:
    from trader.reward_engine import RewardConfig
    st.session_state.reward_config=RewardConfig()
if "feature_config" not in st.session_state:
    from trader.feature_builder import FeatureConfig
    st.session_state.feature_config=FeatureConfig()
if "meta_running" not in st.session_state:
    st.session_state.meta_running=False
if "meta_progress" not in st.session_state:
    st.session_state.meta_progress=None
if "meta_instance" not in st.session_state:
    st.session_state.meta_instance=None
if "meta_hitl_waiting" not in st.session_state:
    st.session_state.meta_hitl_waiting=False
if "meta_hitl_data" not in st.session_state:
    st.session_state.meta_hitl_data={}

# sidebar
with st.sidebar:
    st.markdown("### ⚙️ Konfiguracja"); st.markdown("---")
    PAIRS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT"]
    symbol=st.selectbox("Para walutowa",PAIRS,index=0,key="sb_symbol")
    interval=st.selectbox("Interwał",["1m","5m","15m","1h"],index=1,key="sb_interval")
    st.markdown("---")
    use_sentiment=st.checkbox("Algorytm #1 — Sentyment",value=True)
    use_trend=st.checkbox("Algorytm #2 — Trend",value=True)
    st.markdown("---")
    auto_refresh=st.checkbox("Auto-odświeżanie",value=False,key="sb_autorefresh")
    refresh_sec=st.slider("Co ile sekund",30,300,60,step=30,disabled=not auto_refresh)
    refresh_btn=st.button("🔄  Odśwież",use_container_width=True)
    st.markdown("---")
    st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:10px;color:#8b949e;">{symbol} · {interval}<br>{datetime.now().strftime("%H:%M:%S")}</div>',unsafe_allow_html=True)

st.markdown(f'<h1 style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;margin-bottom:4px;">📈 ALGO TRADING DASHBOARD</h1><p style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-bottom:16px;">{symbol} · {interval} · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',unsafe_allow_html=True)

tab_monitor,tab_rewards,tab_features,tab_training,tab_meta=st.tabs(["📊  Monitor","🎁  Parametry RL","🔧  Dane wejściowe","🎯  Trening","🧠  Meta-Agent"])

# ══ TAB 1: Monitor ══
with tab_monitor:
    @st.cache_data(ttl=30,show_spinner=False)
    def fetch_sentiment():
        try:
            from sentiment_analyzer import SentimentAnalyzer
            a=SentimentAnalyzer(); s=run_async(a.get_signal_async()); a.stop(); return s
        except: return None

    @st.cache_data(ttl=20,show_spinner=False)
    def fetch_trend(sym):
        try:
            from trend_analyzer import TrendAnalyzer
            a=TrendAnalyzer(symbol=sym); s=run_async(a.get_signal_async()); a.stop(); return s
        except: return None

    if refresh_btn: st.cache_data.clear()
    with st.spinner("Pobieranie danych..."):
        s_signal=fetch_sentiment() if use_sentiment else None
        t_signal=fetch_trend(symbol) if use_trend else None

    cl,cr=st.columns([1,1],gap="large")

    with cl:
        st.markdown("<div class='section-header'>Algorytm #1 — Sentyment</div>",unsafe_allow_html=True)
        if not use_sentiment: st.info("Algorytm #1 wyłączony")
        elif not s_signal: st.error("Brak danych sentymentu")
        else:
            d=s_signal.direction.value
            st.markdown(f'<div class="card"><div class="card-title">Composite Score</div><div class="card-value {dc(d)}">{s_signal.composite_score:+.4f}</div><div class="card-sub">{de(d)} {d.replace("_"," ").title()}</div>{sbar(s_signal.composite_score)}</div>',unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            with m1:
                qc={"high":"green","medium":"yellow","low":"red"}.get(s_signal.quality.value,"muted")
                st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Quality</div><div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:var(--{qc});">{s_signal.quality.value.upper()}</div></div>',unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Confidence</div><div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;">{s_signal.confidence:.0%}</div>{cbar(s_signal.confidence)}</div>',unsafe_allow_html=True)
            with m3:
                vc="green" if s_signal.sentiment_velocity>0 else "red" if s_signal.sentiment_velocity<0 else "muted"
                st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Velocity</div><div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:var(--{vc});">{s_signal.sentiment_velocity:+.4f}</div></div>',unsafe_allow_html=True)
            rows=""
            for src,lbl in [(s_signal.fear_greed_signal,"Fear & Greed"),(s_signal.trends_signal,"Google Trends")]:
                if src:
                    sc2=dc("bullish" if src.score>0.2 else "bearish" if src.score<-0.2 else "neutral")
                    meta=" · ".join(f"{k}={v}" for k,v in list(src.raw_metadata.items())[:2])
                    rows+=f'<div class="indicator-row"><span class="indicator-label">{lbl}</span><span class="indicator-value {sc2}">{src.score:+.4f}</span><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">{meta}</span></div>'
            st.markdown(f'<div class="card">{rows}</div>',unsafe_allow_html=True)

    with cr:
        st.markdown("<div class='section-header'>Algorytm #2 — Trend</div>",unsafe_allow_html=True)
        if not use_trend: st.info("Algorytm #2 wyłączony")
        elif not t_signal: st.error("Brak danych trendu")
        else:
            d=t_signal.primary_direction.value
            st.markdown(f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:flex-start;"><div><div class="card-title">Kierunek (15m)</div><div class="card-value {dc(d)}">{d.replace("_"," ").title()}</div><div class="card-sub">{de(d)} Score: {t_signal.score:+.4f}</div>{sbar(t_signal.score)}</div><div style="text-align:right;"><div class="card-title">Cena</div><div style="font-family:Space Mono,monospace;font-size:24px;font-weight:700;">${t_signal.current_price:,.2f}</div></div></div></div>',unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            with m1:
                sc3="green" if t_signal.suggested_side=="long" else "red" if t_signal.suggested_side=="short" else "muted"
                st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Strona</div><div style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;color:var(--{sc3});">{(t_signal.suggested_side or "—").upper()}</div></div>',unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Confidence</div><div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;">{t_signal.confidence:.0%}</div>{cbar(t_signal.confidence)}</div>',unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="card" style="padding:14px;"><div class="card-title">Faza</div><div style="font-family:Space Mono,monospace;font-size:13px;font-weight:700;color:#58a6ff;">{t_signal.market_phase.value.upper()}</div></div>',unsafe_allow_html=True)

            # TF alignment
            ac="#3fb950" if t_signal.tf_alignment>0.7 else "#d29922" if t_signal.tf_alignment>0.4 else "#f85149"
            tf_rows=""
            for tfd,lbl in [(t_signal.tf_15m,"15m"),(t_signal.tf_5m,"5m"),(t_signal.tf_1m,"1m")]:
                arr="↑" if tfd.score>0.1 else "↓" if tfd.score<-0.1 else "→"
                tf_rows+=f'<div class="indicator-row"><span class="indicator-label">{lbl}</span><span class="indicator-value {dc(tfd.direction.value)}">{arr} {tfd.score:+.3f}</span><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">RSI {tfd.rsi:.0f} · Vol {tfd.volume_ratio:.1f}x</span></div>'
            st.markdown(f'<div class="card"><div style="display:flex;justify-content:space-between;margin-bottom:12px;"><span style="font-family:Space Mono,monospace;font-size:13px;">{t_signal.tf_alignment_desc}</span><span style="font-family:Space Mono,monospace;font-size:13px;font-weight:700;color:{ac};">{t_signal.tf_alignment:.0%}</span></div>{tf_rows}</div>',unsafe_allow_html=True)

            # Wskaźniki 15m
            tf15=t_signal.tf_15m; rsi_c="#f85149" if tf15.rsi>70 else "#3fb950" if tf15.rsi<30 else "#e6edf3"; macd_c="#3fb950" if tf15.macd_hist>0 else "#f85149"
            st.markdown(f'<div class="card"><div class="indicator-row"><span class="indicator-label">EMA 9/21/50</span><span class="indicator-value">{tf15.ema_fast:,.1f}/{tf15.ema_slow:,.1f}/{tf15.ema_trend:,.1f}</span></div><div class="indicator-row"><span class="indicator-label">RSI (14)</span><span class="indicator-value" style="color:{rsi_c};">{tf15.rsi:.1f}</span></div><div class="indicator-row"><span class="indicator-label">MACD Hist</span><span class="indicator-value" style="color:{macd_c};">{tf15.macd_hist:+.6f}</span></div><div class="indicator-row"><span class="indicator-label">Wolumen</span><span class="indicator-value">{tf15.volume_trend} · {tf15.volume_ratio:.2f}x</span></div><div class="indicator-row" style="border:none;"><span class="indicator-label">Swing H/L</span><span class="indicator-value">{tf15.last_high:,.2f}/{tf15.last_low:,.2f}</span></div></div>',unsafe_allow_html=True)

            # Dywergencje
            div_5m=t_signal.divergences_5m; div_15m=t_signal.divergences_15m; div_score=t_signal.divergence_score
            has_any=(div_5m and div_5m.divergences) or (div_15m and div_15m.divergences)
            if not has_any:
                st.markdown('<div class="card" style="padding:14px 20px;"><span style="font-family:Space Mono,monospace;font-size:12px;color:#8b949e;">✓ Brak dywergencji</span></div>',unsafe_allow_html=True)
            else:
                div_color="#3fb950" if div_score>0.1 else "#f85149" if div_score<-0.1 else "#8b949e"
                div_rows=""
                for lbl2,dr in [("5m",div_5m),("15m",div_15m)]:
                    if not dr or not dr.divergences: continue
                    for d2 in dr.divergences:
                        col2="#3fb950" if d2.type=="bullish" else "#f85149"; emj="🟢" if d2.type=="bullish" else "🔴"
                        sc4={"strong":"#e6edf3","medium":"#d29922","weak":"#8b949e"}.get(d2.strength,"#8b949e")
                        div_rows+=f'<div style="padding:10px 0;border-bottom:1px solid #21262d;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-family:Space Mono,monospace;font-size:12px;font-weight:700;color:{col2};">{emj} {d2.type.upper()} · {d2.indicator.upper()} · {lbl2}</span><span style="font-family:Space Mono,monospace;font-size:11px;color:{sc4};">{d2.strength.upper()} · {d2.confidence:.0%}</span></div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">Cena: {d2.price_desc}</div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">Wsk.: {d2.indic_desc}</div></div>'
                st.markdown(f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;"><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">DIV SCORE</span><span style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:{div_color};">{div_score:+.4f}</span></div>{div_rows}</div>',unsafe_allow_html=True)

    # Ocena łączna
    st.markdown("---")
    st.markdown("<div class='section-header'>Ocena łączna</div>",unsafe_allow_html=True)
    if t_signal:
        side=t_signal.suggested_side or "—"
        if t_signal.is_tradeable and t_signal.skip_reason!="divergence_warning":
            status="GO"; sc5="#3fb950" if side=="long" else "#f85149"; bg="#0d2f1a" if side=="long" else "#2f0d0d"; brd="#3fb950" if side=="long" else "#f85149"; msg=f"Trend {t_signal.strength.value} · {t_signal.tf_alignment_desc}"
        elif t_signal.is_tradeable and t_signal.skip_reason=="divergence_warning":
            status="GO"; sc5="#3fb950" if side=="long" else "#f85149"; bg="#0d2f1a" if side=="long" else "#2f0d0d"; brd="#3fb950" if side=="long" else "#f85149"; msg=f"Trend {t_signal.strength.value} · ⚠️ dywergencja contra trend"
        else:
            status="SKIP"; side="—"; sc5="#f85149"; bg="#2f0d0d"; brd="#f85149"; msg=f"Trend: {t_signal.skip_reason}"
        sent_ctx=""
        if s_signal and use_sentiment:
            sc6="#3fb950" if s_signal.composite_score>0.2 else "#f85149" if s_signal.composite_score<-0.2 else "#8b949e"
            sent_ctx=f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #30363d;"><span style="font-family:Space Mono,monospace;font-size:10px;color:#8b949e;">SENTYMENT (kontekst)</span><span style="font-family:Space Mono,monospace;font-size:12px;font-weight:700;color:{sc6};margin-left:12px;">{s_signal.direction.value.replace("_"," ").upper()} {s_signal.composite_score:+.3f}</span></div>'
        st.markdown(f'<div style="background:{bg};border:1px solid {brd};border-radius:8px;padding:20px 28px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-bottom:4px;">STATUS</div><div style="font-family:Space Mono,monospace;font-size:28px;font-weight:700;color:{brd};">{status}</div><div style="font-family:Inter,sans-serif;font-size:13px;color:#8b949e;margin-top:4px;">{msg}</div></div><div style="text-align:right;"><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-bottom:4px;">KIERUNEK</div><div style="font-family:Space Mono,monospace;font-size:28px;font-weight:700;color:{sc5};">{side.upper()}</div></div></div>{sent_ctx}</div>',unsafe_allow_html=True)
    else: st.error("Brak danych trendu.")

    if auto_refresh: time.sleep(refresh_sec); st.cache_data.clear(); st.rerun()

# ══ TAB 2: Parametry RL ══
with tab_rewards:
    st.markdown("<div class='section-header'>Parametry nagród i kar — Algorytm #3</div>",unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:13px;color:#8b949e;margin-bottom:20px;">Ustaw wagi nagród i kar dla agenta RL.</p>',unsafe_allow_html=True)
    rc=st.session_state.reward_config
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="param-group">💰 Nagrody za zysk</div>',unsafe_allow_html=True)
        rc.profit_multiplier=st.slider("Mnożnik zysku",0.5,5.0,float(rc.profit_multiplier),0.1,help="Jak bardzo agent jest nagradzany za zysk.")
        rc.divergence_confirm_bonus=st.slider("Bonus — dywergencja potwierdza",0.0,2.0,float(rc.divergence_confirm_bonus),0.05,help="Nagroda gdy wejście potwierdzone dywergencją.")
        rc.trend_confirm_bonus=st.slider("Bonus — trend potwierdza",0.0,1.0,float(rc.trend_confirm_bonus),0.05,help="Nagroda za wejście zgodne z trendem 15m.")
        rc.hold_profit_bonus=st.slider("Bonus za trzymanie zysku (per świeca)",0.0,0.5,float(rc.hold_profit_bonus),0.01,help="Premiuje cierpliwość w zyskownej pozycji.")
        rc.divergence_exit_bonus=st.slider("Bonus — zamknięcie na dywergencji",0.0,2.0,float(rc.divergence_exit_bonus),0.05,help="Nagroda za zamknięcie gdy pojawi się dywergencja contra trend.")
        rc.divergence_weight=st.slider("Waga dywergencji w decyzji",0.0,1.0,float(rc.divergence_weight),0.05,help="0 = ignoruj dywergencje, 1 = decydujące.")
        st.markdown('<div class="param-group">📍 Parametry pozycji</div>',unsafe_allow_html=True)
        sl_pct=st.slider("Stop Loss (%)",0.3,5.0,float(rc.stop_loss_pct*100),0.1,help="Automatyczne zamknięcie ze stratą.")/100; rc.stop_loss_pct=sl_pct
        tp_pct=st.slider("Take Profit (%)",0.5,100.0,float(min(rc.take_profit_pct*100,100.0)),0.5,help="Automatyczne zamknięcie z zyskiem.")/100; rc.take_profit_pct=tp_pct
        rc.max_hold_candles=st.slider("Maks. świece w pozycji",5,200,int(rc.max_hold_candles),5,help="Po ilu świecach kara za przetrzymanie.")
        rc.min_candles_between_trades=st.slider("Min. przerwa między transakcjami",1,20,int(rc.min_candles_between_trades),1,help="Zapobiega overtradingowi.")
    with c2:
        st.markdown('<div class="param-group">⚠️ Kary za straty</div>',unsafe_allow_html=True)
        rc.loss_multiplier=st.slider("Mnożnik straty",0.5,5.0,float(rc.loss_multiplier),0.1,help="Wyższy niż profit = agent bardziej unika strat.")
        rc.stop_loss_penalty=st.slider("Kara — stop loss",0.0,3.0,float(rc.stop_loss_penalty),0.1,help="Dodatkowa kara za zamknięcie przez SL.")
        rc.hold_loss_penalty=st.slider("Kara za trzymanie straty (per świeca)",0.0,0.5,float(rc.hold_loss_penalty),0.01,help="Uczy szybkiego cięcia strat.")
        rc.counter_trend_penalty=st.slider("Kara — wejście contra trend",0.0,1.0,float(rc.counter_trend_penalty),0.05,help="Kara za wejście przeciw trendowi 15m.")
        rc.overtrading_penalty=st.slider("Kara — overtrading",0.0,1.0,float(rc.overtrading_penalty),0.05,help="Kara za zbyt szybkie ponowne wejście.")
        st.markdown('<div class="param-group">😴 Kary za bezczynność</div>',unsafe_allow_html=True)
        rc.missed_opportunity_penalty=st.slider("Kara — pominięty sygnał",0.0,1.0,float(rc.missed_opportunity_penalty),0.01,help="Kara za nieotwarcie pozycji przy silnym sygnale.")
        rc.idle_penalty=st.slider("Kara — bezczynność (per świeca)",0.0,0.1,float(rc.idle_penalty),0.005,help="Mała kara gdy są sygnały a agent nic nie robi.")
        st.markdown('<div class="param-group">🔀 Dywergencja</div>',unsafe_allow_html=True)
        rc.divergence_min_strength=st.selectbox("Minimalna siła dywergencji",["medium","strong"],index=0 if rc.divergence_min_strength=="medium" else 1,help="Słabsze dywergencje są ignorowane.",key="rc_div_strength")
    st.session_state.reward_config=rc
    cs1,cs2,_=st.columns([1,1,3])
    with cs1:
        if st.button("💾 Zapisz",use_container_width=True):
            os.makedirs("configs",exist_ok=True)
            with open("configs/reward_config.json","w") as f: json.dump(rc.to_dict(),f,indent=2)
            st.success("Zapisano!")
    with cs2:
        if st.button("↺ Reset",use_container_width=True):
            from trader.reward_engine import RewardConfig
            st.session_state.reward_config=RewardConfig(); st.rerun()

# ══ TAB 3: Dane wejściowe ══
with tab_features:
    st.markdown("<div class='section-header'>Dane wejściowe — Algorytm #3</div>",unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:13px;color:#8b949e;margin-bottom:20px;">Wybierz cechy które agent uwzględnia podczas uczenia.</p>',unsafe_allow_html=True)
    from trader.feature_builder import FEATURE_GROUPS,FEATURE_DESCRIPTIONS
    fc=st.session_state.feature_config
    st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:12px;color:#58a6ff;margin-bottom:16px;">Aktywne cechy: {fc.feature_count()}</div>',unsafe_allow_html=True)
    for grp,keys in FEATURE_GROUPS.items():
        st.markdown(f'<div class="param-group">{grp}</div>',unsafe_allow_html=True)
        cols=st.columns(2)
        for i,key in enumerate(keys):
            lbl,desc=FEATURE_DESCRIPTIONS.get(key,(key,""))
            with cols[i%2]:
                setattr(fc,key,st.checkbox(f"**{lbl}**",value=getattr(fc,key,False),key=f"feat_{key}",help=desc))
    st.session_state.feature_config=fc
    cf1,cf2,cf3,_=st.columns([1,1,1,2])
    with cf1:
        if st.button("💾 Zapisz cechy",use_container_width=True):
            os.makedirs("configs",exist_ok=True)
            with open("configs/feature_config.json","w") as f: json.dump(fc.to_dict(),f,indent=2)
            st.success("Zapisano!")
    with cf2:
        if st.button("✅ Zaznacz wszystkie",use_container_width=True):
            for k in FEATURE_DESCRIPTIONS: setattr(st.session_state.feature_config,k,True)
            st.rerun()
    with cf3:
        if st.button("☐ Odznacz wszystkie",use_container_width=True):
            for k in FEATURE_DESCRIPTIONS: setattr(st.session_state.feature_config,k,False)
            st.rerun()

# ══ TAB 4: Trening ══
with tab_training:
    st.markdown("<div class='section-header'>Trening — Algorytm #3</div>",unsafe_allow_html=True)
    cc1,cc2=st.columns([1,1],gap="large")

    with cc1:
        st.markdown('<div class="param-group">📋 Parametry treningu</div>',unsafe_allow_html=True)
        t_sym=st.selectbox("Para walutowa (trening)",["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"],key="t_sym")
        t_iv=st.selectbox("Interwał (trening)",["1m","5m","15m"],index=1,key="t_iv")
        td1,td2=st.columns(2)
        with td1: t_df=st.date_input("Data od",value=date(2024,1,1),key="t_df")
        with td2: t_dt=st.date_input("Data do",value=date(2024,12,31),key="t_dt")
        t_ep=st.slider("Liczba epizodów",10,500,100,10,key="t_ep",help="Ile razy agent przechodzi przez dane historyczne.")
        t_bal=st.number_input("Kapitał startowy ($)",100.0,100000.0,1000.0,100.0,key="t_bal")
        t_com=st.slider("Prowizja (%)",0.01,0.5,0.1,0.01,key="t_com")/100
        t_name=st.text_input("Nazwa modelu","trader_ppo",key="t_name")
        st.markdown(f'<div class="card" style="padding:14px 18px;"><div class="card-title">Podsumowanie</div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;line-height:1.8;">{t_sym} · {t_iv}<br>{t_df} → {t_dt}<br>Epizody: {t_ep} · Kapitał: ${t_bal:,.0f}<br>Cechy: {st.session_state.feature_config.feature_count()} · Prowizja: {t_com*100:.2f}%</div></div>',unsafe_allow_html=True)
        tb1,tb2=st.columns(2)
        with tb1: start_btn=st.button("▶  Rozpocznij uczenie",use_container_width=True,disabled=st.session_state.training_running,type="primary")
        with tb2: stop_btn=st.button("⏹  Zatrzymaj",use_container_width=True,disabled=not st.session_state.training_running)

    with cc2:
        st.markdown('<div class="param-group">📈 Postęp treningu</div>',unsafe_allow_html=True)
        prog=st.session_state.training_progress
        pct=prog.pct_complete if prog else 0.0
        bc="#3fb950" if pct>=100 else "#58a6ff"
        ep_now=prog.episode if prog else 0
        st.markdown(f'<div class="card" style="padding:14px 18px;"><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">POSTĘP</span><span style="font-family:Space Mono,monospace;font-size:13px;font-weight:700;color:{bc};">{pct:.0f}%</span></div><div class="progress-bar-wrap"><div style="width:{pct:.0f}%;height:100%;background:{bc};border-radius:4px;"></div></div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-top:6px;">Epizod: {ep_now} / {prog.total_episodes if prog else t_ep}</div></div>',unsafe_allow_html=True)

        if prog and prog.episode>0:
            pm1,pm2,pm3,pm4=st.columns(4)
            rc2="#3fb950" if (prog.current_return or 0)>=0 else "#f85149"
            with pm1: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Return</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:{rc2};">{prog.current_return:+.1f}%</div></div>',unsafe_allow_html=True)
            with pm2: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Best</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:#3fb950;">{prog.best_return:+.1f}%</div></div>',unsafe_allow_html=True)
            with pm3: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Win Rate</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;">{prog.win_rate:.0f}%</div></div>',unsafe_allow_html=True)
            with pm4: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Sharpe</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;">{prog.sharpe:.2f}</div></div>',unsafe_allow_html=True)
            pm5,pm6=st.columns(2)
            with pm5: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Avg Return (10ep)</div><div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;">{prog.avg_return_10:+.1f}%</div></div>',unsafe_allow_html=True)
            with pm6: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Max Drawdown</div><div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;color:#f85149;">{prog.max_drawdown:.1f}%</div></div>',unsafe_allow_html=True)

        st.markdown('<div class="param-group" style="margin-top:8px;">📋 Log treningu</div>',unsafe_allow_html=True)
        log_lines=[]
        if prog:
            for h in (prog.history[-15:] if prog.history else []):
                sign="+" if h["return_pct"]>=0 else ""
                log_lines.append(f"[Ep {h['episode']:>4}] Return:{sign}{h['return_pct']:.2f}%  WR:{h['win_rate']:.0f}%  Trades:{h['trades']:>3}  Sharpe:{h['sharpe']:.2f}")
            if prog.message: log_lines.append(f"\n>>> {prog.message}")
        log_text="\n".join(log_lines) if log_lines else "Oczekiwanie na start treningu..."
        st.markdown(f'<div class="train-log">{log_text}</div>',unsafe_allow_html=True)

        if prog and prog.elapsed_sec>0:
            st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-top:8px;">Czas: {prog.elapsed_sec:.0f}s · ETA: {prog.eta_sec:.0f}s</div>',unsafe_allow_html=True)

    if start_btn and not st.session_state.training_running:
        from trader.trainer import Trainer, TrainingConfig
        tc_obj = TrainingConfig(
            symbol=t_sym, interval=t_iv,
            date_from=str(t_df), date_to=str(t_dt),
            n_episodes=t_ep, initial_balance=t_bal,
            commission_pct=t_com, model_name=t_name, save_path="models/",
        )
        def on_progress(p):
            st.session_state.training_progress = p

        trainer = Trainer(
            training_config   = tc_obj,
            reward_config     = st.session_state.reward_config,
            feature_config    = st.session_state.feature_config,
            progress_callback = on_progress,
        )
        st.session_state.trainer_instance  = trainer
        st.session_state.training_running  = True
        st.session_state.training_progress = trainer.progress

        def run_t():
            try:
                trainer.run()
            finally:
                # Zawsze odblokuj przycisk — nawet przy błędzie lub zatrzymaniu
                st.session_state.training_running = False

        threading.Thread(target=run_t, daemon=True).start()
        st.rerun()

    if stop_btn and st.session_state.training_running:
        if st.session_state.trainer_instance:
            st.session_state.trainer_instance.stop()
        # Nie czekaj na wątek — odblokuj UI natychmiast
        st.session_state.training_running  = False
        st.session_state.trainer_instance  = None
        st.rerun()

    # Przycisk awaryjny — reset jeśli UI się zablokuje
    if not st.session_state.training_running:
        if st.button("🔁  Reset UI (jeśli zablokowany)", use_container_width=False):
            st.session_state.training_running = False
            st.session_state.trainer_instance = None
            st.rerun()

    if st.session_state.training_running:
        time.sleep(2)
        st.rerun()

# ══ TAB 5: Meta-Agent HITL/HOTL ══
with tab_meta:
    # Synchronizuj stan HITL z progress obiektu (może się zmienić w wątku)
    if st.session_state.meta_progress:
        mp_live = st.session_state.meta_progress
        if mp_live.hitl_waiting and not st.session_state.meta_hitl_waiting:
            st.session_state.meta_hitl_waiting = True
        if not mp_live.hitl_waiting and st.session_state.meta_hitl_waiting:
            # HITL zostało obsłużone przez agenta (timeout lub inne)
            pass

    st.markdown("<div class='section-header'>Meta-Agent — HITL/HOTL</div>",unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:13px;color:#8b949e;margin-bottom:20px;">Optuna automatycznie stroi parametry (HOTL). Co X epizodów zatrzymuje się i pyta Ciebie o decyzję (HITL).</p>',unsafe_allow_html=True)

    ma_c1, ma_c2 = st.columns([1,1], gap="large")

    with ma_c1:
        st.markdown('<div class="param-group">📋 Konfiguracja Meta-Agenta</div>',unsafe_allow_html=True)

        ma_sym   = st.selectbox("Para walutowa",["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"],key="ma_sym")
        ma_iv    = st.selectbox("Interwał",["1m","5m","15m"],index=1,key="ma_iv")
        ma_d1,ma_d2 = st.columns(2)
        with ma_d1: ma_df = st.date_input("Data od",value=date(2024,1,1),key="ma_df")
        with ma_d2: ma_dt = st.date_input("Data do",value=date(2024,6,30),key="ma_dt")

        st.markdown('<div class="param-group">🔬 Parametry Optuna (HOTL)</div>',unsafe_allow_html=True)
        ma_trials  = st.slider("Liczba prób Optuna",5,200,50,5,key="ma_trials",help="Ile zestawów parametrów Optuna przetestuje.")
        ma_ep_trial= st.slider("Epizodów na próbę",5,50,15,5,key="ma_ep_trial",help="Ile epizodów treningu na każdą próbę Optuna.")

        st.markdown('<div class="param-group">💾 Pamięć sesji Optuna</div>',unsafe_allow_html=True)
        ma_study_name  = st.text_input("Nazwa study",value="trader_optuna",key="ma_study_name",help="Każda unikalna nazwa to osobna sesja. Ta sama nazwa = kontynuacja.")
        ma_continue    = st.checkbox("Kontynuuj poprzednią sesję",value=True,key="ma_continue",help="Jeśli zaznaczone, Optuna wczyta poprzednie próby z bazy i kontynuuje od nich. Odznacz żeby zacząć od zera.")
        ma_storage     = st.text_input("Plik bazy danych",value="sqlite:///checkpoints/optuna_study.db",key="ma_storage",help="Ścieżka do pliku SQLite. Możesz mieć kilka baz dla różnych eksperymentów.")

        # Info o istniejącej sesji
        try:
            import optuna, os
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            storage_path = ma_storage.replace("sqlite:///","")
            if os.path.exists(storage_path):
                study_tmp = optuna.load_study(study_name=ma_study_name, storage=ma_storage)
                n_prev = len(study_tmp.trials)
                try:
                    best_prev = study_tmp.best_value
                    st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#3fb950;margin-top:4px;">✅ Znaleziono {n_prev} poprzednich prób · best score: {best_prev:.4f}</div>',unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#58a6ff;margin-top:4px;">📂 Znaleziono {n_prev} poprzednich prób (brak best)</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-top:4px;">🆕 Nowa sesja — baza zostanie utworzona</div>',unsafe_allow_html=True)
        except Exception:
            pass
        ma_balance = st.number_input("Kapitał startowy ($)",100.0,100000.0,1000.0,100.0,key="ma_bal")
        ma_com     = st.slider("Prowizja (%)",0.01,0.5,0.1,0.01,key="ma_com")/100

        st.markdown('<div class="param-group">🔒 Parametry zablokowane (Optuna ich nie zmienia)</div>',unsafe_allow_html=True)

        # Blokowanie wskaźników technicznych
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-bottom:6px;">Wskaźniki techniczne — Optuna NIE będzie ich zmieniać:</div>',unsafe_allow_html=True)
        ind_c1, ind_c2, ind_c3 = st.columns(3)
        with ind_c1:
            lock_rsi   = st.checkbox("🔒 RSI (1m/5m/15m)", value=False, key="lock_rsi",
                                      help="Zablokuj RSI — zawsze włączone lub wyłączone")
            rsi_val    = st.checkbox("Włącz RSI", value=True, key="lock_rsi_val",
                                      disabled=not lock_rsi)
        with ind_c2:
            lock_macd  = st.checkbox("🔒 MACD", value=False, key="lock_macd",
                                      help="Zablokuj MACD histogram i crossover")
            macd_val   = st.checkbox("Włącz MACD", value=True, key="lock_macd_val",
                                      disabled=not lock_macd)
        with ind_c3:
            lock_ema   = st.checkbox("🔒 EMA (9/21/50)", value=False, key="lock_ema",
                                      help="Zablokuj EMA — zawsze włączone lub wyłączone")
            ema_val    = st.checkbox("Włącz EMA", value=True, key="lock_ema_val",
                                      disabled=not lock_ema)

        # Zapisz do session_state
        locked_indicators = {
            "rsi":  (lock_rsi,  rsi_val),
            "macd": (lock_macd, macd_val),
            "ema":  (lock_ema,  ema_val),
        }
        st.session_state["_locked_indicators"] = locked_indicators

        lock_c1, lock_c2 = st.columns(2)
        with lock_c1:
            lock_sl    = st.checkbox("🔒 Stop Loss %",     value=True,  key="lock_sl",  help="Zablokuj Stop Loss — Optuna użyje Twojej wartości")
            lock_tp    = st.checkbox("🔒 Take Profit %",   value=True,  key="lock_tp",  help="Zablokuj Take Profit — Optuna użyje Twojej wartości")
        with lock_c2:
            lock_lev   = st.checkbox("🔒 Dźwignia",        value=True,  key="lock_lev", help="Zablokuj dźwignię — zawsze stała")
            lock_pos   = st.checkbox("🔒 Max kapitał na pozycję %", value=True, key="lock_pos", help="Zablokuj max % kapitału na pozycję")

        val_c1, val_c2, val_c3, val_c4 = st.columns(4)
        with val_c1:
            fixed_sl  = st.number_input("SL (%)",  min_value=0.1, max_value=10.0, value=1.5, step=0.1, key="fixed_sl",  disabled=not lock_sl,  format="%.1f")
        with val_c2:
            fixed_tp  = st.number_input("TP (%)",  min_value=0.1, max_value=100.0, value=3.0, step=0.5, key="fixed_tp",  disabled=not lock_tp,  format="%.1f")
        with val_c3:
            fixed_lev = st.number_input("Dźwignia", min_value=1,  max_value=20,   value=5,   step=1,   key="fixed_lev", disabled=not lock_lev, format="%d")
        with val_c4:
            fixed_pos = st.number_input("Max poz. (%)", min_value=0.5, max_value=100.0, value=2.0, step=0.5, key="fixed_pos", disabled=not lock_pos, format="%.1f")

        # Zapisz do session_state
        st.session_state["_locked_params"] = {
            "stop_loss_pct":    (lock_sl,  fixed_sl  / 100),
            "take_profit_pct":  (lock_tp,  fixed_tp  / 100),
            "leverage":         (lock_lev, int(fixed_lev)),
            "max_position_pct": (lock_pos, fixed_pos / 100),
        }

        st.markdown('<div class="param-group">👤 Parametry HITL</div>',unsafe_allow_html=True)
        ma_hitl_on       = st.checkbox("Włącz HITL",value=True,key="ma_hitl_on",help="Czy system ma się zatrzymywać i pytać o decyzję.")
        ma_hitl_interval = st.slider("HITL co ile prób",5,50,10,5,key="ma_hitl_int",disabled=not ma_hitl_on,help="Co ile prób Optuna włącza się tryb HITL.")
        ma_plateau       = st.slider("Plateau (próby bez poprawy)",5,30,15,5,key="ma_plateau",help="Po ilu próbach bez poprawy HITL się włącza automatycznie.")
        # Zapisz do session_state żeby był dostępny poza blokiem kolumny
        st.session_state["_ma_hitl_on"]       = ma_hitl_on
        st.session_state["_ma_hitl_interval"] = ma_hitl_interval
        st.session_state["_ma_plateau"]       = ma_plateau

        st.markdown(f'<div class="card" style="padding:14px 18px;"><div class="card-title">Podsumowanie</div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;line-height:1.8;">{ma_sym} · {ma_iv}<br>{ma_df} → {ma_dt}<br>Próby: {ma_trials} × {ma_ep_trial} ep. = {ma_trials*ma_ep_trial} epizodów<br>Dźwignia: x5 (stałe) · Max poz.: 2% (stałe)<br>HITL: {"co "+str(ma_hitl_interval)+" prób" if ma_hitl_on else "wyłączony"}</div></div>',unsafe_allow_html=True)

        mb1,mb2 = st.columns(2)
        with mb1: ma_start = st.button("▶  Uruchom Meta-Agenta",use_container_width=True,disabled=st.session_state.meta_running,type="primary",key="ma_start")
        with mb2: ma_stop  = st.button("⏹  Zatrzymaj",use_container_width=True,disabled=not st.session_state.meta_running,key="ma_stop")

        if not st.session_state.meta_running:
            if st.button("🔁  Reset UI",use_container_width=False,key="ma_reset"):
                st.session_state.meta_running=False
                st.session_state.meta_instance=None
                st.session_state.meta_hitl_waiting=False
                st.rerun()

    with ma_c2:
        st.markdown('<div class="param-group">📈 Postęp Meta-Agenta</div>',unsafe_allow_html=True)
        mp = st.session_state.meta_progress
        pct = mp.pct_complete if mp else 0.0
        bc  = "#3fb950" if pct>=100 else "#d29922" if (mp and mp.hitl_waiting) else "#58a6ff"
        trial_now = mp.trial if mp else 0

        st.markdown(f'<div class="card" style="padding:14px 18px;"><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">POSTĘP OPTUNA</span><span style="font-family:Space Mono,monospace;font-size:13px;font-weight:700;color:{bc};">{pct:.0f}%</span></div><div class="progress-bar-wrap"><div style="width:{pct:.0f}%;height:100%;background:{bc};border-radius:4px;"></div></div><div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-top:6px;">Trial: {trial_now} / {mp.total_trials if mp else ma_trials}</div></div>',unsafe_allow_html=True)

        if mp and mp.trial > 0:
            mm1,mm2,mm3,mm4 = st.columns(4)
            bs_c = "#3fb950" if mp.best_score > 0 else "#f85149"
            cs_c = "#3fb950" if (mp.current_score or 0) > 0 else "#f85149"
            with mm1: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Best Score</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:{bs_c};">{mp.best_score:.3f}</div></div>',unsafe_allow_html=True)
            with mm2: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Current</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:{cs_c};">{mp.current_score:.3f}</div></div>',unsafe_allow_html=True)
            with mm3: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">Best Trial</div><div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;">#{mp.best_trial}</div></div>',unsafe_allow_html=True)
            with mm4: st.markdown(f'<div class="card" style="padding:10px 14px;"><div class="card-title">ETA</div><div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;">{mp.eta_sec:.0f}s</div></div>',unsafe_allow_html=True)

        # Log
        st.markdown('<div class="param-group" style="margin-top:8px;">📋 Log Meta-Agenta</div>',unsafe_allow_html=True)
        log_lines=[]
        if mp and mp.history:
            for h in mp.history[-15:]:
                sc_sign="+" if h["score"]>=0 else ""
                bal  = h.get("balance", 0)
                trd  = h.get("trades", 0)
                log_lines.append(
                    f"[T{h['trial']:>3}] "
                    f"Score:{sc_sign}{h['score']:.3f}  "
                    f"Ret:{h['return_pct']:+.1f}%  "
                    f"Bal:${bal:,.0f}  "
                    f"Trades:{trd:>3}  "
                    f"WR:{h['win_rate']:.0f}%  "
                    f"Sharpe:{h['sharpe']:.2f}  "
                    f"DD:{h['drawdown']:.1f}%"
                )
        if mp and mp.message:
            log_lines.append(f"\n>>> {mp.message}")
        log_text="\n".join(log_lines) if log_lines else "Oczekiwanie na start meta-agenta..."
        st.markdown(f'<div class="train-log">{log_text}</div>',unsafe_allow_html=True)

        if mp and mp.elapsed_sec > 0:
            st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;margin-top:6px;">Czas: {mp.elapsed_sec:.0f}s · ETA: {mp.eta_sec:.0f}s</div>',unsafe_allow_html=True)

    # ── HITL Panel ─────────────────────────────────────────────────
    if st.session_state.meta_hitl_waiting:
        st.markdown("---")
        hitl_data = st.session_state.meta_hitl_data
        rec       = hitl_data.get("recommendation", {})

        st.markdown(f'<div style="background:#1a1a2f;border:2px solid #d29922;border-radius:8px;padding:20px 28px;"><div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;color:#d29922;margin-bottom:12px;">⏸️ HITL — OCZEKIWANIE NA TWOJĄ DECYZJĘ</div><div style="font-family:Space Mono,monospace;font-size:12px;color:#e6edf3;margin-bottom:8px;">{rec.get("summary","")}</div></div>',unsafe_allow_html=True)

        hitl_c1, hitl_c2 = st.columns(2)

        with hitl_c1:
            if rec.get("issues"):
                st.markdown('<div class="param-group">⚠️ Wykryte problemy</div>',unsafe_allow_html=True)
                for issue in rec["issues"]:
                    st.markdown(f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#f85149;padding:4px 0;">• {issue}</div>',unsafe_allow_html=True)
            if rec.get("recommendations"):
                st.markdown('<div class="param-group">💡 Rekomendacje</div>',unsafe_allow_html=True)
                for r2 in rec["recommendations"]:
                    st.markdown(f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#3fb950;padding:4px 0;">→ {r2}</div>',unsafe_allow_html=True)

        with hitl_c2:
            next_p = rec.get("optuna_next", {})
            if next_p:
                st.markdown('<div class="param-group">🔬 Sugestia Optuna (następna próba)</div>',unsafe_allow_html=True)
                rows=""
                for k,v in list(next_p.items())[:8]:
                    if isinstance(v, float): v_str=f"{v:.4f}"
                    else: v_str=str(v)
                    rows+=f'<div class="indicator-row"><span class="indicator-label">{k[:30]}</span><span class="indicator-value" style="font-size:11px;">{v_str}</span></div>'
                st.markdown(f'<div class="card" style="padding:12px 16px;">{rows}</div>',unsafe_allow_html=True)

        st.markdown('<div class="param-group">✏️ Twoje modyfikacje (opcjonalne)</div>',unsafe_allow_html=True)
        mod_c1, mod_c2, mod_c3 = st.columns(3)
        with mod_c1:
            mod_profit = st.slider("Profit multiplier",0.5,5.0,2.0,0.1,key="hitl_profit")
            mod_loss   = st.slider("Loss multiplier",0.5,5.0,2.5,0.1,key="hitl_loss")
        with mod_c2:
            mod_div_w  = st.slider("Divergence weight",0.0,1.0,0.6,0.05,key="hitl_divw")
            mod_sl     = st.slider("Stop Loss (%)",0.5,5.0,1.5,0.1,key="hitl_sl")
        with mod_c3:
            mod_hold   = st.slider("Max hold candles",5,100,50,5,key="hitl_hold")
            mod_ot     = st.slider("Overtrading penalty",0.0,1.0,0.2,0.05,key="hitl_ot")

        hb1,hb2,hb3 = st.columns(3)
        with hb1:
            if st.button("✅ Akceptuj rekomendacje Optuna",use_container_width=True,key="hitl_accept"):
                if st.session_state.meta_instance:
                    st.session_state.meta_instance.user_accept_hitl()
                st.session_state.meta_hitl_waiting=False
                st.rerun()
        with hb2:
            if st.button("✏️ Zastosuj moje modyfikacje",use_container_width=True,key="hitl_modify"):
                mods = {
                    "profit_multiplier": mod_profit,
                    "loss_multiplier":   mod_loss,
                    "divergence_weight": mod_div_w,
                    "stop_loss_pct":     mod_sl/100,
                    "max_hold_candles":  mod_hold,
                    "overtrading_penalty": mod_ot,
                }
                if st.session_state.meta_instance:
                    st.session_state.meta_instance.user_modify_hitl(mods)
                st.session_state.meta_hitl_waiting=False
                st.rerun()
        with hb3:
            if st.button("⏭️ Pomiń — kontynuuj bez zmian",use_container_width=True,key="hitl_skip"):
                if st.session_state.meta_instance:
                    st.session_state.meta_instance.user_skip_hitl()
                st.session_state.meta_hitl_waiting=False
                st.rerun()

    # ── Top 3 wyniki ───────────────────────────────────────────────
    if st.session_state.meta_instance or (st.session_state.meta_progress and st.session_state.meta_progress.trial > 0):
        st.markdown("---")
        st.markdown("<div class='section-header'>🏆 Top 3 Najlepsze Wyniki</div>",unsafe_allow_html=True)

        top3 = st.session_state.meta_instance.get_top3() if st.session_state.meta_instance else []

        if not top3:
            st.markdown('<div class="card" style="padding:14px;"><span style="font-family:Space Mono,monospace;font-size:12px;color:#8b949e;">Brak wyników — uruchom meta-agenta</span></div>',unsafe_allow_html=True)
        else:
            for i, entry in enumerate(top3, 1):
                m  = entry.get("metrics", {})
                medal = ["🥇","🥈","🥉"][i-1]
                ret_c = "#3fb950" if m.get("total_return_pct",0)>=0 else "#f85149"
                st.markdown(f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;"><span style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;">{medal} #{i} — Score: {entry["composite_score"]:.4f}</span><span style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;">Trial #{entry.get("trial_number","?")} · {entry.get("timestamp","")}</span></div><div style="display:flex;gap:24px;flex-wrap:wrap;"><span style="font-family:Space Mono,monospace;font-size:13px;color:{ret_c};">Return: {m.get("total_return_pct",0):+.2f}%</span><span style="font-family:Space Mono,monospace;font-size:13px;">Sharpe: {m.get("sharpe",0):.3f}</span><span style="font-family:Space Mono,monospace;font-size:13px;color:#f85149;">DD: {m.get("max_drawdown_pct",0):.1f}%</span><span style="font-family:Space Mono,monospace;font-size:13px;">WR: {m.get("win_rate",0):.0f}%</span><span style="font-family:Space Mono,monospace;font-size:13px;">Trades: {m.get("total_trades",0)}</span></div></div>',unsafe_allow_html=True)

        if top3:
            if st.button("💾 Eksportuj Top 3 do pliku",use_container_width=False,key="export_top3"):
                if st.session_state.meta_instance:
                    st.session_state.meta_instance._ckpt._save_top3_txt()
                    st.success("Zapisano do checkpoints/top3.txt")

    # ── Akcje ─────────────────────────────────────────────────────
    if ma_start and not st.session_state.meta_running:
        import requests as _req
        from trader.meta_agent import MetaAgent, MetaAgentConfig
        from trader.hitl_controller import HITLConfig
        from trader.trainer import TrainingConfig

        tc_obj = TrainingConfig(
            symbol=ma_sym, interval=ma_iv,
            date_from=str(ma_df), date_to=str(ma_dt),
            initial_balance=ma_balance, commission_pct=ma_com,
        )
        mc_obj = MetaAgentConfig(
            n_trials             = ma_trials,
            n_episodes_per_trial = ma_ep_trial,
            study_name           = ma_study_name,
            storage              = ma_storage if ma_continue else None,
            load_if_exists       = ma_continue,
            locked_params        = st.session_state.get("_locked_params", {}),
            locked_indicators    = st.session_state.get("_locked_indicators", {}),
        )
        _hitl_on       = st.session_state.get("_ma_hitl_on", False)
        _hitl_interval = st.session_state.get("_ma_hitl_interval", 10)
        _plateau       = st.session_state.get("_ma_plateau", 15)
        hc_obj = HITLConfig(
            hitl_enabled      = _hitl_on,
            hitl_interval     = _hitl_interval,
            plateau_threshold = _plateau,
            plateau_hitl      = _hitl_on,
        )

        def on_meta_progress(p):
            st.session_state.meta_progress     = p
            st.session_state.meta_hitl_waiting = p.hitl_waiting

        def on_hitl(state, metrics, rec):
            st.session_state.meta_hitl_waiting = True
            st.session_state.meta_hitl_data    = {"state": state, "recommendation": rec, "metrics": metrics}

        # Pobierz świece
        from trader.trainer import Trainer
        trainer_tmp = Trainer(
            training_config=tc_obj,
            reward_config=st.session_state.reward_config,
            feature_config=st.session_state.feature_config,
        )
        candles = trainer_tmp._fetch_candles()

        if not candles:
            st.error("Nie udało się pobrać danych historycznych.")
        else:
            meta = MetaAgent(
                meta_config=mc_obj,
                training_config=tc_obj,
                hitl_config=hc_obj,
                candles=candles,
                progress_callback=on_meta_progress,
                hitl_callback=on_hitl,
            )
            st.session_state.meta_instance = meta
            st.session_state.meta_running  = True
            st.session_state.meta_progress = meta.progress

            def run_meta():
                try:
                    meta.run()
                finally:
                    st.session_state.meta_running = False

            threading.Thread(target=run_meta, daemon=True).start()
            st.rerun()

    if ma_stop and st.session_state.meta_running:
        if st.session_state.meta_instance:
            st.session_state.meta_instance.stop()
        st.session_state.meta_running      = False
        st.session_state.meta_hitl_waiting = False
        st.rerun()

    # Auto-refresh: podczas treningu AND podczas oczekiwania na HITL
    if st.session_state.meta_running or st.session_state.meta_hitl_waiting:
        time.sleep(1)
        st.rerun()
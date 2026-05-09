# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import plotly.graph_objects as go

from data_fetcher.binance_fetcher import BinanceDataFetcher
from features.engineer import add_features
from config.config import config

st.set_page_config(page_title="Algo Trading RL v3", layout="wide")
st.title("🚀 Algo Trading v3 - Reinforcement Learning + CVD")

if "fetcher" not in st.session_state:
    st.session_state.fetcher = BinanceDataFetcher(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET
    )

fetcher = st.session_state.fetcher

# ====================== ZAKŁADKI ======================
tab_main, tab_params, tab_reward, tab_charts, tab_data, tab_rl, tab_auto = st.tabs([
    "🏠 Główny Panel",
    "⚙️ Parametry Wejściowe",
    "🎯 Reward & Risk Settings",
    "📈 Wykresy",
    "📋 Dane Surowe",
    "🤖 Trening RL",
    "🔄 Auto-Optymalizacja (HITL + HOTL)"
])

# ====================== 1. GŁÓWNY PANEL ======================
with tab_main:
    st.header("📥 Pobieranie Danych")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        symbol = st.selectbox("Para walutowa", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"], index=0)
        timeframe = st.selectbox("Interwał", ["5m", "15m", "30m", "1h", "4h", "1d"], index=0)

    with col_b:
        start_date = st.date_input("Data początkowa", datetime.now() - timedelta(days=30))
        end_date = st.date_input("Data końcowa", datetime.now())

    if st.button("📥 POBIERZ DANE + OBLICZ CVD", type="primary", width='stretch'):
        with st.spinner("Pobieranie danych..."):
            df_raw = fetcher.get_historical_data(symbol, timeframe, days=(end_date - start_date).days + 1)
            df = add_features(df_raw)
            
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}"
            
            selected = st.session_state.get("selected_features", [])
            if selected:
                cols_to_save = ["open", "high", "low", "close", "volume"] + [c for c in selected if c in df.columns]
                df_to_save = df[[c for c in cols_to_save if c in df.columns]]
            else:
                df_to_save = df
            
            csv_path = f"data/{base_name}.csv"
            df_to_save.to_csv(csv_path)
            
            config_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "downloaded_at": datetime.now().isoformat(),
                "selected_features": st.session_state.get("selected_features", []),
                "trading_config": st.session_state.get("trading_config", {}),
                "total_rows": len(df)
            }
            
            json_path = f"data/{base_name}_config.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            st.session_state.df = df
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
            st.session_state.last_data_file = csv_path
            
            st.success(f"✅ Zapisano dane: {csv_path}")
            st.success(f"✅ Zapisano konfigurację: {json_path}")

# ====================== 2. PARAMETRY WEJŚCIOWE ======================
with tab_params:
    st.header("⚙️ Parametry Wejściowe – Cechy dla algorytmu")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Podstawowe Dane")
        use_ohlc = st.checkbox("OHLC (Open, High, Low, Close)", value=True)
        use_volume = st.checkbox("Volume", value=True)
        use_returns = st.checkbox("Returns i Log Returns", value=True)

        st.subheader("Volume Analysis")
        use_buy_sell = st.checkbox("Buy & Sell Volume", value=True)
        use_volume_delta = st.checkbox("Volume Delta", value=True)
        use_cvd = st.checkbox("Cumulative Volume Delta (CVD)", value=True)
        use_buy_ratio = st.checkbox("Buy/Sell Ratio", value=True)

    with col2:
        st.subheader("Wskaźniki Techniczne")
        use_rsi = st.checkbox("RSI (14)", value=True)
        use_macd = st.checkbox("MACD", value=True)
        use_bollinger = st.checkbox("Bollinger Bands", value=True)
        use_atr = st.checkbox("ATR", value=True)
        use_ema = st.checkbox("EMA (9, 21, 50)", value=True)
        use_sma = st.checkbox("SMA 200", value=False)
        use_stochastic = st.checkbox("Stochastic Oscillator", value=False)
        use_adx = st.checkbox("ADX", value=False)

        st.subheader("Dodatkowe")
        use_time_features = st.checkbox("Godzina + Dzień tygodnia", value=False)

    st.divider()
    if st.button("💾 Zapisz wybrane cechy", width='stretch'):
        selected_features = []
        if use_ohlc: selected_features.extend(["open", "high", "low", "close"])
        if use_volume: selected_features.append("volume")
        if use_returns: selected_features.extend(["returns", "log_returns"])
        if use_buy_sell: selected_features.extend(["buy_volume", "sell_volume"])
        if use_volume_delta: selected_features.append("volume_delta")
        if use_cvd: selected_features.append("cvd")
        if use_rsi: selected_features.append("rsi")
        if use_macd: selected_features.extend(["macd", "macd_signal"])
        if use_bollinger: selected_features.extend(["bb_upper", "bb_middle", "bb_lower", "bb_width"])
        if use_atr: selected_features.append("atr")
        if use_ema: selected_features.extend(["ema_9", "ema_21", "ema_50"])

        st.session_state.selected_features = selected_features
        st.success(f"✅ Zapisano {len(selected_features)} cech")

# ====================== 3. REWARD & RISK SETTINGS ======================
with tab_reward:
    st.header("🎯 Reward & Risk Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Kapitał i Pozycja")
        initial_capital = st.number_input("Początkowy kapitał (USDT)", min_value=100.0, max_value=1000000.0, value=10000.0, step=500.0)

        position_mode = st.radio("Tryb wielkości pozycji", ["Procent kapitału", "Stała kwota (USDT)"], horizontal=True)
        if position_mode == "Procent kapitału":
            position_size_pct = st.slider("Wielkość pozycji (% kapitału)", 1, 100, 10, 1)
            position_size_usdt = None
        else:
            position_size_usdt = st.number_input("Wielkość pozycji (USDT)", 10.0, 100000.0, 1000.0, 50.0)
            position_size_pct = None

        max_position_pct = st.slider("Maksymalna wielkość jednej pozycji (% kapitału)", 5, 50, 25, 1)

    with col2:
        st.subheader("⚖️ Zarządzanie Ryzykiem")
        stop_loss_pct = st.slider("Stop Loss (%)", 0.5, 10.0, 3.0, 0.1)
        take_profit_pct = st.slider("Take Profit (%)", 1.0, 100.0, 15.0, 0.5)
        max_drawdown_limit = st.number_input("Maksymalny dopuszczalny Drawdown (%)", value=15.0, min_value=5.0, max_value=50.0)

    st.divider()
    st.subheader("📈 Nagrody i Kary")

    col3, col4 = st.columns(2)
    with col3:
        reward_profit = st.slider("Nagroda za zrealizowany zysk (PnL)", -5.0, 5.0, 1.2, 0.1)
        reward_unrealized = st.slider("Nagroda za niezrealizowany zysk (floating PnL)", -2.0, 2.0, 0.4, 0.1)
        reward_sharpe = st.slider("Nagroda za wysoki Sharpe Ratio", 0.0, 3.0, 0.8, 0.1)
        reward_holding = st.slider("Nagroda za trzymanie dobrej pozycji (per krok)", -0.5, 0.5, 0.05, 0.01)

    with col4:
        penalty_drawdown = st.slider("Kara za Drawdown (%)", 0.0, 10.0, 2.5, 0.1)
        penalty_transaction = st.slider("Koszt transakcji (commission + slippage)", 0.0, 2.0, 0.15, 0.01)
        penalty_holding_time = st.slider("Kara za zbyt długie trzymanie pozycji", 0.0, 1.0, 0.08, 0.01)
        penalty_large_position = st.slider("Kara za zbyt dużą pozycję", 0.0, 5.0, 1.2, 0.1)
        penalty_inactivity = st.slider("Kara za brak aktywności", 0.0, 1.0, 0.02, 0.005)

    st.divider()
    st.subheader("🔧 Zaawansowane parametry")
    col5, col6 = st.columns(2)
    with col5:
        risk_aversion = st.slider("Aversion do ryzyka (Risk Aversion)", 0.0, 5.0, 1.5, 0.1)
        win_rate_bonus = st.checkbox("Bonus za wysoki Win Rate", value=True)
    with col6:
        use_sortino = st.checkbox("Użyj Sortino Ratio zamiast Sharpe", value=False)

    st.divider()
    st.subheader("📉 Futures & Dźwignia")
    use_futures = st.checkbox("Użyj trybu Futures (z dźwignią)", value=False)
    if use_futures:
        leverage = st.select_slider("Dźwignia", options=[1, 5, 10, 20, 50, 75, 100, 125], value=10)
        st.caption("Maksymalna dźwignia na Binance USDT-M Perpetual: 125x")
        st.session_state.leverage = leverage
        st.session_state.use_futures = True
    else:
        st.session_state.leverage = 1
        st.session_state.use_futures = False

    st.divider()
    if st.button("💾 Zapisz wszystkie ustawienia Reward & Risk", type="primary", width='stretch'):
        st.session_state.trading_config = {
            "initial_capital": initial_capital,
            "position_size_pct": position_size_pct,
            "position_size_usdt": position_size_usdt,
            "max_position_pct": max_position_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "max_drawdown_limit": max_drawdown_limit,
            "reward_profit": reward_profit,
            "reward_unrealized": reward_unrealized,
            "reward_sharpe": reward_sharpe,
            "reward_holding": reward_holding,
            "penalty_drawdown": penalty_drawdown,
            "penalty_transaction": penalty_transaction,
            "penalty_holding_time": penalty_holding_time,
            "penalty_large_position": penalty_large_position,
            "penalty_inactivity": penalty_inactivity,
            "risk_aversion": risk_aversion,
            "win_rate_bonus": win_rate_bonus,
            "use_sortino": use_sortino,
            "leverage": st.session_state.get("leverage", 1),
            "use_futures": st.session_state.get("use_futures", False),
        }
        st.success("✅ Wszystkie ustawienia Reward & Risk zostały zapisane!")

# ====================== 4. WYKRESY ======================
with tab_charts:
    if "df" in st.session_state and st.session_state.df is not None:
        st.subheader(f"Wykres: {st.session_state.get('symbol', '---')}")
        st.dataframe(st.session_state.df.tail(10), width='stretch')
        st.success("✅ Dane załadowane — wykres będzie wkrótce")
    else:
        st.info("Pobierz dane w zakładce Główny Panel")

# ====================== 5. DANE SUROWE ======================
with tab_data:
    if "df" in st.session_state and st.session_state.df is not None:
        st.subheader("Dane surowe")
        st.dataframe(st.session_state.df.tail(20), width='stretch')
    else:
        st.info("Pobierz dane w zakładce Główny Panel")

# ====================== 6. TRENING RL ======================
with tab_rl:
    st.header("🤖 Trening Agenta RL")
    
    has_data = "df" in st.session_state and st.session_state.df is not None
    
    if not has_data:
        st.warning("📥 Najpierw pobierz dane w zakładce 'Główny Panel'")
    
    st.subheader("📁 Wybierz plik z danymi")
    data_files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.exists("data") else []
    
    if data_files:
        selected_file = st.selectbox("Plik CSV", data_files, key="selected_csv_rl")
        if st.button("📥 Załaduj wybrany plik", width='stretch'):
            df = pd.read_csv(f"data/{selected_file}", index_col=0, parse_dates=True)
            st.session_state.df = df
            st.success(f"✅ Załadowano: {selected_file}")
    
    st.subheader("⚙️ Parametry uczenia (PPO)")
    
    col_lr, col_steps = st.columns(2)
    
    with col_lr:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.01,
            value=0.0002,
            step=0.00001,
            format="%.5f",
            help="Mniejsza wartość = bardziej stabilne uczenie"
        )
    
    with col_steps:
        total_timesteps = st.select_slider(
            "Liczba kroków treningu",
            options=[50000, 100000, 200000, 500000],
            value=100000
        )
    
    st.session_state.learning_rate = learning_rate
    
    if st.button("🚀 Rozpocznij trening", type="primary", width='stretch', disabled=not has_data):
        with st.spinner("Trwa trening... To może potrwać kilka minut."):
            try:
                from rl.train import train_agent
                config = st.session_state.get("trading_config", {})
                
                result = train_agent(
                    df=st.session_state.df,
                    config=config,
                    total_timesteps=total_timesteps,
                    learning_rate=learning_rate,
                    save_path="models/"
                )
                
                st.success("✅ Trening zakończony pomyślnie!")
                st.info(f"Model zapisany w folderze `models/`")
                
            except Exception as e:
                st.error(f"❌ Błąd podczas treningu: {str(e)}")

# ====================== 7. AUTO-OPTIMALIZACJA (HITL + HOTL) ======================
with tab_auto:
    st.header("🔄 Auto-Optymalizacja (HITL + HOTL)")
    st.info("System sam dostosowuje parametry. Optuna uczy się z poprzednich wyników i idzie w stronę lepszych rozwiązań.")
    
    has_data = "df" in st.session_state and st.session_state.df is not None
    
    if not has_data:
        st.warning("📥 Najpierw pobierz dane w zakładce 'Główny Panel'")
    
    # === OGRANICZENIE: TYLKO BASIC + VOLUME ANALYSIS ===
    st.subheader("📁 Wybierz plik z danymi (dla optymalizacji)")
    st.caption("Do optymalizacji używane są tylko: Podstawowe dane + Volume Analysis (bez wskaźników technicznych)")
    
    data_files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.exists("data") else []
    
    if data_files:
        selected_file = st.selectbox("Plik CSV", data_files, key="selected_csv_auto")
        if st.button("📥 Załaduj plik do optymalizacji", width='stretch'):
            df = pd.read_csv(f"data/{selected_file}", index_col=0, parse_dates=True)
            
            # === FILTR: Tylko Basic + Volume Analysis ===
            allowed_cols = ["open", "high", "low", "close", "volume", 
                           "buy_volume", "sell_volume", "volume_delta", "cvd", "buy_ratio"]
            df_filtered = df[[col for col in allowed_cols if col in df.columns]]
            
            st.session_state.df = df_filtered
            st.success(f"✅ Załadowano (tylko Basic + Volume): {selected_file}")
            st.info(f"Używane kolumny: {list(df_filtered.columns)}")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.number_input("Liczba prób (trials)", min_value=10, max_value=100, value=25, step=5)
    with col2:
        checkpoint_every = st.number_input("Zapisz checkpoint co ile prób", min_value=3, max_value=10, value=5)
    
    if st.button("🚀 Uruchom Auto-Optymalizację", type="primary", width='stretch', disabled=not has_data):
        with st.spinner("Uruchamianie optymalizacji... To może potrwać 30–90 minut."):
            try:
                from rl.optimizer import RLOptimizer
                
                config = st.session_state.get("trading_config", {})
                
                optimizer = RLOptimizer(
                    df=st.session_state.df,
                    base_config=config,
                    n_trials=n_trials
                )
                
                result = optimizer.run()
                
                st.success("✅ Optymalizacja zakończona!")
                st.write("**Najlepsze parametry:**", result["best_params"])
                st.write("**Najlepszy wynik (Score):**", round(result["best_score"], 4))
                
                if result["best_model_path"]:
                    st.info(f"📁 Najlepszy model zapisany: {result['best_model_path']}")
                
            except Exception as e:
                st.error(f"❌ Błąd optymalizacji: {str(e)}")

    # Checkpoint
    st.divider()
    st.subheader("📂 Wczytaj checkpoint (kontynuacja)")
    
    checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".json")] if os.path.exists("checkpoints") else []
    
    if checkpoint_files:
        selected_checkpoint = st.selectbox("Wybierz checkpoint", checkpoint_files)
        if st.button("📥 Wczytaj i kontynuuj", width='stretch'):
            with open(f"checkpoints/{selected_checkpoint}") as f:
                checkpoint = json.load(f)
            st.success(f"✅ Wczytano checkpoint z próby {checkpoint['trial']}")
            st.write("Ostatnie wyniki:", checkpoint.get("results", []))
    else:
        st.info("Brak zapisanych checkpointów.")

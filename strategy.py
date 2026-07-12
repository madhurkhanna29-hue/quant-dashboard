import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="Quant Trading System", layout="wide")

st.sidebar.header("Dashboard Settings")
display_currency = st.sidebar.selectbox("Display Currency", ["INR", "USD", "GBP"], index=0)

# Central Normalization Engine
usd_to_inr_rate = 83.50  
gbp_to_inr_rate = 105.20

if display_currency == "INR":
    conversion_factor = 1.0
    sym = "₹"
elif display_currency == "USD":
    conversion_factor = 1 / usd_to_inr_rate
    sym = "$"
elif display_currency == "GBP":
    conversion_factor = 1 / gbp_to_inr_rate
    sym = "£"

# Initialize Portfolio State
if 'portfolios' not in st.session_state:
    starting_capital_usd = 20000.0
    st.session_state.portfolios = {
        'Live Strategy Execution': [],
        'Madhur (acc2)': [
            {'Date': '2026-07-10', 'Asset': 'XRP', 'Action': 'LONG', 'Size': 10000, 'Entry Price': 0.4200, 'LTP': 0.4512}
        ]
    }
    st.session_state.cash_usd = starting_capital_usd

# --- 2. DATA & ENGINE ---
@st.cache_data(ttl=3600)
def load_data():
    ticker = yf.Ticker("^NDX")
    df = ticker.history(period="max")
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.sort_values('Date', inplace=True)
    return df

@st.cache_data
def run_strategy(df):
    df['Ret'] = df['Close'].pct_change()
    
    # Indicators
    df['EMA_60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['EMA_230'] = df['Close'].ewm(span=230, adjust=False).mean()
    df['Bull'] = df['EMA_60'] > df['EMA_230']
    df['Bear'] = ~df['Bull']
    
    df['HV_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['HV_Rank'] = df['HV_20'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_2'] = 100 - (100 / (1 + rs))
    
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR_14'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = df['ATR_14'] / df['Close']
    df['ATR_Baseline'] = df['ATR_Pct'].rolling(252).median()
    df['Size_Multiplier'] = (df['ATR_Baseline'] / df['ATR_Pct']).clip(upper=1.0).fillna(1.0)
    
    df['Quiet'] = df['HV_Rank'] < 0.85
    df['Volatile'] = df['HV_Rank'] >= 0.85

    core_sig, swing_sig = [], []
    short_exposure, swing_exposure = 0.0, 0.0
    core_entry_price, swing_entry_price = 0.0, 0.0

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        rsi = df['RSI_2'].iloc[i]
        bull, bear = df['Bull'].iloc[i], df['Bear'].iloc[i]
        q, v = df['Quiet'].iloc[i], df['Volatile'].iloc[i]
        ema_10 = df['EMA_10'].iloc[i]
        size_mult = df['Size_Multiplier'].iloc[i] if not pd.isna(df['Size_Multiplier'].iloc[i]) else 1.0
            
        # Core System
        if bull and q:
            short_exposure, core_entry_price = 0.0, 0.0
            core_sig.append(1.0 * size_mult)
        elif bear and v:
            if short_exposure < 0 and core_entry_price > 0 and price > core_entry_price * 1.05:
                short_exposure, core_entry_price = 0.0, 0.0
                
            if short_exposure == 0.0 and rsi > 75:
                short_exposure = -0.33 * size_mult
                core_entry_price = price
            elif short_exposure == -0.33 * size_mult and rsi > 80:
                short_exposure = -0.66 * size_mult
                core_entry_price = (core_entry_price + price) / 2
            elif short_exposure >= -0.66 * size_mult and short_exposure < 0 and rsi > 85:
                short_exposure = -1.0 * size_mult
                core_entry_price = (core_entry_price + price) / 2
                
            if rsi < 50:
                short_exposure, core_entry_price = 0.0, 0.0
            core_sig.append(short_exposure)
        else:
            short_exposure, core_entry_price = 0.0, 0.0
            core_sig.append(0.0)
            
        # Swing System
        if core_sig[-1] == 0:
            if swing_exposure < 0 and swing_entry_price > 0 and price > swing_entry_price * 1.05:
                swing_exposure, swing_entry_price = 0.0, 0.0
            
            if bear and q:
                if price < ema_10:
                    if swing_exposure == 0.0 and rsi > 70:
                        swing_exposure = -0.33 * size_mult
                        swing_entry_price = price
                    elif swing_exposure == -0.33 * size_mult and rsi > 80:
                        swing_exposure = -0.66 * size_mult
                        swing_entry_price = (swing_entry_price + price) / 2
                    elif swing_exposure <= -0.33 * size_mult and swing_exposure > -1.0 * size_mult and rsi > 90:
                        swing_exposure = -1.0 * size_mult
                        swing_entry_price = (swing_entry_price + price) / 2
                if rsi < 40 and swing_exposure < 0:
                    swing_exposure, swing_entry_price = 0.0, 0.0
            else:
                 swing_exposure, swing_entry_price = 0.0, 0.0
            swing_sig.append(swing_exposure)
        else:
            swing_exposure, swing_entry_price = 0.0, 0.0
            swing_sig.append(0.0)

    df['Core_Sig'] = core_sig
    df['Swing_Sig'] = swing_sig
    df['Core_Sig'] = df['Core_Sig'].shift(1).fillna(0)
    df['Swing_Sig'] = df['Swing_Sig'].shift(1).fillna(0)
    
    df['Ret_3x'] = df['Ret'] * 3
    df['Ret_m3x'] = df['Ret'] * -3
    df['Ret_1x'] = df['Ret']
    
    df['Core_Ret'] = np.where(df['Core_Sig'] > 0, df['Ret_3x'] * df['Core_Sig'], np.where(df['Core_Sig'] < 0, df['Ret_m3x'] * abs(df['Core_Sig']), 0))
    df['Swing_Ret'] = np.where(df['Swing_Sig'] > 0, df['Ret_1x'] * df['Swing_Sig'], np.where(df['Swing_Sig'] < 0, df['Ret_1x'] * df['Swing_Sig'], 0))
    df['Total_Strat_Ret'] = df['Core_Ret'] + df['Swing_Ret']
    
    last = df.iloc[-1]
    
    if last['Core_Sig'] > 0: signal, leverage = "CORE LONG", last['Core_Sig'] * 3.0
    elif last['Core_Sig'] < 0: signal, leverage = "CORE SHORT", last['Core_Sig'] * 3.0
    elif last['Swing_Sig'] < 0: signal, leverage = "SWING SHORT", last['Swing_Sig'] * 1.0
    else: signal, leverage = "CASH (Regime Filtered)", 0.0
        
    return df, last, signal, leverage, core_entry_price, swing_entry_price

# --- 3. UI RENDERING ---
st.title(f"Systematic 4-Quadrant Strategy ({display_currency})")

df_market = load_data()
df_strat, latest_data, current_signal, target_leverage, c_entry, s_entry = run_strategy(df_market)

# Format pricing to requested display currency via INR normalization block
usd_close = latest_data['Close']
norm_close_inr = usd_close * usd_to_inr_rate
display_price = norm_close_inr * conversion_factor

avg_entry = c_entry if c_entry > 0 else s_entry
display_entry = ((avg_entry * usd_to_inr_rate) * conversion_factor) if avg_entry > 0 else 0.0

st.header("1. Daily Execution Command")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric(label="Action Signal", value=current_signal)
with col2: st.metric(label="Target Leverage", value=f"{target_leverage:.2f}x")
with col3: st.metric(label=f"Nasdaq 100 ({display_currency})", value=f"{sym}{display_price:,.2f}")
with col4: st.metric(label=f"Avg Entry Price ({display_currency})", value=f"{sym}{display_entry:,.2f}" if display_entry > 0 else "N/A")

st.markdown("---")
st.header("2. Regime Diagnostics")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="Macro Trend (60/230)", value="Bullish" if latest_data['Bull'] else "Bearish")
c2.metric(label="Volatility Rank", value=f"{latest_data['HV_Rank']*100:.1f}%")
c3.metric(label="2-Day RSI", value=f"{latest_data['RSI_2']:.1f}")
c4.metric(label="ATR Size Multiplier", value=f"{latest_data['Size_Multiplier']:.2f}")

st.markdown("---")
st.header("3. Historical Backtest Results")
cum_total = (1 + df_strat['Total_Strat_Ret']).cumprod()
years_len = len(df_strat) / 252.0
cagr = (cum_total.iloc[-1]) ** (1 / years_len) - 1
vol = df_strat['Total_Strat_Ret'].std() * np.sqrt(252)
sharpe = (df_strat['Total_Strat_Ret'].mean() * 252) / vol
max_dd = (cum_total / cum_total.cummax() - 1).min()

c1, c2, c3, c4 = st.columns(4)
c1.metric(label="System CAGR", value=f"{cagr*100:.2f}%")
c2.metric(label="Max Drawdown", value=f"{max_dd*100:.2f}%")
c3.metric(label="Sharpe Ratio", value=f"{sharpe:.2f}")
c4.metric(label="Annualized Vol", value=f"{vol*100:.2f}%")

st.subheader("Yearly Performance Table")
years = df_strat['Date'].dt.year.unique()
yearly_stats = []
for yr in sorted(years, reverse=True):
    df_yr = df_strat[df_strat['Date'].dt.year == yr]
    strat_perf = (1 + df_yr['Total_Strat_Ret']).prod() - 1
    bh_perf = (1 + df_yr['Ret']).prod() - 1
    yearly_stats.append({
        'Year': yr,
        'Strategy Return': f"{strat_perf*100:.2f}%",
        'Benchmark (1x) Return': f"{bh_perf*100:.2f}%"
    })
st.dataframe(pd.DataFrame(yearly_stats), use_container_width=True)

st.markdown("---")
st.header("4. Trade Log & Portfolio Tracker")

# Manual Trade Input Form
with st.form("trade_entry"):
    st.subheader("Log New Trade Execution")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trade_date = st.date_input("Date", datetime.today())
    with col2:
        trade_action = st.selectbox("Action", ["LONG", "SHORT", "COVER/SELL"])
    with col3:
        trade_size = st.number_input("Position Size", min_value=0.01, step=0.1)
    with col4:
        trade_entry = st.number_input("Entry Price (USD)", min_value=0.01, step=0.1)
    
    submitted = st.form_submit_button("Log Trade")
    if submitted:
        st.session_state.portfolios['Live Strategy Execution'].append({
            'Date': trade_date.strftime('%Y-%m-%d'),
            'Asset': 'QQQ3.L',
            'Action': trade_action,
            'Size': trade_size,
            'Entry Price': trade_entry,
            'LTP': trade_entry # Defaults to entry upon logging
        })
        st.success("Trade logged successfully.")

# Display Portfolios
for port_name, holdings in st.session_state.portfolios.items():
    st.subheader(f"Portfolio: {port_name}")
    if len(holdings) > 0:
        df_port = pd.DataFrame(holdings)
        
        # Apply currency normalization and exact formatting rules
        df_port['Entry Price'] = (df_port['Entry Price'] * usd_to_inr_rate) * conversion_factor
        df_port['LTP'] = (df_port['LTP'] * usd_to_inr_rate) * conversion_factor
        
        # Format the LTP column to exactly 4 decimal places
        styled_port = df_port.style.format({
            'Entry Price': f"{sym}{{:.2f}}",
            'LTP': f"{sym}{{:.4f}}"
        })
        st.table(styled_port)
    else:
        st.write("No active trades logged.")
        
# Starting Capital Tracker
st.markdown(f"**Working Capital Base:** {sym}{((st.session_state.cash_usd * usd_to_inr_rate) * conversion_factor):,.2f} ({display_currency})")
                core_entry_price = (core_entry_price + price) / 2
                
            if rsi < 50:
                short_exposure, core_entry_price = 0.0, 0.0
            core_sig.append(short_exposure)
        else:
            short_exposure, core_entry_price = 0.0, 0.0
            core_sig.append(0.0)
            
        # Swing System
        if core_sig[-1] == 0:
            if swing_exposure < 0 and swing_entry_price > 0 and price > swing_entry_price * 1.05:
                swing_exposure, swing_entry_price = 0.0, 0.0
            
            if bear and q:
                if price < ema_10:
                    if swing_exposure == 0.0 and rsi > 70:
                        swing_exposure = -0.33 * size_mult
                        swing_entry_price = price
                    elif swing_exposure == -0.33 * size_mult and rsi > 80:
                        swing_exposure = -0.66 * size_mult
                        swing_entry_price = (swing_entry_price + price) / 2
                    elif swing_exposure <= -0.33 * size_mult and swing_exposure > -1.0 * size_mult and rsi > 90:
                        swing_exposure = -1.0 * size_mult
                        swing_entry_price = (swing_entry_price + price) / 2
                if rsi < 40 and swing_exposure < 0:
                    swing_exposure, swing_entry_price = 0.0, 0.0
            else:
                 swing_exposure, swing_entry_price = 0.0, 0.0
            swing_sig.append(swing_exposure)
        else:
            swing_exposure, swing_entry_price = 0.0, 0.0
            swing_sig.append(0.0)

    df['Core_Sig'] = core_sig
    df['Swing_Sig'] = swing_sig
    df['Core_Sig'] = df['Core_Sig'].shift(1).fillna(0)
    df['Swing_Sig'] = df['Swing_Sig'].shift(1).fillna(0)
    
    df['Ret_3x'] = df['Ret'] * 3
    df['Ret_m3x'] = df['Ret'] * -3
    df['Ret_1x'] = df['Ret']
    
    df['Core_Ret'] = np.where(df['Core_Sig'] > 0, df['Ret_3x'] * df['Core_Sig'], np.where(df['Core_Sig'] < 0, df['Ret_m3x'] * abs(df['Core_Sig']), 0))
    df['Swing_Ret'] = np.where(df['Swing_Sig'] > 0, df['Ret_1x'] * df['Swing_Sig'], np.where(df['Swing_Sig'] < 0, df['Ret_1x'] * df['Swing_Sig'], 0))
    df['Total_Strat_Ret'] = df['Core_Ret'] + df['Swing_Ret']
    
    last = df.iloc[-1]
    
    if last['Core_Sig'] > 0: signal, leverage = "CORE LONG", last['Core_Sig'] * 3.0
    elif last['Core_Sig'] < 0: signal, leverage = "CORE SHORT", last['Core_Sig'] * 3.0
    elif last['Swing_Sig'] < 0: signal, leverage = "SWING SHORT", last['Swing_Sig'] * 1.0
    else: signal, leverage = "CASH (Regime Filtered)", 0.0
        
    return df, last, signal, leverage, core_entry_price, swing_entry_price

# --- 3. UI RENDERING ---
st.title(f"Systematic 4-Quadrant Strategy ({display_currency})")

df_market = load_data()
df_strat, latest_data, current_signal, target_leverage, c_entry, s_entry = run_strategy(df_market)

# Format pricing to requested display currency via INR normalization block
usd_close = latest_data['Close']
norm_close_inr = usd_close * usd_to_inr_rate
display_price = norm_close_inr * conversion_factor

avg_entry = c_entry if c_entry > 0 else s_entry
display_entry = ((avg_entry * usd_to_inr_rate) * conversion_factor) if avg_entry > 0 else 0.0

st.header("1. Daily Execution Command")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric(label="Action Signal", value=current_signal)
with col2: st.metric(label="Target Leverage", value=f"{target_leverage:.2f}x")
with col3: st.metric(label=f"Nasdaq 100 ({display_currency})", value=f"{sym}{display_price:,.2f}")
with col4: st.metric(label=f"Avg Entry Price ({display_currency})", value=f"{sym}{display_entry:,.2f}" if display_entry > 0 else "N/A")

st.markdown("---")
st.header("2. Regime Diagnostics")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="Macro Trend (60/230)", value="Bullish" if latest_data['Bull'] else "Bearish")
c2.metric(label="Volatility Rank", value=f"{latest_data['HV_Rank']*100:.1f}%")
c3.metric(label="2-Day RSI", value=f"{latest_data['RSI_2']:.1f}")
c4.metric(label="ATR Size Multiplier", value=f"{latest_data['Size_Multiplier']:.2f}")

st.markdown("---")
st.header("3. Historical Backtest Results")
cum_total = (1 + df_strat['Total_Strat_Ret']).cumprod()
years_len = len(df_strat) / 252.0
cagr = (cum_total.iloc[-1]) ** (1 / years_len) - 1
vol = df_strat['Total_Strat_Ret'].std() * np.sqrt(252)
sharpe = (df_strat['Total_Strat_Ret'].mean() * 252) / vol
max_dd = (cum_total / cum_total.cummax() - 1).min()

c1, c2, c3, c4 = st.columns(4)
c1.metric(label="System CAGR", value=f"{cagr*100:.2f}%")
c2.metric(label="Max Drawdown", value=f"{max_dd*100:.2f}%")
c3.metric(label="Sharpe Ratio", value=f"{sharpe:.2f}")
c4.metric(label="Annualized Vol", value=f"{vol*100:.2f}%")

st.subheader("Yearly Performance Table")
years = df_strat['Date'].dt.year.unique()
yearly_stats = []
for yr in sorted(years, reverse=True):
    df_yr = df_strat[df_strat['Date'].dt.year == yr]
    strat_perf = (1 + df_yr['Total_Strat_Ret']).prod() - 1
    bh_perf = (1 + df_yr['Ret']).prod() - 1
    yearly_stats.append({
        'Year': yr,
        'Strategy Return': f"{strat_perf*100:.2f}%",
        'Benchmark (1x) Return': f"{bh_perf*100:.2f}%"
    })
st.dataframe(pd.DataFrame(yearly_stats), use_container_width=True)

st.markdown("---")
st.header("4. Trade Log & Portfolio Tracker")

# Manual Trade Input Form
with st.form("trade_entry"):
    st.subheader("Log New Trade Execution")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trade_date = st.date_input("Date", datetime.today())
    with col2:
        trade_action = st.selectbox("Action", ["LONG", "SHORT", "COVER/SELL"])
    with col3:
        trade_size = st.number_input("Position Size", min_value=0.01, step=0.1)
    with col4:
        trade_entry = st.number_input("Entry Price (USD)", min_value=0.01, step=0.1)
    
    submitted = st.form_submit_button("Log Trade")
    if submitted:
        st.session_state.portfolios['Live Strategy Execution'].append({
            'Date': trade_date.strftime('%Y-%m-%d'),
            'Asset': 'QQQ3.L',
            'Action': trade_action,
            'Size': trade_size,
            'Entry Price': trade_entry,
            'LTP': trade_entry # Defaults to entry upon logging
        })
        st.success("Trade logged successfully.")

# Display Portfolios
for port_name, holdings in st.session_state.portfolios.items():
    st.subheader(f"Portfolio: {port_name}")
    if len(holdings) > 0:
        df_port = pd.DataFrame(holdings)
        
        # Apply currency normalization and exact formatting rules
        df_port['Entry Price'] = (df_port['Entry Price'] * usd_to_inr_rate) * conversion_factor
        df_port['LTP'] = (df_port['LTP'] * usd_to_inr_rate) * conversion_factor
        
        # Format the LTP column to exactly 4 decimal places
        styled_port = df_port.style.format({
            'Entry Price': f"{sym}{{:.2f}}",
            'LTP': f"{sym}{{:.4f}}"
        })
        st.table(styled_port)
    else:
        st.write("No active trades logged.")
        
# Starting Capital Tracker
st.markdown(f"**Working Capital Base:** {sym}{((st.session_state.cash_usd * usd_to_inr_rate) * conversion_factor):,.2f} ({display_currency})")

```


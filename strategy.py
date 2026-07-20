import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="Quant Trading System", layout="wide")

if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {
        'Live Strategy Execution': [],
        'Madhur (acc2)': [
            {'Date': '2026-07-10', 'Asset': 'XRP', 'Action': 'LONG', 'Size': 10000, 'Entry Price': 0.4200, 'LTP': 0.4512}
        ]
    }
    st.session_state.cash_usd = 20000.0

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
            
        stop_triggered = False

        # --- CORE SYSTEM ---
        if bull and q:
            short_exposure = 0.0
            core_entry_price = 0.0
            
            # Dynamic Momentum Floor (Replaces broken fixed stop/lockout)
            if price > ema_10:
                core_sig.append(1.0 * size_mult)
            else:
                core_sig.append(0.0) # Step aside to cash on momentum breaks
                
        elif bear and v:
            # 5% Hard stop on underlying index for the Short Position
            if short_exposure < 0 and core_entry_price > 0 and price > core_entry_price * 1.05:
                short_exposure = 0.0
                core_entry_price = 0.0
                stop_triggered = True
                
            if short_exposure < 0 and rsi < 50:
                short_exposure = 0.0
                core_entry_price = 0.0
                
            if not stop_triggered:
                if short_exposure == 0.0 and rsi > 75:
                    short_exposure = -0.33 * size_mult
                    core_entry_price = price
                elif short_exposure == -0.33 * size_mult and rsi > 80:
                    short_exposure = -0.66 * size_mult
                    core_entry_price = (core_entry_price + price) / 2
                elif short_exposure >= -0.66 * size_mult and short_exposure < 0 and rsi > 85:
                    short_exposure = -1.0 * size_mult
                    core_entry_price = (core_entry_price + price) / 2
                    
            core_sig.append(short_exposure)
        else:
            short_exposure = 0.0
            core_entry_price = 0.0
            core_sig.append(0.0)
            
        # --- SWING SYSTEM ---
        if core_sig[-1] == 0:
            if swing_exposure < 0 and swing_entry_price > 0 and price > swing_entry_price * 1.05:
                swing_exposure = 0.0
                swing_entry_price = 0.0
                stop_triggered = True
                
            if swing_exposure < 0 and rsi < 40:
                swing_exposure = 0.0
                swing_entry_price = 0.0
                
            if bear and q and not stop_triggered:
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
            else:
                if not (bear and q):
                    swing_exposure = 0.0
                    swing_entry_price = 0.0
            swing_sig.append(swing_exposure)
        else:
            swing_exposure = 0.0
            swing_entry_price = 0.0
            swing_sig.append(0.0)

    df['Core_Sig'] = core_sig
    df['Swing_Sig'] = swing_sig
    
    df['Core_Sig'] = df['Core_Sig'].shift(1).fillna(0)
    df['Swing_Sig'] = df['Swing_Sig'].shift(1).fillna(0)
    
    # Calculate Plot Triggers
    df['Prev_Core_Sig'] = df['Core_Sig'].shift(1).fillna(0)
    df['Prev_Swing_Sig'] = df['Swing_Sig'].shift(1).fillna(0)
    
    df['Long_Entry'] = np.where((df['Core_Sig'] > 0) & (df['Prev_Core_Sig'] <= 0), df['Close'], np.nan)
    df['Short_Entry'] = np.where(
        ((df['Core_Sig'] < 0) & (df['Prev_Core_Sig'] >= 0)) | 
        ((df['Swing_Sig'] < 0) & (df['Prev_Swing_Sig'] >= 0)), 
        df['Close'], np.nan
    )
    
    # Dynamic trailing stop out points for UI plotting
    df['Stop_Out'] = np.where((df['Prev_Core_Sig'] > 0) & (df['Core_Sig'] == 0) & (df['Bull']) & (df['Quiet']), df['Close'], np.nan)
    
    # FIX: Explicit 1x Returns array for Short routing
    df['Ret_3x'] = df['Ret'] * 3
    df['Ret_1x'] = df['Ret']
    
    # FIX: Core Longs use 3x. Core Shorts strictly use 1x to cap squeeze drawdowns.
    df['Core_Ret'] = np.where(df['Core_Sig'] > 0, df['Ret_3x'] * df['Core_Sig'], np.where(df['Core_Sig'] < 0, df['Ret_1x'] * df['Core_Sig'], 0))
    df['Swing_Ret'] = np.where(df['Swing_Sig'] < 0, df['Ret_1x'] * df['Swing_Sig'], 0)
    
    df['Total_Strat_Ret'] = df['Core_Ret'] + df['Swing_Ret']
    
    last = df.iloc[-1]
    
    if last['Core_Sig'] > 0: signal, leverage = "CORE LONG", last['Core_Sig'] * 3.0
    elif last['Core_Sig'] < 0: signal, leverage = "CORE SHORT", abs(last['Core_Sig']) * 1.0
    elif last['Swing_Sig'] < 0: signal, leverage = "SWING SHORT", abs(last['Swing_Sig']) * 1.0
    else: signal, leverage = "CASH (Regime Filtered)", 0.0
        
    return df, last, signal, leverage, core_entry_price, swing_entry_price

# --- 3. UI RENDERING ---
st.title("Systematic 4-Quadrant Strategy (USD)")

df_market = load_data()
df_strat, latest_data, current_signal, target_leverage, c_entry, s_entry = run_strategy(df_market)

display_price = latest_data['Close']
avg_entry = c_entry if c_entry > 0 else s_entry

current_date_str = latest_data['Date'].strftime('%Y-%m-%d')

# --- SECTION 1 ---
st.header(f"1. Daily Execution Command ({current_date_str})")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric(label="Action Signal", value=current_signal)
with col2: st.metric(label="Target Leverage", value=f"{target_leverage:.2f}x")
with col3: st.metric(label="Nasdaq 100 (Live Data)", value=f"${display_price:,.2f}")
with col4: st.metric(label="Avg Entry Price", value=f"${avg_entry:,.2f}" if avg_entry > 0 else "N/A")

st.markdown("#### Interactive Strategy Map")
df_plot = df_strat.tail(600).copy()

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name='NDX Close', line=dict(color='#1f77b4', width=2)))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['EMA_10'], name='EMA 10 (Momentum Floor)', line=dict(color='rgba(44, 160, 44, 0.6)', width=1.5)))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['EMA_60'], name='EMA 60 (Fast)', line=dict(color='#ff7f0e', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['EMA_230'], name='EMA 230 (Slow)', line=dict(color='#d62728', width=1.5, dash='dot')))


fig.add_trace(go.Scatter(
    x=df_plot['Date'], y=df_plot['Long_Entry'],
    name='Long Entry Trigger',
    mode='markers',
    marker=dict(symbol='triangle-up', size=11, color='#2ca02c', line=dict(width=1, color='black')),
    hovertemplate='<b>Long Entry</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=df_plot['Date'], y=df_plot['Short_Entry'],
    name='Short Entry Trigger',
    mode='markers',
    marker=dict(symbol='triangle-down', size=11, color='#d62728', line=dict(width=1, color='black')),
    hovertemplate='<b>Short Entry</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=df_plot['Date'], y=df_plot['Stop_Out'],
    name='Step-Aside / Stop Out',
    mode='markers',
    marker=dict(symbol='x', size=10, color='#ff7f0e', line=dict(width=1.5, color='black')),
    hovertemplate='<b>Stop / Step-Aside Triggered</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
))

fig.update_layout(
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(title="Date", rangeslider=dict(visible=True)),
    yaxis=dict(title="Nasdaq 100 Price (USD)", side="right"),
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=470
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# --- REGIME DIAGNOSTICS ---
st.header("2. Regime Diagnostics")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="Macro Trend (60/230)", value="Bullish" if latest_data['Bull'] else "Bearish")
c2.metric(label="Volatility Rank", value=f"{latest_data['HV_Rank']*100:.1f}%")
c3.metric(label="2-Day RSI", value=f"{latest_data['RSI_2']:.1f}")
c4.metric(label="ATR Size Multiplier", value=f"{latest_data['Size_Multiplier']:.2f}")

st.markdown("---")

# --- TOMORROW'S TRIGGER CALCULATOR ---
st.header("3. Tomorrow's Trading Triggers")

c_today = latest_data['Close']
c_yesterday = df_strat.iloc[-2]['Close']
d_today = c_today - c_yesterday

def get_rsi_target(target_rsi):
    r = target_rsi / (100 - target_rsi)
    if target_rsi >= 50:
        d_next = r * max(-d_today, 0) - max(d_today, 0)
    else:
        d_next = max(-d_today, 0) - max(d_today, 0) / r
    return max(0.0, c_today + d_next)

alpha_60, alpha_230 = 2 / 61, 2 / 231
ema60_today, ema230_today = latest_data['EMA_60'], latest_data['EMA_230']
trend_flip_tomorrow = ((1 - alpha_230) * ema230_today - (1 - alpha_60) * ema60_today) / (alpha_60 - alpha_230)

st.markdown("*(These are the exact mathematical price levels or conditions required to trigger a new trade execution or state change in tomorrow's session).*")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.subheader("Macro Trend & Exits")
    if trend_flip_tomorrow > 0 and trend_flip_tomorrow < (c_today * 2):
        st.write(f"**Macro Trend Flip Level:** ${trend_flip_tomorrow:,.2f}")
    else:
        st.write("**Macro Trend Flip Level:** Not mathematically possible in 1 day.")
        
    if "CORE SHORT" in current_signal:
        st.write(f"**Take Profit (RSI < 50):** ${get_rsi_target(50):,.2f} or lower")
        st.write(f"**Hard Stop Loss (5%):** ${avg_entry * 1.05:,.2f} or higher")
    elif "SWING SHORT" in current_signal:
        st.write(f"**Take Profit (RSI < 40):** ${get_rsi_target(40):,.2f} or lower")
        st.write(f"**Hard Stop Loss (5%):** ${avg_entry * 1.05:,.2f} or higher")
    elif "CORE LONG" in current_signal:
        st.write(f"**Trailing Momentum Stop:** Liquidate to cash if Price closes below EMA 10 (**${latest_data['EMA_10']:,.2f}**)")
        st.write("**Volatility Exit:** Liquidate to cash if Volatility Rank spikes ≥ 85%.")
    else:
        st.write("**Exits:** Currently in Cash. No active stop-losses.")

with col_t2:
    st.subheader("Pending Entry Conditions")
    
    if "CASH" in current_signal:
        if latest_data['Bull'] and latest_data['Volatile']:
            st.write("**Waiting for Volatility to Subside or Trend to Flip.**")
            st.write(f"- **To go CORE LONG:** Volatility Rank must drop below 85% (Current: {latest_data['HV_Rank']*100:.1f}%).")
            st.write("- **To go SHORT:** Macro Trend (EMA 60) must cross below EMA 230.")
        elif latest_data['Bull'] and latest_data['Quiet']:
            st.write("**Waiting for Momentum Recovery.**")
            st.write(f"- **To re-enter CORE LONG:** Price must close above EMA 10 (**${latest_data['EMA_10']:,.2f}**)")
        else:
            st.write("**Waiting for an RSI Overbought Spike.**")
            if latest_data['Volatile']:
                st.write(f"- **To enter CORE SHORT (Stage 1):** RSI > 75 (Target: **${get_rsi_target(75):,.2f}**)")
            else:
                st.write(f"- **To enter SWING SHORT (Stage 1):** RSI > 70 (Target: **${get_rsi_target(70):,.2f}**)")
                st.write(f"*(Note: Price must also close below EMA 10: ${latest_data['EMA_10']:,.2f})*")
                
    elif "CORE LONG" in current_signal:
        st.write("**Status: Fully Invested.**")
        st.write("No new entry conditions pending. Riding the Bull Quiet trend above EMA 10.")
        
    elif "CORE SHORT" in current_signal:
        st.write("**Waiting for Further RSI Spikes to Scale In:**")
        st.write(f"- **Stage 2 (RSI > 80):** **${get_rsi_target(80):,.2f}**")
        st.write(f"- **Stage 3 (RSI > 85):** **${get_rsi_target(85):,.2f}**")
        
    elif "SWING SHORT" in current_signal:
        st.write("**Waiting for Further RSI Spikes to Scale In:**")
        st.write(f"- **Stage 2 (RSI > 80):** **${get_rsi_target(80):,.2f}**")
        st.write(f"- **Stage 3 (RSI > 90):** **${get_rsi_target(90):,.2f}**")

st.markdown("---")
st.header("4. Historical Backtest Results")

df_metrics = df_strat.dropna(subset=['HV_Rank', 'ATR_Baseline']).copy()
cum_total = (1 + df_metrics['Total_Strat_Ret']).cumprod()
years_len = len(df_metrics) / 252.0
cagr = (cum_total.iloc[-1]) ** (1 / years_len) - 1
vol = df_metrics['Total_Strat_Ret'].std() * np.sqrt(252)
sharpe = (df_metrics['Total_Strat_Ret'].mean() * 252) / (vol + 1e-10)
max_dd = (cum_total / cum_total.cummax() - 1).min()

c1, c2, c3, c4 = st.columns(4)
c1.metric(label="System CAGR", value=f"{cagr*100:.2f}%")
c2.metric(label="Max Drawdown", value=f"{max_dd*100:.2f}%")
c3.metric(label="Sharpe Ratio", value=f"{sharpe:.2f}")
c4.metric(label="Annualized Vol", value=f"{vol*100:.2f}%")

st.subheader("Yearly Performance Table")
years = df_metrics['Date'].dt.year.unique()
yearly_stats = []
for yr in sorted(years, reverse=True):
    df_yr = df_metrics[df_metrics['Date'].dt.year == yr]
    strat_perf = (1 + df_yr['Total_Strat_Ret']).prod() - 1
    bh_perf = (1 + df_yr['Ret']).prod() - 1
    yearly_stats.append({
        'Year': yr,
        'Strategy Return': f"{strat_perf*100:.2f}%",
        'Benchmark (1x) Return': f"{bh_perf*100:.2f}%"
    })
st.dataframe(pd.DataFrame(yearly_stats), use_container_width=True)

st.markdown("---")
st.header("5. Trade Log & Portfolio Tracker")

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
            'Asset': 'QQQ3',
            'Action': trade_action,
            'Size': trade_size,
            'Entry Price': trade_entry,
            'LTP': trade_entry
        })
        st.success("Trade logged successfully.")

for port_name, holdings in st.session_state.portfolios.items():
    st.subheader(f"Portfolio: {port_name}")
    if len(holdings) > 0:
        df_port = pd.DataFrame(holdings)
        styled_port = df_port.style.format({
            'Entry Price': "${:.2f}",
            'LTP': "${:.4f}"
        })
        st.table(styled_port)
    else:
        st.write("No active trades logged.")
        
st.markdown(f"**Working Capital Base:** ${st.session_state.cash_usd:,.2f}")


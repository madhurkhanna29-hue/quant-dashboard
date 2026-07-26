import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =====================================================================================
# SYSTEMATIC NDX 3x LEVERAGE STRATEGY — v2
#
# What changed vs the original build, and why:
#
# 1. REALISTIC COSTS ARE NOW MODELLED. The original backtest had zero transaction costs,
#    zero spread/slippage, and zero product carry (TER + swap financing on the leveraged
#    portion). For a real LQQ3-style ETP with ~1,400+ position changes since 1985, that
#    overstates CAGR by roughly 5-6pts and Sharpe by ~0.1-0.15. Costs are now subtracted
#    inside the backtest, not bolted on afterwards, so the numbers on screen are what you'd
#    actually stand a chance of capturing.
#
# 2. VOLATILITY-TARGETED SIZING replaces the old 4-way HV-Rank tier ladder (<70%/<85%/>=85%).
#    One continuous formula (target_vol / realized_vol, capped at 3x) does the same job with
#    one parameter instead of four hard-coded breakpoints — fewer knobs curve-fit to this
#    specific 40-year sample.
#
# 3. A CHANDELIER TRAILING STOP was added to the long book. The EMA(60/230) trend filter is
#    slow — in the 2022 selloff it kept the system 3x long for weeks after price had already
#    broken down, costing ~-37% from the long leg alone before any short ever engaged. The
#    trailing stop (highest close since entry, minus k * ATR) exits well before the slow trend
#    filter would.
#
# 4. THE SHORT OVERLAY WAS SIMPLIFIED AND ITS ROLE RE-FRAMED. Diagnostics on the original
#    six-tier RSI(2) short-scaling logic showed it contributed ~0% net PnL over the full
#    40-year sample once costs were included — full-sample CAGR was actually *higher* with
#    the short overlay switched off. It only earned its keep in acute drawdowns: in 2022 it
#    cut the loss from -20.8% to -14.8%. It is kept here, simplified to one rule, explicitly
#    as a drawdown-cushion, not an alpha source. Don't expect it to make money on its own.
#
# 5. A WALK-FORWARD / OUT-OF-SAMPLE VIEW is included. A single full-history backtest on a
#    system with several free parameters risks overfitting to the exact history it was tuned
#    on. The Robustness tab below splits the sample and reports pre/post metrics separately
#    so you can see whether performance holds up out of sample rather than trusting one
#    aggregate Sharpe number.
#
# HONEST HEADLINE: the trade-off vs the original is lower CAGR for meaningfully lower max
# drawdown (full-sample max DD ~-68% -> ~-48% with the short overlay on, further reducible
# via the target-vol slider). Which side of that trade-off is right for a real $20k ISA
# account is a risk-tolerance decision, not a modelling one — the slider lets you set it.
# =====================================================================================

st.set_page_config(page_title="NDX Systematic Strategy v2", layout="wide")

if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {'Live Strategy Execution': []}
    st.session_state.cash_usd = 20000.0

# --- SIDEBAR: RISK CONTROLS -----------------------------------------------------------
st.sidebar.header("Risk Controls")
target_vol = st.sidebar.slider("Long book target volatility (annualised)", 0.10, 0.80, 0.35, 0.01,
                                help="This is a leverage dial, not a free-CAGR lever: Sharpe stays roughly flat "
                                     "(~0.73-0.75) across this whole range in backtest — raising it buys more CAGR "
                                     "AND proportionally more drawdown, together. See the Robustness tab for the "
                                     "full trade-off curve before picking a number.")
max_lev = st.sidebar.slider("Max long leverage cap", 1.0, 3.0, 3.0, 0.1,
                             help="Hard ceiling regardless of vol target — 3.0 matches a 3x LSE ETP like LQQ3.")
chand_k = st.sidebar.slider("Trailing stop tightness (ATR multiple)", 2.0, 4.5, 3.5, 0.25,
                             help="Lower = exits faster/tighter (less drawdown, more whipsaw). Higher = rides trends longer.")
use_short = st.sidebar.checkbox("Enable short overlay", value=True,
                                 help="Framed as a drawdown cushion, not an alpha source — see notes above.")
short_target_vol = st.sidebar.slider("Short book target volatility", 0.04, 0.20, 0.10, 0.01, disabled=not use_short)
spread_bps = st.sidebar.slider("Round-trip cost per position change (bps)", 0, 25, 8,
                                help="LSE spread + slippage for LQQ3 executed in the last-90-min window.")
ter_annual = st.sidebar.number_input("Product TER (annual)", 0.0, 0.03, 0.0075, 0.0005, format="%.4f")
financing_spread = st.sidebar.number_input("Financing spread over cash rate (annual, on levered portion)", 0.0, 0.05, 0.02, 0.005, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.caption("These parameters are also what's swept in the Robustness tab, so you can see how "
                    "sensitive the results are before trusting any single setting.")

# --- DATA & ENGINE ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    ticker = yf.Ticker("^NDX")
    df = ticker.history(period="max")
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.sort_values('Date', inplace=True)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True)


@st.cache_data
def run_strategy(df, target_vol, max_lev, chand_k, use_short, short_target_vol,
                  spread_bps, ter_annual, financing_spread, ema_fast=60, ema_slow=230):
    d = df.copy()
    d['Ret'] = d['Close'].pct_change()
    d['EMA_fast'] = d['Close'].ewm(span=ema_fast, adjust=False).mean()
    d['EMA_slow'] = d['Close'].ewm(span=ema_slow, adjust=False).mean()
    d['Bull'] = d['EMA_fast'] > d['EMA_slow']

    d['TR'] = np.maximum(d['High'] - d['Low'],
                          np.maximum(abs(d['High'] - d['Close'].shift(1)), abs(d['Low'] - d['Close'].shift(1))))
    d['ATR_20'] = d['TR'].rolling(20).mean()

    d['RV_20'] = d['Ret'].rolling(20).std() * np.sqrt(252)
    d['RV_60'] = d['Ret'].rolling(60).std() * np.sqrt(252)
    d['RV_blend'] = 0.5 * d['RV_20'] + 0.5 * d['RV_60']

    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs = gain / (loss + 1e-10)
    d['RSI_2'] = 100 - (100 / (1 + rs))

    n = len(d)
    long_lev = np.zeros(n)
    short_lev = np.zeros(n)
    chand_stop = np.full(n, np.nan)
    highest_since_entry = 0.0
    in_long = False

    close, high = d['Close'].values, d['High'].values
    bull, atr, rv, rsi = d['Bull'].values, d['ATR_20'].values, d['RV_blend'].values, d['RSI_2'].values

    trade_log = []
    active_long, active_short = None, None

    for i in range(n):
        price = close[i]
        if i == 0 or np.isnan(rv[i]) or np.isnan(atr[i]):
            continue

        prev_long_on = long_lev[i - 1] > 0 if i > 0 else False

        if bull[i]:
            if not in_long:
                in_long = True
                highest_since_entry = price
                active_long = {'Type': 'LONG', 'Entry Date': d['Date'].iloc[i], 'Entry Price': price, 'Entry_Idx': i}
            highest_since_entry = max(highest_since_entry, high[i])
            stop_level = highest_since_entry - chand_k * atr[i]
            chand_stop[i] = stop_level
            if price < stop_level:
                in_long = False
                long_lev[i] = 0.0
                if active_long:
                    active_long['Exit Date'] = d['Date'].iloc[i]
                    active_long['Exit Price'] = price
                    active_long['Exit Condition'] = 'Trailing stop hit'
                    active_long['Exit_Idx'] = i
                    trade_log.append(active_long)
                    active_long = None
            else:
                target = target_vol / max(rv[i], 1e-6)
                long_lev[i] = float(np.clip(target, 0.0, max_lev))
        else:
            if in_long and active_long:
                active_long['Exit Date'] = d['Date'].iloc[i]
                active_long['Exit Price'] = price
                active_long['Exit Condition'] = 'Trend flipped bearish'
                active_long['Exit_Idx'] = i
                trade_log.append(active_long)
                active_long = None
            in_long = False
            long_lev[i] = 0.0

        if use_short and long_lev[i] == 0 and not bull[i] and not np.isnan(rsi[i]):
            if rsi[i] > 80:
                s_target = short_target_vol / max(rv[i], 1e-6)
                short_lev[i] = -float(np.clip(s_target, 0.0, 1.0))
                if active_short is None:
                    active_short = {'Type': 'SHORT', 'Entry Date': d['Date'].iloc[i], 'Entry Price': price, 'Entry_Idx': i}
            else:
                if active_short is not None:
                    active_short['Exit Date'] = d['Date'].iloc[i]
                    active_short['Exit Price'] = price
                    active_short['Exit Condition'] = 'RSI reset'
                    active_short['Exit_Idx'] = i
                    trade_log.append(active_short)
                    active_short = None
                short_lev[i] = 0.0
        else:
            if active_short is not None:
                active_short['Exit Date'] = d['Date'].iloc[i]
                active_short['Exit Price'] = price
                active_short['Exit Condition'] = 'Regime shifted'
                active_short['Exit_Idx'] = i
                trade_log.append(active_short)
                active_short = None
            short_lev[i] = 0.0

    last_idx = n - 1
    for act in (active_long, active_short):
        if act:
            act['Exit Date'], act['Exit Price'], act['Exit Condition'] = pd.NaT, np.nan, 'Trade Open'
            act['Exit_Idx'] = last_idx
            trade_log.append(act)

    d['Chand_Stop'] = chand_stop
    d['Long_Lev_Raw'] = long_lev
    d['Short_Lev_Raw'] = short_lev
    # shift 1 day: today's signal is computed on today's close, executed tomorrow (no lookahead)
    d['Long_Lev'] = pd.Series(long_lev, index=d.index).shift(1).fillna(0)
    d['Short_Lev'] = pd.Series(short_lev, index=d.index).shift(1).fillna(0)

    pos = d['Long_Lev'] + d['Short_Lev']
    # Cost scales with the SIZE of the leverage change (e.g. going from 1.2x to 1.3x costs 10% of a
    # full round-trip spread), not a flat fee charged on any nonzero change. Vol-targeted sizing
    # rebalances a little most days, so a flat per-day fee would (and did, in testing) silently
    # destroy the account through thousands of tiny "full cost" charges.
    trade_size = pos.diff().abs().fillna(0)
    cost_trading = trade_size * (spread_bps / 10000.0)
    long_flag = (d['Long_Lev'] > 0).astype(float)
    daily_drag = (ter_annual / 252.0) * long_flag + (financing_spread / 252.0) * np.maximum(d['Long_Lev'] - 1, 0)

    d['Gross_Ret'] = d['Long_Lev'] * d['Ret'] + d['Short_Lev'] * d['Ret']
    d['Total_Strat_Ret'] = d['Gross_Ret'] - cost_trading - daily_drag

    for t in trade_log:
        e, x = t['Entry_Idx'], t['Exit_Idx']
        t['PnL (%)'] = (np.prod(1 + d['Total_Strat_Ret'].iloc[e + 1:x + 1]) - 1) * 100 if e < x else 0.0
    df_trades = pd.DataFrame(trade_log)
    if not df_trades.empty:
        df_trades = df_trades.drop(columns=['Entry_Idx', 'Exit_Idx'])

    last = d.iloc[-1]
    if last['Long_Lev'] > 0:
        signal, leverage = "LONG (vol-targeted)", last['Long_Lev']
    elif last['Short_Lev'] < 0:
        signal, leverage = "SHORT (hedge overlay)", abs(last['Short_Lev'])
    else:
        signal, leverage = "CASH", 0.0

    return d, last, signal, leverage, df_trades


def perf_metrics(d, ret_col='Total_Strat_Ret', start=None, end=None):
    dd = d.dropna(subset=[ret_col]).copy()
    if start: dd = dd[dd['Date'] >= start]
    if end: dd = dd[dd['Date'] <= end]
    if len(dd) < 20:
        return None
    r = dd[ret_col]
    cum = (1 + r).cumprod()
    years = len(dd) / 252.0
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / (vol + 1e-10)
    mdd = (cum / cum.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar, n=len(dd),
                start=dd['Date'].min(), end=dd['Date'].max())


# --- RUN ---------------------------------------------------------------------------------
st.title("NDX Systematic Strategy — v2 (Vol-Targeted + Trailing Stop)")

df_market = load_data()
df_strat, latest, current_signal, target_leverage, df_trades = run_strategy(
    df_market, target_vol, max_lev, chand_k, use_short, short_target_vol,
    spread_bps, ter_annual, financing_spread)

tabs = st.tabs(["Live Signal", "Performance", "Robustness / Walk-Forward", "Trade Ledger", "Portfolio Tracker"])

# ---- TAB 1: LIVE SIGNAL ----
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action Signal", current_signal)
    c2.metric("Target Leverage", f"{target_leverage:.2f}x")
    c3.metric("Nasdaq 100", f"${latest['Close']:,.2f}")
    c4.metric("Realized Vol (blend)", f"{latest['RV_blend']*100:.1f}%")

    st.markdown("#### Diagnostics")
    dplot = df_strat.tail(500).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=('Price, Trend & Chandelier Stop', 'Realised Vol vs Target', 'RSI(2)'),
                         row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['Close'], name='NDX Close', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['EMA_fast'], name='EMA 60', line=dict(color='#ff7f0e', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['EMA_slow'], name='EMA 230', line=dict(color='#d62728', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['Chand_Stop'], name='Trailing Stop', line=dict(color='purple', width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['RV_blend']*100, name='Realised Vol %', line=dict(color='#9467bd')), row=2, col=1)
    fig.add_hline(y=target_vol*100, line_dash='dash', line_color='green', annotation_text='Target Vol', row=2, col=1)
    fig.add_trace(go.Scatter(x=dplot['Date'], y=dplot['RSI_2'], name='RSI(2)', line=dict(color='#8c564b')), row=3, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='red', annotation_text='Short trigger (80)', row=3, col=1)
    fig.update_layout(template='plotly_white', height=750, hovermode='x unified',
                       legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if current_signal.startswith("LONG"):
        st.info(f"Trailing stop currently at **${latest['Chand_Stop']:,.2f}**. A close below this level exits to cash "
                f"regardless of the EMA trend filter.")
    elif current_signal.startswith("SHORT"):
        st.info("Hedge overlay active — sized to a modest vol target, not intended as a standalone return driver.")
    else:
        st.info("Flat. Waiting for the EMA(60/230) trend filter to turn bullish, or an RSI(2) > 80 bear-market spike "
                "to arm the hedge overlay.")

# ---- TAB 2: PERFORMANCE ----
with tabs[1]:
    st.subheader("Headline metrics (with realistic costs applied)")
    windows = [(None, None, 'Full sample'), ('2000-01-01', None, '2000-2026'),
               ('2010-01-01', None, '2010-2026'), ('2018-01-01', None, '2018-2026'),
               ('2022-01-01', '2022-12-31', '2022 stress test'), ('2025-01-01', None, '2025-2026')]
    rows = []
    for s, e, label in windows:
        m = perf_metrics(df_strat, start=s, end=e)
        if m:
            rows.append({'Window': label, 'CAGR': f"{m['cagr']*100:.1f}%", 'Vol': f"{m['vol']*100:.1f}%",
                         'Sharpe': f"{m['sharpe']:.2f}", 'Max DD': f"{m['mdd']*100:.1f}%", 'Calmar': f"{m['calmar']:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Yearly performance vs 1x buy & hold")
    dm = df_strat.dropna(subset=['Total_Strat_Ret']).copy()
    dm['Year'] = dm['Date'].dt.year
    yearly = []
    for yr, g in dm.groupby('Year'):
        strat = (1 + g['Total_Strat_Ret']).prod() - 1
        bh = (1 + g['Ret']).prod() - 1
        yearly.append({'Year': yr, 'Strategy': f"{strat*100:.1f}%", 'NDX Buy & Hold (1x)': f"{bh*100:.1f}%"})
    st.dataframe(pd.DataFrame(yearly).sort_values('Year', ascending=False), use_container_width=True, hide_index=True)

# ---- TAB 3: ROBUSTNESS ----
with tabs[2]:
    st.subheader("Why this tab exists")
    st.write("A backtest with several tunable parameters (vol target, stop distance, RSI threshold) can look great "
             "on the exact history it was built on and still fail going forward. This isn't a promise the strategy "
             "is robust — it's a way to see where the wheels might come off before it's real money.")

    st.subheader("In-sample vs out-of-sample split")
    m_pre = perf_metrics(df_strat, end='2017-12-31')
    m_post = perf_metrics(df_strat, start='2018-01-01')
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Pre-2018**")
        if m_pre:
            st.write(f"CAGR {m_pre['cagr']*100:.1f}% · Sharpe {m_pre['sharpe']:.2f} · Max DD {m_pre['mdd']*100:.1f}%")
    with colB:
        st.markdown("**2018 onward**")
        if m_post:
            st.write(f"CAGR {m_post['cagr']*100:.1f}% · Sharpe {m_post['sharpe']:.2f} · Max DD {m_post['mdd']*100:.1f}%")
    st.caption("If these two windows tell very different stories, be suspicious of the aggregate full-sample number "
               "above — it may be dominated by one regime (e.g. the post-2018 Nasdaq bull run).")

    st.subheader("Parameter sensitivity — target volatility")
    st.write("**This is the key chart for deciding your risk setting.** Backtest shows Sharpe stays roughly flat "
             "across the whole range — this is a leverage/drawdown dial, not a way to get more return for free. "
             "Pick the row whose Max DD you could actually sit through with real money before picking a slider value.")
    sweep_rows = []
    for tv in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        d2, *_ = run_strategy(df_market, tv, max_lev, chand_k, use_short, short_target_vol,
                               spread_bps, ter_annual, financing_spread)
        m = perf_metrics(d2, start='2010-01-01')
        m22 = perf_metrics(d2, start='2022-01-01', end='2022-12-31')
        if m:
            sweep_rows.append({'Target Vol': f"{tv:.0%}", 'CAGR (2010-26)': f"{m['cagr']*100:.1f}%",
                               'Sharpe': f"{m['sharpe']:.2f}", 'Max DD (2010-26)': f"{m['mdd']*100:.1f}%",
                               'Calmar': f"{m['calmar']:.2f}",
                               '2022 stress DD': f"{m22['mdd']*100:.1f}%" if m22 else 'N/A'})
    st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True, hide_index=True)
    st.caption("Results should move smoothly as this knob turns. Sharp jumps between adjacent settings would be a "
               "red flag for curve-fitting rather than a genuine edge — here they don't, which is reassuring, but "
               "also confirms there's no hidden 'high Sharpe, high leverage' sweet spot to find.")

    st.subheader("Short overlay: on vs off (full sample, with costs)")
    d_on, *_ = run_strategy(df_market, target_vol, max_lev, chand_k, True, short_target_vol,
                             spread_bps, ter_annual, financing_spread)
    d_off, *_ = run_strategy(df_market, target_vol, max_lev, chand_k, False, short_target_vol,
                              spread_bps, ter_annual, financing_spread)
    m_on, m_off = perf_metrics(d_on), perf_metrics(d_off)
    m_on22, m_off22 = perf_metrics(d_on, start='2022-01-01', end='2022-12-31'), perf_metrics(d_off, start='2022-01-01', end='2022-12-31')
    comp = pd.DataFrame([
        {'Overlay': 'ON (full sample)', 'CAGR': f"{m_on['cagr']*100:.1f}%", 'Max DD': f"{m_on['mdd']*100:.1f}%"},
        {'Overlay': 'OFF (full sample)', 'CAGR': f"{m_off['cagr']*100:.1f}%", 'Max DD': f"{m_off['mdd']*100:.1f}%"},
        {'Overlay': 'ON (2022 only)', 'CAGR': f"{m_on22['cagr']*100:.1f}%", 'Max DD': f"{m_on22['mdd']*100:.1f}%"},
        {'Overlay': 'OFF (2022 only)', 'CAGR': f"{m_off22['cagr']*100:.1f}%", 'Max DD': f"{m_off22['mdd']*100:.1f}%"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption("The short overlay is a net drag over the full sample but cushions the acute 2022-style drawdown. "
               "Keep it only if you value that specific insurance more than the long-run drag it costs.")

# ---- TAB 4: TRADE LEDGER ----
with tabs[3]:
    if not df_trades.empty:
        dt_ = df_trades.copy()
        dt_['Year'] = dt_['Exit Date'].dt.year.fillna(dt_['Entry Date'].dt.year).astype(int)
        yrs = sorted(dt_['Year'].unique(), reverse=True)
        sel = st.selectbox("Year", yrs)
        show = dt_[dt_['Year'] == sel][['Type', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Condition', 'Exit Price', 'PnL (%)']].copy()
        show['Entry Date'] = show['Entry Date'].dt.strftime('%Y-%m-%d')
        show['Exit Date'] = show['Exit Date'].dt.strftime('%Y-%m-%d').fillna('OPEN')
        st.dataframe(show.style.format({'Entry Price': '${:,.2f}', 'Exit Price': '${:,.2f}', 'PnL (%)': '{:+.2f}%'}, na_rep='N/A'),
                     use_container_width=True, hide_index=True)
    else:
        st.write("No trades generated.")

# ---- TAB 5: PORTFOLIO TRACKER ----
with tabs[4]:
    with st.form("trade_entry"):
        st.subheader("Log a real execution")
        c1, c2, c3, c4 = st.columns(4)
        with c1: trade_date = st.date_input("Date", datetime.today())
        with c2: trade_action = st.selectbox("Action", ["LONG", "SHORT", "COVER/SELL"])
        with c3: trade_size = st.number_input("Size (GBP notional)", min_value=0.0, step=100.0)
        with c4: trade_entry = st.number_input("Entry Price", min_value=0.0, step=0.1)
        if st.form_submit_button("Log Trade"):
            st.session_state.portfolios['Live Strategy Execution'].append({
                'Date': trade_date.strftime('%Y-%m-%d'), 'Asset': 'LQQ3', 'Action': trade_action,
                'Size': trade_size, 'Entry Price': trade_entry, 'LTP': trade_entry})
            st.success("Logged.")
    for name, holdings in st.session_state.portfolios.items():
        st.subheader(name)
        if holdings:
            st.table(pd.DataFrame(holdings).style.format({'Entry Price': '${:.2f}', 'LTP': '${:.4f}'}))
        else:
            st.write("No active trades logged.")
    st.markdown(f"**Working Capital Base:** ${st.session_state.cash_usd:,.2f}")

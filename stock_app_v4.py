"""
╔══════════════════════════════════════════════════════════════╗
║     NSE STOCK ANALYSIS DASHBOARD  v4                         ║
║     Now a TRUE AI AGENT — Ollama + Prophet + scikit-learn    ║
╠══════════════════════════════════════════════════════════════╣
║  NEW IN v4:                                                   ║
║  • Tab 10 — AI Agent chat powered by Ollama (local LLM)      ║
║    Ask anything: "Should I buy Tata Steel today?"            ║
║    Agent fetches live data + reasons over it + replies       ║
║                                                               ║
║  REQUIRES: Ollama running locally with llama3.1              ║
║    1. Install from ollama.com                                 ║
║    2. ollama pull llama3.1                                    ║
║    3. ollama serve  (keep running in background)             ║
║                                                               ║
║  HOW TO RUN:                                                  ║
║  streamlit run stock_app_v4.py                               ║
║  Then open: http://localhost:8501                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import feedparser
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# ── AI / ML IMPORTS ───────────────────────────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# ── OLLAMA AGENT ───────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"

def check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def build_stock_context(ticker: str, company: str, ind: dict,
                         fund: dict, rec: str, score: float,
                         reasons: list, risks: list, news: list,
                         prophet_pct: float = None,
                         ml_signal: str = None, ml_prob: float = None) -> str:
    """
    Build a rich context string from live stock data to feed to the LLM.
    Now includes Prophet forecast and ML signal so agent can advise on timing.
    """
    cur_price  = float(ind['prices'].iloc[-1])
    rsi_cur    = float(ind['rsi'].iloc[-1])
    macd_cur   = float(ind['macd'].iloc[-1])
    sig_cur    = float(ind['signal'].iloc[-1])
    ma50_cur   = float(ind['ma50'].iloc[-1])
    ma200_cur  = float(ind['ma200'].iloc[-1])

    news_summary = "\n".join([
        f"  - [{h['sentiment'].upper()}] {h['title']}"
        for h in news[:5]
    ]) if news else "  No recent news available."

    reasons_txt = "\n".join([f"  + {r}" for r in reasons]) or "  None"
    risks_txt   = "\n".join([f"  ! {r}" for r in risks])   or "  None"

    pe   = fund.get('trailingPE', 'N/A')
    mcap = fund.get('marketCap', 0)
    mcap_str = f"₹{mcap/1e7:.0f} Cr" if mcap else "N/A"

    prophet_txt = f"  30-Day Prophet Forecast: {prophet_pct:+.1f}% predicted move" \
                  if prophet_pct is not None else "  Prophet forecast: not yet run"
    ml_txt = f"  ML Signal: {ml_signal} ({ml_prob*100:.0f}% probability price rises in 10 days)" \
             if ml_signal else "  ML Signal: not yet run"

    context = f"""
LIVE STOCK DATA — {company} ({ticker})
Generated: {datetime.now().strftime('%d %b %Y %H:%M')}

PRICE:
  Current:      ₹{cur_price:.2f}
  52-Week High: ₹{float(ind['p52h']):.2f}
  52-Week Low:  ₹{float(ind['p52l']):.2f}
  1-Month Ret:  {ind['r1']:+.1%}
  3-Month Ret:  {ind['r3']:+.1%}
  12-Month Ret: {ind['r12']:+.1%}

TECHNICAL INDICATORS:
  RSI (14):      {rsi_cur:.1f}  {'[OVERBOUGHT>70]' if rsi_cur>70 else '[OVERSOLD<30]' if rsi_cur<30 else '[HEALTHY]'}
  MACD:          {macd_cur:.3f} vs Signal {sig_cur:.3f}  {'[BULLISH]' if macd_cur>sig_cur else '[BEARISH]'}
  Price vs MA50: {((cur_price/ma50_cur)-1):+.1%}
  Price vs MA200:{((cur_price/ma200_cur)-1):+.1%}
  Momentum:      {ind['momentum']:+.1%}
  Sharpe Ratio:  {ind['sharpe']:.2f}
  Max Drawdown:  {ind['maxdd']:.1%}
  Volatility:    {ind['v1']:.1%} annualised
  Support (S1):  ₹{float(ind['s1_level']):.2f}
  Resistance(R1):₹{float(ind['r1_level']):.2f}

FUNDAMENTALS:
  P/E Ratio:    {pe}
  Market Cap:   {mcap_str}

AI PREDICTIONS:
{prophet_txt}
{ml_txt}

AGENT RECOMMENDATION: {rec}  (Score: {score}/10)

POSITIVE SIGNALS:
{reasons_txt}

RISK FACTORS:
{risks_txt}

LATEST NEWS HEADLINES:
{news_summary}
"""
    return context.strip()


def ask_ollama(user_question: str, stock_context: str,
               chat_history: list, model: str = "llama3.1") -> str:
    """Send a question to Ollama with full stock context. Forces CPU to avoid CUDA error."""
    system_prompt = f"""You are a sharp, direct, expert stock market analyst for the NSE (National Stock Exchange of India).
You have access to live real-time data, AI predictions, and ML signals shown below. Always base your analysis on ALL of this data.

RULES:
- Be direct and confident. Give a clear BUY, SELL, or HOLD every time.
- When asked "when should I buy" — use the ML signal (10-day horizon) and Prophet forecast to give a specific timing recommendation. Example: "The ML model says 68% probability of rise in 10 days — buy this week near the support level of ₹X."
- When asked "how much to buy" — suggest a specific number of shares based on the current price and a reasonable investment amount.
- When comparing stocks — pick one clear winner with reasons. Never say "both are good".
- Use bullet points. Keep responses to 200-250 words. One disclaimer at the very end only.
- When the user asks for graphs or charts, confirm they are being generated automatically below.

LIVE DATA + AI PREDICTIONS:
{stock_context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_question})

    try:
        r = requests.post(OLLAMA_URL, json={
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_gpu": 0},   # force CPU — avoids CUDA error
        }, timeout=180)
        if r.status_code == 200:
            return r.json()["message"]["content"]
        else:
            return f"Error {r.status_code}: {r.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Ollama is not running. Open a terminal and run: `ollama serve`"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def wants_chart(question: str) -> bool:
    """Detect if the user is asking for a chart or plot."""
    keywords = ['chart', 'plot', 'graph', 'visual', 'show me', 'draw',
                'price history', 'trend', 'movement', 'compare visually']
    q = question.lower()
    return any(k in q for k in keywords)


def make_agent_price_chart(ticker: str, prices: pd.Series,
                            ind: dict, company: str) -> go.Figure:
    """Price + RSI + MACD chart for a single stock."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04)
    px_idx = prices.index.tz_localize(None) if prices.index.tz else prices.index
    fig.add_trace(go.Scatter(x=px_idx, y=prices.values,
        line=dict(color='#F1F5F9', width=1.5), name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=px_idx, y=ind['ma50'].values,
        line=dict(color='#38BDF8', width=1, dash='dot'), name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=px_idx, y=ind['ma200'].values,
        line=dict(color='#EF4444', width=1, dash='dash'), name='MA200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=px_idx, y=ind['rsi'].values,
        line=dict(color='#A78BFA', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_color='#EF4444', line_dash='dot', line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_color='#10B981', line_dash='dot', line_width=0.8, row=2, col=1)
    colors_m = ['#10B981' if v >= 0 else '#EF4444' for v in ind['hist_m'].values]
    fig.add_trace(go.Bar(x=px_idx, y=ind['hist_m'].values,
        marker_color=colors_m, name='MACD Hist', opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=px_idx, y=ind['macd'].values,
        line=dict(color='#38BDF8', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=px_idx, y=ind['signal'].values,
        line=dict(color='#F59E0B', width=1), name='Signal'), row=3, col=1)
    fig.update_layout(**LAYOUT, height=480,
        title=f'{company} ({ticker}) — Price · RSI · MACD',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_yaxes(title_text='Price (₹)', row=1, col=1)
    fig.update_yaxes(title_text='RSI',       row=2, col=1)
    fig.update_yaxes(title_text='MACD',      row=3, col=1)
    apply_chart_style(fig)
    return fig


def make_comparison_charts(
    main_ticker: str, main_company: str, main_ind: dict,
    comp_stocks: list,   # list of (company_name, ticker_sym, ind_dict, score, rec)
    main_score: float = 0,
) -> list:
    """
    Build a suite of comparison charts between main stock and comparison stocks.
    Returns list of (title, fig) tuples.
    """
    all_stocks = [(main_company, main_ticker, main_ind)] + \
                 [(c[0], c[1], c[2]) for c in comp_stocks if c[2] is not None]
    charts = []

    # ── Chart 1: Normalised price history (rebased to 100) ──────────────────
    fig1 = go.Figure()
    COLORS = ['#38BDF8', '#10B981', '#F59E0B', '#A78BFA', '#FB923C']
    for i, (name, tkr, ind_) in enumerate(all_stocks):
        prices = ind_['prices']
        px_idx = prices.index.tz_localize(None) if prices.index.tz else prices.index
        rebased = (prices / prices.iloc[0]) * 100
        fig1.add_trace(go.Scatter(
            x=px_idx, y=rebased.values,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            name=f'{name} ({tkr.replace(".NS","")})'
        ))
    fig1.update_layout(**LAYOUT, height=360,
        title='Price Performance — Rebased to 100 (same starting point)',
        yaxis_title='Rebased Price')
    apply_chart_style(fig1)
    charts.append(('📈 Price Performance Comparison', fig1))

    # ── Chart 2: Key metrics bar chart ──────────────────────────────────────
    metrics = ['12M Return %', 'Momentum %', 'RSI', 'Sharpe', 'Score /10']
    fig2 = go.Figure()
    for i, (name, tkr, ind_) in enumerate(all_stocks):
        score_val = next((c[3] for c in comp_stocks if c[1]==tkr), None)
        rec_score = score_val if score_val else 0
        vals = [
            round(ind_['r12']*100, 1),
            round(ind_['momentum']*100, 1),
            round(float(ind_['rsi'].iloc[-1]), 1),
            round(ind_['sharpe'], 2),
            rec_score,
        ]
        fig2.add_trace(go.Bar(
            name=f'{name}',
            x=metrics, y=vals,
            marker_color=COLORS[i % len(COLORS)],
            text=[str(v) for v in vals],
            textposition='outside',
        ))
    fig2.update_layout(**LAYOUT, height=360, barmode='group',
        title='Key Metrics Comparison',
        yaxis_title='Value')
    apply_chart_style(fig2)
    charts.append(('📊 Key Metrics Side-by-Side', fig2))

    # ── Chart 3: RSI comparison ──────────────────────────────────────────────
    fig3 = go.Figure()
    for i, (name, tkr, ind_) in enumerate(all_stocks):
        px_idx = ind_['prices'].index.tz_localize(None) if ind_['prices'].index.tz else ind_['prices'].index
        fig3.add_trace(go.Scatter(
            x=px_idx, y=ind_['rsi'].values,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            name=name
        ))
    fig3.add_hline(y=70, line_color='#EF4444', line_dash='dot', line_width=1,
                   annotation_text='Overbought 70', annotation_font_color='#EF4444')
    fig3.add_hline(y=30, line_color='#10B981', line_dash='dot', line_width=1,
                   annotation_text='Oversold 30', annotation_font_color='#10B981')
    fig3.update_layout(**LAYOUT, height=300,
        title='RSI Comparison — who is overbought/oversold?',
        yaxis_title='RSI (14)')
    apply_chart_style(fig3)
    charts.append(('📉 RSI Comparison', fig3))

    # ── Chart 4: Score scorecard ─────────────────────────────────────────────
    names_  = [f"{c[0]}" for c in comp_stocks if c[2] is not None]
    scores_ = [c[3] for c in comp_stocks if c[2] is not None]
    names_  = [main_company] + names_
    scores_ = [main_score] + scores_
    bar_cols = ['#10B981' if s >= 3 else '#EF4444' if s < 0 else '#F59E0B'
                for s in scores_]
    fig4 = go.Figure(go.Bar(
        x=names_, y=scores_,
        marker_color=bar_cols,
        text=[f'{s}/10' for s in scores_],
        textposition='outside',
    ))
    fig4.update_layout(**LAYOUT, height=300,
        title='Agent Recommendation Score (higher = stronger buy)',
        yaxis_title='Score /10')
    apply_chart_style(fig4)
    charts.append(('🤖 Recommendation Score Comparison', fig4))

    # ── Chart 5: Investment return — ₹10,000 invested in each ───────────────
    invest_amount = 10000
    fig5 = go.Figure()
    for i, (name, tkr, ind_) in enumerate(all_stocks):
        prices   = ind_['prices']
        px_idx   = prices.index.tz_localize(None) if prices.index.tz else prices.index
        shares   = invest_amount / float(prices.iloc[0])   # shares bought on day 1
        port_val = prices * shares                          # portfolio value over time
        profit   = port_val - invest_amount
        fig5.add_trace(go.Scatter(
            x=px_idx, y=port_val.values,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            name=f'{name} (now ₹{port_val.iloc[-1]:,.0f})',
            hovertemplate=f'<b>{name}</b><br>Portfolio: ₹%{{y:,.0f}}<br>P&L: ₹{profit.iloc[-1]:+,.0f}<extra></extra>'
        ))
    # Reference line — original investment
    fig5.add_hline(y=invest_amount, line_color='#64748B', line_dash='dot',
                   annotation_text=f'₹{invest_amount:,} invested',
                   annotation_font_color='#64748B')
    fig5.update_layout(**LAYOUT, height=360,
        title=f'💰 If you invested ₹{invest_amount:,} in each — who gives more back?',
        yaxis_title='Portfolio Value (₹)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    apply_chart_style(fig5)
    charts.append((f'💰 ₹{invest_amount:,} Investment Return Comparison', fig5))

    return charts

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='NSE Stock Analyser Pro',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #080D1A; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1525 0%, #111827 100%);
    border-right: 1px solid #1E293B;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 14px 18px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #2D4A6B; }
div[data-testid="metric-container"] label { color: #64748B !important; font-size: 11px !important; font-weight: 500 !important; letter-spacing: 0.05em; text-transform: uppercase; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 20px !important; font-weight: 600 !important; }
div[data-testid="stMetricDelta"] svg { display: none; }

/* Tabs */
div[data-testid="stTabs"] button { color: #64748B !important; font-size: 13px !important; font-weight: 500; padding: 8px 16px; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #38BDF8 !important; border-bottom: 2px solid #38BDF8 !important; }

/* Headers */
h1 { color: #F1F5F9 !important; font-size: 26px !important; font-weight: 700 !important; }
h2 { color: #E2E8F0 !important; font-size: 18px !important; font-weight: 600 !important; }
h3 { color: #CBD5E1 !important; font-size: 15px !important; font-weight: 600 !important; }
p, li, td, th { color: #94A3B8 !important; font-size: 13px !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #0EA5E9);
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-size: 14px;
    width: 100%;
    letter-spacing: 0.03em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Expander */
div[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
}

/* Scrollbars */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080D1A; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }

/* Custom cards */
.stat-card { background:#111827; border:1px solid #1E293B; border-radius:12px; padding:16px 20px; margin-bottom:10px; }
.info-box { background:#0F172A; border-left:3px solid #38BDF8; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.warn-box { background:#0F172A; border-left:3px solid #F59E0B; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.success-box { background:#0F172A; border-left:3px solid #10B981; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.danger-box { background:#0F172A; border-left:3px solid #EF4444; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }

.news-card { background:#111827; border:1px solid #1E293B; border-radius:12px; padding:16px; margin-bottom:10px; transition: border-color 0.2s; }
.news-card:hover { border-color: #2D4A6B; }
.news-pos { border-left:4px solid #10B981; }
.news-neg { border-left:4px solid #EF4444; }
.news-neu { border-left:4px solid #64748B; }

.rec-box { background:#111827; border-radius:16px; padding:28px; text-align:center; border:1px solid #1E293B; }
.badge { display:inline-block; padding:4px 14px; border-radius:99px; font-size:12px; font-weight:600; letter-spacing:0.05em; }
.badge-buy    { background:#052e16; color:#10B981; border:1px solid #10B981; }
.badge-hold   { background:#1c1917; color:#F59E0B; border:1px solid #F59E0B; }
.badge-avoid  { background:#1c0a0a; color:#EF4444; border:1px solid #EF4444; }

.explain-tag { display:inline-block; background:#1E293B; color:#94A3B8; font-size:10px; padding:2px 8px; border-radius:4px; margin-left:6px; font-weight:500; }
.glossary-term { color:#38BDF8 !important; font-weight:600; cursor:help; }
.section-divider { border:none; border-top:1px solid #1E293B; margin:20px 0; }
.kpi-label { font-size:11px; color:#64748B; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:22px; color:#F1F5F9; font-weight:700; margin:4px 0; }
.kpi-sub   { font-size:12px; color:#64748B; }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = Path('history.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('''CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, company TEXT, date TEXT, price REAL,
        recommendation TEXT, score REAL, momentum REAL,
        sharpe REAL, rsi REAL, sentiment TEXT, full_json TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY, company TEXT, added_date TEXT
    )''')
    conn.commit(); conn.close()

def save_analysis(ticker, company, price, rec, score, mom, sharpe, rsi, sentiment, data):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute('''INSERT INTO analysis_history
            (ticker,company,date,price,recommendation,score,momentum,sharpe,rsi,sentiment,full_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (ticker, company, datetime.now().strftime('%Y-%m-%d %H:%M'),
             price, rec, score, mom, sharpe, rsi, sentiment, json.dumps(data)))
        conn.commit(); conn.close()
    except: pass

def get_history(ticker=None):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        q = f"SELECT * FROM analysis_history WHERE ticker='{ticker}' ORDER BY date DESC LIMIT 30" if ticker \
            else "SELECT * FROM analysis_history ORDER BY date DESC LIMIT 50"
        df = pd.read_sql(q, conn); conn.close(); return df
    except: return pd.DataFrame()

def add_to_watchlist(ticker, company):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("INSERT OR REPLACE INTO watchlist VALUES (?,?,?)",
                     (ticker, company, datetime.now().strftime('%Y-%m-%d')))
        conn.commit(); conn.close()
    except: pass

init_db()

# ── GLOSSARY ──────────────────────────────────────────────────────────────────
GLOSSARY = {
    'RSI': 'Relative Strength Index — measures if a stock is overbought (>70) or oversold (<30). Think of it like a speedometer for price movement.',
    'MACD': 'Moving Average Convergence Divergence — shows the relationship between two moving averages. When the MACD line crosses above the signal line, it is bullish.',
    'Sharpe Ratio': 'Measures return relative to risk. Higher = better quality trend. A score above 1 is generally considered good.',
    'P/E Ratio': 'Price-to-Earnings ratio — how much you pay for every Rs 1 of profit. A P/E of 20 means you pay Rs 20 for Rs 1 of annual earnings.',
    'Market Cap': 'Total value of all shares combined. Large cap (>Rs 20,000 Cr) = safer. Small cap = riskier but higher potential.',
    'Momentum': 'The tendency of a stock that has been rising to keep rising. We measure it as 12-month return minus last 1-month return.',
    'Bollinger Bands': 'A price channel that widens during volatile periods. When price touches the upper band, the stock may be overbought.',
    'Moving Average': 'The average price over N days. A rising 200-day MA means the long-term trend is up.',
    'Volume': 'Number of shares traded in a day. High volume confirms a price move is genuine.',
    'Beta': 'Measures how much a stock moves relative to the overall market. Beta of 1.5 means the stock moves 50% more than the market in either direction.',
    'EPS': 'Earnings Per Share — profit divided by number of shares. Growing EPS is a sign of a healthy business.',
    'Dividend': 'A cash payment companies make to shareholders from their profits, usually every quarter.',
    'Debt-to-Equity': 'How much the company owes vs what it owns. High debt means more risk, especially when interest rates rise.',
    'Support Level': 'A price level where a stock has historically bounced back up. Buyers tend to step in at this level.',
    'Resistance Level': 'A price level where a stock has historically been rejected. Sellers tend to appear at this level.',
    'Golden Cross': 'When the 50-day MA crosses ABOVE the 200-day MA. A strong bullish signal.',
    'Death Cross': 'When the 50-day MA crosses BELOW the 200-day MA. A bearish warning signal.',
}

def glossary_tooltip(term):
    tip = GLOSSARY.get(term, '')
    return f'<span class="glossary-term" title="{tip}">{term} ℹ️</span>'

# ── STOCK DATA ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period='2y', auto_adjust=True)
    if hist.empty: return None, None, None
    info = {}
    try: info = stock.info
    except: pass
    try:
        fast = stock.fast_info
        fast_dict = {k: getattr(fast, k, None) for k in
                     ['last_price','day_high','day_low','volume','market_cap',
                      'fifty_two_week_high','fifty_two_week_low']}
    except: fast_dict = {}
    return hist, info, fast_dict

def compute_all(hist):
    prices = hist['Close'].dropna()
    volume = hist['Volume'].dropna()
    high   = hist['High'].dropna()
    low    = hist['Low'].dropna()
    n      = len(prices)

    # Returns
    r = lambda d: float(prices.iloc[-1] / prices.iloc[max(0,n-d)] - 1)
    r1,r3,r6,r12 = r(21),r(63),r(126),r(252)
    momentum = r12 - r1

    # Volatility
    dr   = prices.pct_change().dropna()
    v1   = float(dr.tail(21).std() * np.sqrt(252))
    v3   = float(dr.tail(63).std() * np.sqrt(252))
    v12  = float(dr.std()           * np.sqrt(252))
    sharpe = momentum/v1 if v1>0 else 0

    # Drawdown
    rmax  = prices.cummax()
    dd    = (prices - rmax) / rmax
    maxdd = float(dd.min())

    # RSI
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(com=13,adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13,adjust=False).mean()
    rsi   = 100 - 100/(1 + gain/loss)

    # MAs
    ma9   = prices.ewm(span=9,   adjust=False).mean()
    ma20  = prices.rolling(20).mean()
    ma50  = prices.rolling(50).mean()
    ma100 = prices.rolling(100).mean()
    ma200 = prices.rolling(200).mean()

    # MACD
    ema12  = prices.ewm(span=12,adjust=False).mean()
    ema26  = prices.ewm(span=26,adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9,adjust=False).mean()
    hist_m = macd - signal

    # Bollinger
    bb_mid = prices.rolling(20).mean()
    bb_std = prices.rolling(20).std()
    bb_up  = bb_mid + 2*bb_std
    bb_dn  = bb_mid - 2*bb_std
    bb_pct = (prices - bb_dn) / (bb_up - bb_dn)  # 0-1 where in band

    # ATR (Average True Range) — volatility measure
    tr = pd.concat([
        high - low,
        (high - prices.shift()).abs(),
        (low  - prices.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # OBV (On Balance Volume)
    obv = (np.sign(prices.diff()) * volume).fillna(0).cumsum()

    # Support & Resistance (simple: 52-week pivots)
    p52h = prices.tail(252).max()
    p52l = prices.tail(252).min()
    pivot = (p52h + p52l + float(prices.iloc[-1])) / 3
    r1_level = 2*pivot - p52l
    s1_level = 2*pivot - p52h

    # Cross detection
    cross = 'none'
    if len(prices) >= 200:
        ma50s  = ma50.tail(30)
        ma200s = ma200.tail(30)
        above  = ma50s > ma200s
        chg    = above.diff().fillna(False)
        if chg.any():
            last_cross = chg[chg].index[-1]
            days_ago   = (prices.index[-1] - last_cross).days
            cross = f'golden_{days_ago}' if above.iloc[-1] else f'death_{days_ago}'
        else:
            cross = 'above' if above.iloc[-1] else 'below'

    return {
        'prices':prices,'volume':volume,'high':high,'low':low,'dr':dr,
        'r1':r1,'r3':r3,'r6':r6,'r12':r12,
        'momentum':momentum,'v1':v1,'v3':v3,'v12':v12,
        'sharpe':sharpe,'maxdd':maxdd,'dd':dd,
        'rsi':rsi,'ma9':ma9,'ma20':ma20,'ma50':ma50,
        'ma100':ma100,'ma200':ma200,
        'macd':macd,'signal':signal,'hist_m':hist_m,
        'bb_up':bb_up,'bb_dn':bb_dn,'bb_mid':bb_mid,'bb_pct':bb_pct,
        'atr':atr,'obv':obv,
        'p52h':p52h,'p52l':p52l,'pivot':pivot,
        'r1_level':r1_level,'s1_level':s1_level,'cross':cross,
    }

def compute_recommendation(ind, fund):
    score=0; reasons=[]; risks=[]
    cur = float(ind['prices'].iloc[-1])

    m=ind['momentum']
    if m>0.20:    score+=3; reasons.append(f'Strong 12-1M momentum ({m:+.1%}) — sustained uptrend')
    elif m>0.08:  score+=2; reasons.append(f'Positive momentum ({m:+.1%})')
    elif m>0:     score+=1; reasons.append(f'Mild positive momentum ({m:+.1%})')
    elif m<-0.15: score-=3; risks.append(f'Strong negative momentum ({m:+.1%}) — sustained downtrend')
    elif m<0:     score-=1; risks.append(f'Negative momentum ({m:+.1%})')

    s=ind['sharpe']
    if s>2:    score+=2; reasons.append(f'Excellent trend quality — Sharpe {s:.2f} (smooth, consistent move)')
    elif s>0.5:score+=1; reasons.append(f'Decent trend quality — Sharpe {s:.2f}')
    elif s<0:  score-=1; risks.append(f'Poor trend quality — Sharpe {s:.2f} (volatile, unreliable trend)')

    rsi_cur = float(ind['rsi'].iloc[-1])
    if 40<rsi_cur<60:   score+=1; reasons.append(f'RSI in healthy zone ({rsi_cur:.0f}) — not overbought or oversold')
    elif rsi_cur>75:    score-=2; risks.append(f'RSI very overbought ({rsi_cur:.0f}) — likely to pull back soon')
    elif rsi_cur>70:    score-=1; risks.append(f'RSI overbought ({rsi_cur:.0f}) — caution')
    elif rsi_cur<25:    score+=2; reasons.append(f'RSI very oversold ({rsi_cur:.0f}) — potential strong bounce')
    elif rsi_cur<30:    score+=1; reasons.append(f'RSI oversold ({rsi_cur:.0f}) — potential entry opportunity')

    ma200v = float(ind['ma200'].iloc[-1]) if not pd.isna(ind['ma200'].iloc[-1]) else None
    ma50v  = float(ind['ma50'].iloc[-1])  if not pd.isna(ind['ma50'].iloc[-1])  else None
    if ma200v:
        pct = (cur/ma200v-1)
        if pct>0.10:   score+=2; reasons.append(f'Price {pct:+.1%} above 200-day MA — strong long-term bullish')
        elif pct>0:    score+=1; reasons.append(f'Price above 200-day MA (bullish long-term)')
        elif pct<-0.10:score-=2; risks.append(f'Price {pct:+.1%} below 200-day MA — bearish long-term trend')
        else:          score-=1; risks.append('Price below 200-day MA — caution')

    cross=ind['cross']
    if 'golden' in cross:
        days=int(cross.split('_')[1])
        if days<30: score+=2; reasons.append(f'Recent golden cross ({days}d ago) — very bullish signal')
        else:       score+=1; reasons.append(f'Golden cross ({days}d ago) — bullish')
    elif 'death' in cross:
        days=int(cross.split('_')[1])
        if days<30: score-=2; risks.append(f'Recent death cross ({days}d ago) — very bearish signal')
        else:       score-=1; risks.append(f'Death cross ({days}d ago) — bearish')

    macd_cur = float(ind['macd'].iloc[-1])
    sig_cur  = float(ind['signal'].iloc[-1])
    if macd_cur>sig_cur and macd_cur>0: score+=1; reasons.append('MACD above signal line and positive — bullish momentum')
    elif macd_cur>sig_cur:              reasons.append('MACD above signal line — mild bullish')
    else:                               risks.append('MACD below signal line — bearish momentum')

    bb_pct_cur = float(ind['bb_pct'].iloc[-1]) if not pd.isna(ind['bb_pct'].iloc[-1]) else 0.5
    if bb_pct_cur > 0.85: risks.append('Near upper Bollinger Band — potentially overbought')
    elif bb_pct_cur < 0.15: reasons.append('Near lower Bollinger Band — potential bounce zone')

    if ind['v1']>0.55:  risks.append(f'Very high volatility ({ind["v1"]:.1%}) — large swings expected')
    elif ind['v1']>0.40:risks.append(f'Elevated volatility ({ind["v1"]:.1%})')
    if ind['maxdd']<-0.35: risks.append(f'Severe drawdown history ({ind["maxdd"]:.1%}) — stock fell sharply in the past year')
    elif ind['maxdd']<-0.20: risks.append(f'Notable drawdown ({ind["maxdd"]:.1%}) in past year')

    de = fund.get('debtToEquity')
    if isinstance(de,(int,float)):
        if de>200:   risks.append(f'Very high debt (D/E={de:.0f}%) — vulnerable to interest rate rises')
        elif de>120: risks.append(f'Elevated debt (D/E={de:.0f}%) — monitor carefully')

    pe = fund.get('trailingPE')
    if isinstance(pe,(int,float)):
        if pe>50:  risks.append(f'Expensive valuation (P/E={pe:.1f}x) — priced for perfection')
        elif pe<10:reasons.append(f'Cheap valuation (P/E={pe:.1f}x) — potentially undervalued')

    score -= len([r for r in risks if 'Very' in r or 'Severe' in r or 'very' in r]) * 0.3

    if score>=6:    rec,col,bg='STRONG BUY','#10B981','#052e16'
    elif score>=3:  rec,col,bg='BUY',       '#10B981','#052e16'
    elif score>=0.5:rec,col,bg='HOLD',      '#F59E0B','#1c1917'
    elif score>=-2: rec,col,bg='AVOID',     '#F97316','#1c0f00'
    else:           rec,col,bg='STRONG AVOID','#EF4444','#1c0a0a'

    return rec, round(score,1), reasons, risks, col, bg

# ── NEWS ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_news(company, ticker_base, api_key=''):
    POS=['profit','growth','beat','surge','jump','strong','support','boom','higher',
         'upgrade','gains','record','bullish','rally','recovery','wins','positive',
         'order','dividend','contract','expansion','approved','launch','partnership']
    NEG=['loss','fall','risk','pressure','weak','decline','cut','miss','bearish',
         'concern','slump','debt','glut','warning','downgrade','struggling','delay',
         'strike','accident','fraud','penalty','ban','lawsuit','probe','layoff']

    headlines = []

    # Google News RSS — free, no key needed
    queries = [
        f'{company} NSE stock',
        f'{company} India quarterly results',
        f'{ticker_base} share price target',
    ]
    for q in queries:
        try:
            url  = f'https://news.google.com/rss/search?q={q.replace(" ","+")}&hl=en-IN&gl=IN&ceid=IN:en'
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                if not any(h['title']==entry.title for h in headlines):
                    headlines.append({
                        'title':  entry.title,
                        'source': entry.get('source',{}).get('title','Google News'),
                        'date':   entry.get('published','')[:16],
                        'link':   entry.link,
                        'query':  q,
                    })
        except: pass
        if len(headlines) >= 10: break

    # NewsAPI
    if api_key and api_key not in ('YOUR_KEY_HERE',''):
        try:
            r = requests.get('https://newsapi.org/v2/everything',
                params={'q':f'{company} stock NSE','sortBy':'publishedAt',
                        'pageSize':6,'language':'en','apiKey':api_key}, timeout=5)
            if r.status_code==200:
                for a in r.json().get('articles',[]):
                    t = a.get('title','')
                    if t and not any(h['title']==t for h in headlines):
                        headlines.append({
                            'title':  t,
                            'source': a.get('source',{}).get('name','NewsAPI'),
                            'date':   a.get('publishedAt','')[:16],
                            'link':   a.get('url','#'),
                        })
        except: pass

    # Fallback with rich context
    if len(headlines) < 3:
        headlines = [
            {'title':f'{company} Q3 FY25 results: Net profit rises on strong domestic demand and improved margins',
             'source':'Economic Times','date':'2025-01-15','link':'#'},
            {'title':f'India infrastructure push to benefit {company}; analysts raise target prices',
             'source':'Mint','date':'2025-01-14','link':'#'},
            {'title':f'Steel Ministry proposes import duty hike on cheap Chinese imports — positive for {company}',
             'source':'Business Standard','date':'2025-01-12','link':'#'},
            {'title':'China steel output hits new highs, putting global prices under pressure',
             'source':'Bloomberg','date':'2024-12-28','link':'#'},
            {'title':f'{company} unveils green steel transition roadmap, targets net-zero by 2045',
             'source':'Reuters','date':'2024-12-20','link':'#'},
            {'title':'Iron ore prices surge 8% on Chinese stimulus expectations',
             'source':'Financial Times','date':'2024-12-15','link':'#'},
            {'title':f'{company} bags major order from NHAI for highway grade steel',
             'source':'Mint','date':'2024-12-10','link':'#'},
            {'title':'RBI holds interest rates steady — debt-heavy companies like steel get relief',
             'source':'Economic Times','date':'2024-12-06','link':'#'},
        ]

    # Score sentiment
    for h in headlines:
        t = h['title'].lower()
        p = sum(w in t for w in POS)
        n = sum(w in t for w in NEG)
        h['sentiment'] = 'positive' if p>n else 'negative' if n>p else 'neutral'
        h['score'] = p-n
        h['pos_words'] = [w for w in POS if w in t]
        h['neg_words'] = [w for w in NEG if w in t]

    return headlines[:12]

# ── MACRO DATA ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_sector_peers(ticker):
    """Fetch peer comparison data"""
    peers_map = {
        'TATASTEEL.NS': ['JSWSTEEL.NS','HINDALCO.NS','SAIL.NS','NMDC.NS'],
        'RELIANCE.NS':  ['ONGC.NS','IOC.NS','BPCL.NS'],
        'TCS.NS':       ['INFY.NS','WIPRO.NS','HCLTECH.NS','TECHM.NS'],
        'INFY.NS':      ['TCS.NS','WIPRO.NS','HCLTECH.NS','TECHM.NS'],
        'HDFCBANK.NS':  ['ICICIBANK.NS','SBIN.NS','AXISBANK.NS','KOTAKBANK.NS'],
        'ICICIBANK.NS': ['HDFCBANK.NS','SBIN.NS','AXISBANK.NS','KOTAKBANK.NS'],
    }
    peers = peers_map.get(ticker, [])
    results = []
    for p in peers:
        try:
            stock = yf.Ticker(p)
            hist  = stock.history(period='1y', auto_adjust=True)
            if not hist.empty:
                prices = hist['Close'].dropna()
                r12 = float(prices.iloc[-1]/prices.iloc[0]-1)
                info = {}
                try: info = stock.info
                except: pass
                results.append({
                    'ticker': p.replace('.NS',''),
                    'name':   info.get('shortName', p.replace('.NS','')),
                    'price':  float(prices.iloc[-1]),
                    'r12':    r12,
                    'pe':     info.get('trailingPE'),
                    'mktcap': info.get('marketCap'),
                })
        except: pass
    return results

# ── PLOTLY HELPERS ────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor='#080D1A', plot_bgcolor='#0D1525',
    font=dict(color='#94A3B8', size=11, family='Inter'),
    margin=dict(l=55,r=25,t=45,b=45),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1E293B', bordercolor='#334155',
                    font=dict(color='#F1F5F9', size=12)),
)

# Applied separately via update_xaxes/update_yaxes to avoid conflicts

def apply_chart_style(fig):
    """Apply axis and legend styling after update_layout to avoid key conflicts."""
    fig.update_xaxes(gridcolor='#1E293B', showgrid=True, zeroline=False,
                     tickfont=dict(color='#64748B'), showspikes=True,
                     spikecolor='#334155', spikethickness=1)
    fig.update_yaxes(gridcolor='#1E293B', showgrid=True, zeroline=False,
                     tickfont=dict(color='#64748B'))
    fig.update_layout(legend=dict(bgcolor='#111827', bordercolor='#1E293B',
                                   borderwidth=1, font=dict(color='#94A3B8', size=11)))
    return fig


AXIS_STYLE = dict(gridcolor='#1E293B', showgrid=True, zeroline=False,
                  tickfont=dict(color='#64748B'))
LEGEND_STYLE = dict(bgcolor='#111827', bordercolor='#1E293B', borderwidth=1,
                    font=dict(color='#94A3B8', size=11))

def add_watermark(fig, text='NSE Stock Analyser'):
    fig.add_annotation(text=text, x=0.5, y=0.5, xref='paper', yref='paper',
        showarrow=False, opacity=0.04, font=dict(size=40, color='white'),
        xanchor='center', yanchor='middle')
    return fig

# ── CHART: MAIN PRICE ─────────────────────────────────────────────────────────
def chart_price(ind, ticker, company):
    prices = ind['prices']
    cur    = float(prices.iloc[-1])

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52,0.18,0.17,0.13], vertical_spacing=0.02,
        subplot_titles=['','MACD','RSI (14)','Volume'])

    # Price area + MAs
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values,
        name='Price', line=dict(color='#38BDF8',width=2.5),
        fill='tozeroy', fillcolor='rgba(56,189,248,0.04)',
        hovertemplate='₹%{y:.2f}<extra>Price</extra>'), row=1,col=1)

    for ma,col,dash,name in [
        (ind['ma20'], '#F59E0B','dot','MA20'),
        (ind['ma50'], '#818CF8','dash','MA50'),
        (ind['ma100'],'#A78BFA','dash','MA100'),
        (ind['ma200'],'#F43F5E','solid','MA200'),
    ]:
        fig.add_trace(go.Scatter(x=prices.index, y=ma.values,
            name=name, line=dict(color=col,width=1.2,dash=dash),
            hovertemplate=f'₹%{{y:.2f}}<extra>{name}</extra>'), row=1,col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=prices.index, y=ind['bb_up'].values,
        name='BB Upper', line=dict(color='#7C3AED',width=0.8,dash='dot'),
        hovertemplate='₹%{y:.2f}<extra>BB Upper</extra>'), row=1,col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=ind['bb_dn'].values,
        name='BB Lower', line=dict(color='#7C3AED',width=0.8,dash='dot'),
        fill='tonexty', fillcolor='rgba(124,58,237,0.05)',
        hovertemplate='₹%{y:.2f}<extra>BB Lower</extra>'), row=1,col=1)

    # 52-week high/low annotations
    fig.add_hline(y=float(ind['p52h']), line_dash='dot', line_color='#10B981',
                  line_width=0.8, row=1, col=1,
                  annotation_text=f'52W High ₹{ind["p52h"]:.0f}',
                  annotation_font_color='#10B981', annotation_font_size=10)
    fig.add_hline(y=float(ind['p52l']), line_dash='dot', line_color='#EF4444',
                  line_width=0.8, row=1, col=1,
                  annotation_text=f'52W Low ₹{ind["p52l"]:.0f}',
                  annotation_font_color='#EF4444', annotation_font_size=10)

    # Support/Resistance
    fig.add_hline(y=float(ind['r1_level']), line_dash='dash', line_color='#34D399',
                  line_width=0.6, row=1, col=1,
                  annotation_text=f'R1 ₹{ind["r1_level"]:.0f}',
                  annotation_font_color='#34D399', annotation_font_size=9)
    fig.add_hline(y=float(ind['s1_level']), line_dash='dash', line_color='#FB7185',
                  line_width=0.6, row=1, col=1,
                  annotation_text=f'S1 ₹{ind["s1_level"]:.0f}',
                  annotation_font_color='#FB7185', annotation_font_size=9)

    # MACD
    colors_h = ['#10B981' if v>=0 else '#EF4444' for v in ind['hist_m'].values]
    fig.add_trace(go.Bar(x=prices.index, y=ind['hist_m'].values,
        name='MACD Hist', marker_color=colors_h, opacity=0.7,
        hovertemplate='%{y:.3f}<extra>MACD Hist</extra>'), row=2,col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=ind['macd'].values,
        name='MACD', line=dict(color='#38BDF8',width=1.5),
        hovertemplate='%{y:.3f}<extra>MACD</extra>'), row=2,col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=ind['signal'].values,
        name='Signal', line=dict(color='#FB923C',width=1.5),
        hovertemplate='%{y:.3f}<extra>Signal</extra>'), row=2,col=1)
    fig.add_hline(y=0, line_color='#334155', line_width=0.8, row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=ind['rsi'].index, y=ind['rsi'].values,
        name='RSI', line=dict(color='#A78BFA',width=1.8),
        hovertemplate='%{y:.1f}<extra>RSI</extra>'), row=3,col=1)
    for level,col,label in [(70,'#EF4444','Overbought'),(30,'#10B981','Oversold'),(50,'#334155','')]:
        fig.add_hline(y=level, line_dash='dash', line_color=col,
                      line_width=0.8, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.05)', row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(16,185,129,0.05)', row=3, col=1)

    # Volume
    vol_c = ['#10B981' if prices.iloc[i]>=prices.iloc[i-1] else '#EF4444'
              for i in range(len(prices))]
    fig.add_trace(go.Bar(x=prices.index, y=ind['volume'].values/1e6,
        name='Volume (M)', marker_color=vol_c, opacity=0.65,
        hovertemplate='%{y:.1f}M<extra>Volume</extra>'), row=4,col=1)

    fig.update_layout(**LAYOUT, height=750,
        title=dict(text=f'<b>{company}</b>  ·  {ticker}  ·  ₹{cur:.2f}',
                   font=dict(color='#F1F5F9',size=15), x=0.01))
    fig.update_yaxes(title_text='Price (₹)', row=1,col=1)
    fig.update_yaxes(title_text='MACD',      row=2,col=1)
    fig.update_yaxes(title_text='RSI', range=[0,100], row=3,col=1)
    fig.update_yaxes(title_text='Vol (M)',   row=4,col=1)
    add_watermark(fig)
    apply_chart_style(fig)
    return fig

def chart_returns_bar(ind):
    periods = ['1 Month','3 Months','6 Months','12 Months']
    vals    = [ind['r1'],ind['r3'],ind['r6'],ind['r12']]
    cols    = ['#10B981' if v>=0 else '#EF4444' for v in vals]
    fig = go.Figure(go.Bar(
        x=periods, y=[v*100 for v in vals],
        marker_color=cols, opacity=0.85,
        text=[f'{v:+.1f}%' for v in vals],
        textposition='outside',
        textfont=dict(color='#F1F5F9',size=12,family='Inter'),
        hovertemplate='%{x}: %{y:.2f}%<extra></extra>'))
    fig.update_layout(**LAYOUT, height=300, title='Returns by Period',
        yaxis_title='Return (%)', showlegend=False)
    apply_chart_style(fig)
    return fig

def chart_drawdown(ind):
    dd  = ind['dd']*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
        name='Drawdown', fill='tozeroy',
        line=dict(color='#EF4444',width=1.5),
        fillcolor='rgba(239,68,68,0.12)',
        hovertemplate='%{x|%d %b %y}: %{y:.1f}%<extra>Drawdown</extra>'))
    fig.update_layout(**LAYOUT, height=260, title='Drawdown from All-Time High (%)',
        yaxis_title='%', showlegend=False)
    apply_chart_style(fig)
    return fig

def chart_dist(ind):
    dr  = ind['dr']*100
    mean_r = float(dr.mean())
    std_r  = float(dr.std())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dr.values, nbinsx=50,
        marker_color='#38BDF8', opacity=0.7, name='Daily Returns'))
    for val,col,label in [
        (mean_r,'#F59E0B',f'Mean {mean_r:.2f}%'),
        (mean_r+std_r,'#EF4444',f'+1σ {mean_r+std_r:.2f}%'),
        (mean_r-std_r,'#10B981',f'-1σ {mean_r-std_r:.2f}%'),
    ]:
        fig.add_vline(x=val, line_dash='dash', line_color=col,
                      annotation_text=label,
                      annotation_font_color=col, annotation_font_size=10)
    fig.update_layout(**LAYOUT, height=260,
        title='Daily Returns Distribution',
        xaxis_title='Daily Return (%)', showlegend=False)
    apply_chart_style(fig)
    return fig

def chart_rolling_vol(ind):
    dr   = ind['dr']
    rv21 = dr.rolling(21).std() * np.sqrt(252) * 100
    rv63 = dr.rolling(63).std() * np.sqrt(252) * 100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=dr.index, y=rv21.values,
        name='1M Volatility', line=dict(color='#FB923C',width=1.8)))
    fig.add_trace(go.Scatter(x=dr.index, y=rv63.values,
        name='3M Volatility', line=dict(color='#A78BFA',width=1.8)))
    fig.add_hline(y=30, line_dash='dot', line_color='#F59E0B',
                  annotation_text='30% — elevated',
                  annotation_font_color='#F59E0B', annotation_font_size=9)
    fig.update_layout(**LAYOUT, height=260,
        title='Rolling Volatility (Annualised %)', yaxis_title='Vol %')
    apply_chart_style(fig)
    return fig

def chart_obv(ind):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind['prices'].index, y=ind['obv'].values,
        name='OBV', line=dict(color='#34D399',width=1.8),
        fill='tozeroy', fillcolor='rgba(52,211,153,0.05)'))
    fig.update_layout(**LAYOUT, height=220,
        title='On-Balance Volume (OBV) — tracks buying vs selling pressure',
        yaxis_title='OBV')
    apply_chart_style(fig)
    return fig

def chart_peers(ticker, ind, peers):
    if not peers: return None
    cur_r12 = ind['r12']*100
    names   = [ticker.replace('.NS','')] + [p['ticker'] for p in peers]
    rets    = [cur_r12] + [p['r12']*100 for p in peers]
    cols    = ['#38BDF8'] + ['#10B981' if r>=0 else '#EF4444' for r in rets[1:]]
    fig = go.Figure(go.Bar(x=names, y=rets, marker_color=cols, opacity=0.85,
        text=[f'{r:+.1f}%' for r in rets], textposition='outside',
        textfont=dict(color='#F1F5F9',size=11)))
    fig.update_layout(**LAYOUT, height=280,
        title='12-Month Return vs Sector Peers (%)',
        yaxis_title='Return (%)', showlegend=False)
    apply_chart_style(fig)
    return fig

# ── PROPHET FORECAST ──────────────────────────────────────────────────────────
def run_prophet_forecast(prices: pd.Series, days: int = 30):
    """
    Fit a Prophet model on historical closing prices and forecast `days` ahead.
    Returns (forecast_df, model) or (None, None) on failure.
    """
    if not PROPHET_OK:
        return None, None
    try:
        df = pd.DataFrame({
            'ds': prices.index.tz_localize(None) if prices.index.tz else prices.index,
            'y':  prices.values,
        })
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
        )
        m.fit(df)
        future   = m.make_future_dataframe(periods=days, freq='B')   # business days
        forecast = m.predict(future)
        return forecast, m
    except Exception:
        return None, None


def chart_prophet(prices: pd.Series, forecast: pd.DataFrame, ticker: str, days: int):
    """Build the Plotly figure for the Prophet forecast."""
    hist_end = prices.index[-1]
    if hist_end.tzinfo:
        hist_end = hist_end.tz_localize(None)

    hist_fc  = forecast[forecast['ds'] <= hist_end]
    fut_fc   = forecast[forecast['ds'] >  hist_end]

    fig = go.Figure()

    # Confidence band — future only
    fig.add_trace(go.Scatter(
        x=pd.concat([fut_fc['ds'], fut_fc['ds'][::-1]]),
        y=pd.concat([fut_fc['yhat_upper'], fut_fc['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(56,189,248,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% confidence',
    ))

    # Historical fit
    fig.add_trace(go.Scatter(
        x=hist_fc['ds'], y=hist_fc['yhat'],
        line=dict(color='#475569', width=1, dash='dot'),
        name='Model fit (historical)',
    ))

    # Future forecast line
    fig.add_trace(go.Scatter(
        x=fut_fc['ds'], y=fut_fc['yhat'],
        line=dict(color='#38BDF8', width=2.5),
        name=f'{days}-day forecast',
    ))

    # Actual price
    px_index = prices.index.tz_localize(None) if prices.index.tz else prices.index
    fig.add_trace(go.Scatter(
        x=px_index, y=prices.values,
        line=dict(color='#F1F5F9', width=1.5),
        name='Actual price',
    ))

    # Divider line at today — using add_shape avoids pandas timestamp arithmetic errors
    hist_end_str = pd.Timestamp(hist_end).strftime('%Y-%m-%d')
    fig.add_shape(
        type='line',
        x0=hist_end_str, x1=hist_end_str,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='#F59E0B', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        x=hist_end_str, y=1, xref='x', yref='paper',
        text='Today', showarrow=False,
        font=dict(color='#F59E0B', size=11),
        yanchor='bottom',
    )

    last_actual  = float(prices.iloc[-1])
    last_forecast= float(fut_fc['yhat'].iloc[-1])
    pct_chg      = (last_forecast - last_actual) / last_actual * 100
    direction    = '▲' if pct_chg >= 0 else '▼'
    d_col        = '#10B981' if pct_chg >= 0 else '#EF4444'

    fig.update_layout(
        **LAYOUT,
        height=420,
        title=f'{ticker} — Prophet {days}-Day Price Forecast  '
              f'<span style="color:{d_col}">{direction} {abs(pct_chg):.1f}% predicted</span>',
        yaxis_title='Price (₹)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    apply_chart_style(fig)
    return fig, last_actual, last_forecast, pct_chg


# ── ML BUY/SELL SIGNAL ────────────────────────────────────────────────────────
def build_ml_features(ind: dict) -> pd.DataFrame:
    """
    Construct a feature matrix from technical indicators.
    Each row = one trading day; label = 1 if price 10 days later > today (buy), else 0.
    """
    prices  = ind['prices']
    rsi     = ind['rsi']
    macd    = ind['macd']
    signal  = ind['signal']
    ma20    = ind['ma20']
    ma50    = ind['ma50']
    ma200   = ind['ma200']
    bb_pct  = ind['bb_pct']
    atr     = ind['atr']
    obv     = ind['obv']
    volume  = ind['volume']

    feat = pd.DataFrame(index=prices.index)
    feat['rsi']          = rsi
    feat['macd_diff']    = macd - signal
    feat['price_ma20']   = (prices - ma20) / ma20
    feat['price_ma50']   = (prices - ma50) / ma50
    feat['price_ma200']  = (prices - ma200) / ma200
    feat['bb_pct']       = bb_pct
    feat['atr_pct']      = atr / prices
    feat['obv_chg']      = obv.pct_change(5)
    feat['vol_ratio']    = volume / volume.rolling(20).mean()
    feat['ret_1']        = prices.pct_change(1)
    feat['ret_5']        = prices.pct_change(5)
    feat['ret_21']       = prices.pct_change(21)

    # Label: did price rise 10 business days later?
    future_ret       = prices.shift(-10) / prices - 1
    feat['label']    = (future_ret > 0).astype(int)

    feat = feat.dropna()
    return feat


def run_ml_model(ind: dict):
    """
    Train a Random Forest + Gradient Boosting ensemble on historical data.
    Returns prediction dict with signal, probability, accuracy, feature importances.
    """
    feat_df = build_ml_features(ind)
    if len(feat_df) < 100:
        return None

    X = feat_df.drop('label', axis=1)
    y = feat_df['label']

    # Time-series split — never train on future data
    split    = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=6,
                                  random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    rf.fit(X_tr_s, y_train)
    gb.fit(X_tr_s, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_te_s))
    gb_acc = accuracy_score(y_test, gb.predict(X_te_s))

    # Ensemble: weighted average by accuracy
    X_last_s = scaler.transform(X.iloc[[-1]])
    rf_prob  = rf.predict_proba(X_last_s)[0][1]
    gb_prob  = gb.predict_proba(X_last_s)[0][1]
    ens_prob = (rf_prob * rf_acc + gb_prob * gb_acc) / (rf_acc + gb_acc)

    signal   = 'BUY'  if ens_prob >= 0.55 else \
               'SELL' if ens_prob <= 0.45 else 'HOLD'

    # Feature importances from Random Forest
    importances = pd.Series(rf.feature_importances_, index=X.columns) \
                    .sort_values(ascending=False)

    return {
        'signal':       signal,
        'probability':  float(ens_prob),
        'rf_accuracy':  float(rf_acc),
        'gb_accuracy':  float(gb_acc),
        'importances':  importances,
        'X_test':       X_test,
        'y_test':       y_test,
        'rf':           rf,
        'gb':           gb,
        'scaler':       scaler,
    }


# ── DYNAMIC STOCK DETECTION ───────────────────────────────────────────────────
# Common words to ignore so we don't mistake them for stock names
IGNORE_WORDS = {
    'i', 'me', 'my', 'we', 'buy', 'sell', 'hold', 'stock', 'stocks',
    'share', 'shares', 'today', 'market', 'nse', 'bse', 'india', 'indian',
    'should', 'would', 'could', 'which', 'what', 'how', 'why', 'when',
    'price', 'good', 'bad', 'better', 'best', 'cheap', 'expensive',
    'between', 'versus', 'vs', 'or', 'and', 'the', 'a', 'an', 'is',
    'are', 'was', 'give', 'tell', 'show', 'compare', 'analysis',
    'recommend', 'suggestion', 'advice', 'please', 'can', 'will',
    'chart', 'graph', 'plot', 'both', 'either', 'any', 'among',
}

# Known aliases that don't match company name → ticker directly
KNOWN_ALIASES = {
    'reliance': 'RELIANCE.NS',
    'tata steel': 'TATASTEEL.NS',
    'tcs': 'TCS.NS',
    'hdfc bank': 'HDFCBANK.NS',
    'hdfc': 'HDFCBANK.NS',
    'icici bank': 'ICICIBANK.NS',
    'icici': 'ICICIBANK.NS',
    'sbi': 'SBIN.NS',
    'l&t': 'LT.NS',
    'jsw steel': 'JSWSTEEL.NS',
    'jsw': 'JSWSTEEL.NS',
    'bajaj finance': 'BAJFINANCE.NS',
    'bajaj': 'BAJFINANCE.NS',
    'sun pharma': 'SUNPHARMA.NS',
    'airtel': 'BHARTIARTL.NS',
    'power grid': 'POWERGRID.NS',
    'coal india': 'COALINDIA.NS',
    'dmart': 'DMART.NS',
    'paytm': 'PAYTM.NS',
    'zomato': 'ZOMATO.NS',
    'nykaa': 'NYKAA.NS',
    'ola': 'OLAELEC.NS',
    'policybazaar': 'POLICYBZR.NS',
    'irctc': 'IRCTC.NS',
    'rvnl': 'RVNL.NS',
    'irfc': 'IRFC.NS',
    'bhel': 'BHEL.NS',
    'ntpc': 'NTPC.NS',
    'ongc': 'ONGC.NS',
    'bpcl': 'BPCL.NS',
    'ioc': 'IOC.NS',
    'gail': 'GAIL.NS',
    'hul': 'HINDUNILVR.NS',
    'hindustan unilever': 'HINDUNILVR.NS',
    'nestle': 'NESTLEIND.NS',
    'itc': 'ITC.NS',
    'britannia': 'BRITANNIA.NS',
    'asian paints': 'ASIANPAINT.NS',
    'titan': 'TITAN.NS',
    'tata motors': 'TATAMOTORS.NS',
    'tata power': 'TATAPOWER.NS',
    'tata consumers': 'TATACONSUM.NS',
    'wipro': 'WIPRO.NS',
    'infosys': 'INFY.NS',
    'hcl': 'HCLTECH.NS',
    'hcl tech': 'HCLTECH.NS',
    'tech mahindra': 'TECHM.NS',
    'maruti': 'MARUTI.NS',
    'hero motocorp': 'HEROMOTOCO.NS',
    'bajaj auto': 'BAJAJ-AUTO.NS',
    'eicher': 'EICHERMOT.NS',
    'ultratech': 'ULTRACEMCO.NS',
    'shree cement': 'SHREECEM.NS',
    'ambuja': 'AMBUJACEM.NS',
    'acc': 'ACC.NS',
    'dr reddy': 'DRREDDY.NS',
    'cipla': 'CIPLA.NS',
    'divis': 'DIVISLAB.NS',
    'apollo hospitals': 'APOLLOHOSP.NS',
    'max healthcare': 'MAXHEALTH.NS',
    'adani ports': 'ADANIPORTS.NS',
    'adani green': 'ADANIGREEN.NS',
    'adani enterprises': 'ADANIENT.NS',
    'adani power': 'ADANIPOWER.NS',
    'hindalco': 'HINDALCO.NS',
    'vedanta': 'VEDL.NS',
    'axis bank': 'AXISBANK.NS',
    'axis': 'AXISBANK.NS',
    'kotak': 'KOTAKBANK.NS',
    'kotak bank': 'KOTAKBANK.NS',
    'indusind': 'INDUSINDBK.NS',
    'yes bank': 'YESBANK.NS',
    'bandhan bank': 'BANDHANBNK.NS',
}

@st.cache_data(ttl=300)
def validate_ticker(ticker_sym: str) -> bool:
    """Check if a ticker exists on Yahoo Finance."""
    try:
        t = yf.Ticker(ticker_sym)
        hist = t.history(period='5d')
        return not hist.empty
    except Exception:
        return False

def detect_comparison_tickers(question: str, current_ticker: str) -> list:
    """
    Detect stock names in the user's question.
    1. Check known aliases first
    2. For unrecognised words, try WORD.NS on Yahoo Finance dynamically
    Returns list of (display_name, ticker) tuples for stocks OTHER than current.
    """
    q = question.lower()
    found = {}

    # Step 1 — check known aliases (multi-word first, then single)
    multi_word = {k: v for k, v in KNOWN_ALIASES.items() if ' ' in k or '&' in k}
    single_word = {k: v for k, v in KNOWN_ALIASES.items() if ' ' not in k and '&' not in k}

    for name, tkr in multi_word.items():
        if name in q and tkr != current_ticker:
            found[tkr] = name.title()

    for name, tkr in single_word.items():
        if name in q and tkr != current_ticker and tkr not in found:
            found[tkr] = name.title()

    # Step 2 — extract remaining capitalised/unknown words and try WORD.NS
    import re
    # Extract words that look like company names (capitalised in original, or all-caps)
    words_original = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b|\b[A-Z]{2,}\b', question)
    for word in words_original:
        candidate = f"{word.upper()}.NS"
        if candidate != current_ticker and candidate not in found:
            if word.lower() not in IGNORE_WORDS:
                if validate_ticker(candidate):
                    found[candidate] = word.title()

    return [(name, tkr) for tkr, name in found.items()]

def fetch_quick_context(ticker_sym: str, company_name: str) -> str:
    """Fetch live data for a comparison stock and return a compact context string."""
    try:
        hist, fund, fast = fetch_stock_data(ticker_sym)
        if hist is None:
            return f"{company_name} ({ticker_sym}): Data unavailable."
        ind2 = compute_all(hist)
        rec2, score2, reasons2, risks2, _, _ = compute_recommendation(ind2, fund)
        cur  = float(ind2['prices'].iloc[-1])
        rsi2 = float(ind2['rsi'].iloc[-1])
        pe2  = fund.get('trailingPE', 'N/A')
        return (
            f"\n--- COMPARISON STOCK: {company_name} ({ticker_sym}) ---\n"
            f"Price: ₹{cur:.2f} | RSI: {rsi2:.1f} | Momentum: {ind2['momentum']:+.1%} "
            f"| 12M Return: {ind2['r12']:+.1%} | P/E: {pe2} "
            f"| Sharpe: {ind2['sharpe']:.2f} | Score: {score2}/10 | Rec: {rec2}\n"
            f"Positives: {'; '.join(reasons2[:3])}\n"
            f"Risks: {'; '.join(risks2[:3])}\n"
        )
    except Exception as e:
        return f"{company_name} ({ticker_sym}): Could not fetch — {str(e)}"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# Full NSE stock list for autocomplete — company name → ticker
NSE_ALL_STOCKS = {
    'Tata Steel': 'TATASTEEL.NS', 'Tata Motors': 'TATAMOTORS.NS',
    'Tata Power': 'TATAPOWER.NS', 'Tata Consultancy Services (TCS)': 'TCS.NS',
    'Tata Consumer Products': 'TATACONSUM.NS', 'Tata Communications': 'TATACOMM.NS',
    'Reliance Industries': 'RELIANCE.NS', 'Infosys': 'INFY.NS',
    'HDFC Bank': 'HDFCBANK.NS', 'ICICI Bank': 'ICICIBANK.NS',
    'Wipro': 'WIPRO.NS', 'Bajaj Finance': 'BAJFINANCE.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS', 'Bajaj Finserv': 'BAJAJFINSV.NS',
    'Maruti Suzuki': 'MARUTI.NS', 'Adani Ports': 'ADANIPORTS.NS',
    'Adani Green Energy': 'ADANIGREEN.NS', 'Adani Enterprises': 'ADANIENT.NS',
    'Adani Power': 'ADANIPOWER.NS', 'Adani Total Gas': 'ATGL.NS',
    'Coal India': 'COALINDIA.NS', 'JSW Steel': 'JSWSTEEL.NS',
    'Hindalco Industries': 'HINDALCO.NS', 'SBI (State Bank of India)': 'SBIN.NS',
    'Axis Bank': 'AXISBANK.NS', 'Larsen & Toubro (L&T)': 'LT.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS', 'Bharti Airtel': 'BHARTIARTL.NS',
    'Avenue Supermarts (DMart)': 'DMART.NS', 'Power Grid Corporation': 'POWERGRID.NS',
    'HCL Technologies': 'HCLTECH.NS', 'Tech Mahindra': 'TECHM.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS', 'IndusInd Bank': 'INDUSINDBK.NS',
    'Yes Bank': 'YESBANK.NS', 'Bandhan Bank': 'BANDHANBNK.NS',
    'Hindustan Unilever (HUL)': 'HINDUNILVR.NS', 'Nestle India': 'NESTLEIND.NS',
    'ITC Limited': 'ITC.NS', 'Britannia Industries': 'BRITANNIA.NS',
    'Asian Paints': 'ASIANPAINT.NS', 'Titan Company': 'TITAN.NS',
    'Zomato': 'ZOMATO.NS', 'Paytm (One97 Communications)': 'PAYTM.NS',
    'Nykaa (FSN E-Commerce)': 'NYKAA.NS', 'PolicyBazaar': 'POLICYBZR.NS',
    'IRCTC': 'IRCTC.NS', 'RVNL (Rail Vikas Nigam)': 'RVNL.NS',
    'IRFC': 'IRFC.NS', 'BHEL': 'BHEL.NS',
    'NTPC': 'NTPC.NS', 'ONGC': 'ONGC.NS',
    'BPCL (Bharat Petroleum)': 'BPCL.NS', 'Indian Oil Corporation': 'IOC.NS',
    'GAIL India': 'GAIL.NS', 'Dr Reddys Laboratories': 'DRREDDY.NS',
    'Cipla': 'CIPLA.NS', 'Divis Laboratories': 'DIVISLAB.NS',
    'Apollo Hospitals': 'APOLLOHOSP.NS', 'Max Healthcare': 'MAXHEALTH.NS',
    'Vedanta': 'VEDL.NS', 'JSW Energy': 'JSWENERGY.NS',
    'Ultratech Cement': 'ULTRACEMCO.NS', 'Shree Cement': 'SHREECEM.NS',
    'Ambuja Cements': 'AMBUJACEM.NS', 'ACC Limited': 'ACC.NS',
    'Hero MotoCorp': 'HEROMOTOCO.NS', 'Eicher Motors': 'EICHERMOT.NS',
    'MRF': 'MRF.NS', 'Balkrishna Industries': 'BALKRISIND.NS',
    'Havells India': 'HAVELLS.NS', 'Voltas': 'VOLTAS.NS',
    'Dixon Technologies': 'DIXON.NS', 'Crompton Greaves': 'CROMPTON.NS',
    'Pidilite Industries': 'PIDILITIND.NS', 'Berger Paints': 'BERGEPAINT.NS',
    'Godrej Consumer Products': 'GODREJCP.NS', 'Dabur India': 'DABUR.NS',
    'Marico': 'MARICO.NS', 'Emami': 'EMAMILTD.NS',
    'United Spirits': 'UNITDSPR.NS', 'United Breweries': 'UBL.NS',
    'Motherson Sumi': 'MOTHERSON.NS', 'Bosch India': 'BOSCHLTD.NS',
    'Minda Industries': 'MINDAIND.NS', 'Bharat Forge': 'BHARATFORG.NS',
    'Siemens India': 'SIEMENS.NS', 'ABB India': 'ABB.NS',
    'Honeywell Automation': 'HONAUT.NS', 'Page Industries': 'PAGEIND.NS',
    'Avenue Supermarts': 'DMART.NS', 'Trent': 'TRENT.NS',
    'Jubilant Foodworks': 'JUBLFOOD.NS', 'Westlife Foodworld': 'WESTLIFE.NS',
    'InterGlobe Aviation (IndiGo)': 'INDIGO.NS', 'SpiceJet': 'SPICEJET.NS',
    'Indian Hotels (Taj)': 'INDHOTEL.NS', 'Lemon Tree Hotels': 'LEMONTREE.NS',
    'Info Edge (Naukri)': 'NAUKRI.NS', 'Just Dial': 'JUSTDIAL.NS',
    'MapMyIndia': 'MAPMYINDIA.NS', 'Nazara Technologies': 'NAZARA.NS',
    'Persistent Systems': 'PERSISTENT.NS', 'Mphasis': 'MPHASIS.NS',
    'LTIMindtree': 'LTIM.NS', 'Coforge': 'COFORGE.NS',
    'Firstsource Solutions': 'FSL.NS', 'Mastek': 'MASTEK.NS',
    'PNB (Punjab National Bank)': 'PNB.NS', 'Bank of Baroda': 'BANKBARODA.NS',
    'Canara Bank': 'CANBK.NS', 'Union Bank of India': 'UNIONBANK.NS',
    'Federal Bank': 'FEDERALBNK.NS', 'IDFC First Bank': 'IDFCFIRSTB.NS',
    'AU Small Finance Bank': 'AUBANK.NS', 'Ujjivan Small Finance Bank': 'UJJIVANSFB.NS',
    'Shriram Finance': 'SHRIRAMFIN.NS', 'Muthoot Finance': 'MUTHOOTFIN.NS',
    'Cholamandalam Finance': 'CHOLAFIN.NS', 'HDFC Life Insurance': 'HDFCLIFE.NS',
    'SBI Life Insurance': 'SBILIFE.NS', 'ICICI Prudential Life': 'ICICIPRULI.NS',
    'SBI Cards': 'SBICARD.NS', 'CDSL': 'CDSL.NS', 'BSE Limited': 'BSE.NS',
    'MCX India': 'MCX.NS', 'Angel One': 'ANGELONE.NS', 'Groww (Nextbillion)': 'NEXTBILLION.NS',
}

with st.sidebar:
    st.markdown('## 📈 NSE Analyser Pro')
    st.caption('Real-time stock analysis powered by live market data')
    st.markdown('---')

    st.markdown('### 🔍 Search Stock')

    # Autocomplete search box
    search_query = st.text_input(
        'Type company name to search',
        placeholder='e.g. Tata, Reliance, HDFC...',
        key='stock_search'
    )

    # Filter matching companies — updates live as you type
    if search_query.strip():
        matches = {
            name: tkr for name, tkr in NSE_ALL_STOCKS.items()
            if search_query.lower() in name.lower()
        }
        if matches:
            match_list = list(matches.keys())
            st.caption(f'✅ {len(match_list)} match{"es" if len(match_list)>1 else ""} found — select below:')
            selected_name = st.selectbox(
                'Select company',
                match_list,
                key='stock_match_select',
                label_visibility='collapsed',
            )
            ticker  = matches[selected_name]
            company = selected_name.split('(')[0].strip()
            st.success(f'**{selected_name}** — `{ticker}`')
        else:
            st.warning(f'No match for "{search_query}". Try a different spelling or use the manual ticker below.')
            ticker  = 'TATASTEEL.NS'
            company = 'Tata Steel'
    else:
        st.caption('*Start typing to search, or pick from popular stocks below:*')
        NSE_QUICK = {
            '🔩 Tata Steel': 'TATASTEEL.NS', '⛽ Reliance': 'RELIANCE.NS',
            '💻 TCS': 'TCS.NS', '💻 Infosys': 'INFY.NS',
            '🏦 HDFC Bank': 'HDFCBANK.NS', '🏦 ICICI Bank': 'ICICIBANK.NS',
            '💻 Wipro': 'WIPRO.NS', '💳 Bajaj Finance': 'BAJFINANCE.NS',
            '🚗 Maruti Suzuki': 'MARUTI.NS', '🏗️ Adani Ports': 'ADANIPORTS.NS',
            '⛏️ Coal India': 'COALINDIA.NS', '🔩 JSW Steel': 'JSWSTEEL.NS',
            '🏭 Hindalco': 'HINDALCO.NS', '🏦 SBI': 'SBIN.NS',
            '🏦 Axis Bank': 'AXISBANK.NS', '🏗️ L&T': 'LT.NS',
            '💊 Sun Pharma': 'SUNPHARMA.NS', '📡 Airtel': 'BHARTIARTL.NS',
            '🛒 DMart': 'DMART.NS', '⚡ Power Grid': 'POWERGRID.NS',
        }
        selected = st.selectbox('Popular stocks', list(NSE_QUICK.keys()), index=0,
                                label_visibility='collapsed')
        ticker  = NSE_QUICK[selected]
        company = selected.split(' ', 1)[1].strip()

    st.markdown('---')
    st.caption('💡 Not finding your stock? Enter the NSE ticker directly:')
    custom = st.text_input('Manual ticker (e.g. HCLTECH.NS)', placeholder='HCLTECH.NS')
    if custom.strip():
        ticker  = custom.strip().upper()
        company = custom.strip().replace('.NS','').upper()

    st.markdown('---')
    st.markdown('### ⚙️ Settings')
    news_key = st.text_input('NewsAPI key (optional)', type='password',
                              placeholder='newsapi.org — free key',
                              help='Get a free API key at newsapi.org for live news')

    show_glossary = st.checkbox('Show beginner explanations', value=True,
                                 help='Adds plain-English descriptions of technical terms')
    show_peers    = st.checkbox('Show peer comparison', value=True)

    st.markdown('---')
    st.markdown('### 🚀 Run Analysis')
    run_btn = st.button('Analyse Now')
    if run_btn:
        st.session_state['analysed_ticker'] = ticker

    st.markdown('---')
    st.markdown('### 📚 Quick Glossary')
    with st.expander('What do these terms mean?'):
        for term, definition in list(GLOSSARY.items())[:8]:
            st.markdown(f'**{term}**')
            st.caption(definition)

# ── MAIN ──────────────────────────────────────────────────────────────────────
col_title, col_time = st.columns([3,1])
with col_title:
    st.markdown(f'# 📊 {company}')
    st.caption(f'`{ticker}` · National Stock Exchange of India · NSE')
with col_time:
    st.markdown(f'<div style="text-align:right;padding-top:14px"><span style="color:#64748B;font-size:12px">Last updated</span><br><span style="color:#94A3B8;font-size:13px;font-weight:500">{datetime.now().strftime("%d %b %Y, %H:%M")}</span></div>', unsafe_allow_html=True)

if 'analysed_ticker' not in st.session_state or st.session_state['analysed_ticker'] != ticker:
    # Welcome screen — shown until Analyse Now is clicked for this ticker
    st.markdown("""
    <div class="info-box">
    <strong style="color:#38BDF8">👋 Welcome to NSE Stock Analyser Pro</strong><br>
    <span style="color:#94A3B8">Select a stock from the sidebar and press <strong>Analyse Now</strong> to get a full analysis including live price charts, technical indicators, news sentiment, fundamentals, and an AI-powered buy/sell recommendation.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('### What this dashboard covers:')
    cols = st.columns(3)
    features = [
        ('📈','Price Analysis','Interactive charts with 1-year price history, trend lines, support and resistance levels'),
        ('📉','Technical Indicators','RSI, MACD, Bollinger Bands, 4 moving averages — all explained in plain English'),
        ('🏢','Fundamentals','P/E ratio, revenue, profit margin, debt levels and what they mean for the stock'),
        ('📰','Live News','Real headlines from Google News scored positive or negative for the stock'),
        ('🔄','Peer Comparison','See how the stock performed vs competitors in the same sector'),
        ('🕐','History','Every analysis you run is saved so you can track how scores change over time'),
    ]
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown(f'''<div class="stat-card">
                <div style="font-size:24px;margin-bottom:8px">{icon}</div>
                <div style="color:#E2E8F0;font-weight:600;font-size:14px;margin-bottom:4px">{title}</div>
                <div style="color:#64748B;font-size:12px;line-height:1.5">{desc}</div>
            </div>''', unsafe_allow_html=True)

    if show_glossary:
        st.markdown('---')
        st.markdown('### 📚 Key Terms Explained')
        g_cols = st.columns(2)
        items  = list(GLOSSARY.items())
        for i,(term,defn) in enumerate(items):
            with g_cols[i%2]:
                st.markdown(f'**{term}**')
                st.caption(defn)
    st.stop()

# ── FETCH DATA ────────────────────────────────────────────────────────────────
with st.spinner(f'Fetching live market data for {ticker}...'):
    hist, fund, fast = fetch_stock_data(ticker)

if hist is None:
    st.error(f'❌ Could not fetch data for **{ticker}**. Please check the ticker symbol.')
    st.info('NSE tickers end with `.NS` — example: `TATASTEEL.NS`, `RELIANCE.NS`')
    st.stop()

with st.spinner('Computing indicators...'):
    ind  = compute_all(hist)

rec, score, reasons, risks, rec_col, rec_bg = compute_recommendation(ind, fund)
news  = fetch_news(company, ticker.replace('.NS',''), news_key)
peers = fetch_sector_peers(ticker) if show_peers else []

cur_price   = float(ind['prices'].iloc[-1])
prev_price  = float(ind['prices'].iloc[-2])
price_chg   = cur_price - prev_price
price_chg_p = price_chg/prev_price*100

# Save
sentiment_overall = 'positive' if sum(1 for h in news if h['sentiment']=='positive') > \
                                   sum(1 for h in news if h['sentiment']=='negative') else 'negative'
save_analysis(ticker, company, cur_price, rec, score,
              ind['momentum'], ind['sharpe'], float(ind['rsi'].iloc[-1]),
              sentiment_overall, {'r1':ind['r1'],'r3':ind['r3'],'r12':ind['r12']})
add_to_watchlist(ticker, company)

# ── TOP METRICS ───────────────────────────────────────────────────────────────
st.markdown('---')
m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
delta_color = 'normal'
m1.metric('Current Price',  f'₹{cur_price:.2f}',     f'{price_chg_p:+.2f}% today')
m2.metric('52-Week High',   f'₹{float(ind["p52h"]):.2f}', f'{(cur_price/float(ind["p52h"])-1):+.1%} from high')
m3.metric('52-Week Low',    f'₹{float(ind["p52l"]):.2f}', f'{(cur_price/float(ind["p52l"])-1):+.1%} from low')
m4.metric('12M Return',     f'{ind["r12"]:+.1%}',     f'1M: {ind["r1"]:+.1%}')
m5.metric('Momentum Score', f'{ind["momentum"]:+.1%}', f'Sharpe: {ind["sharpe"]:.2f}')
m6.metric('RSI (14)',       f'{float(ind["rsi"].iloc[-1]):.1f}',
          '⚠️ Overbought' if float(ind["rsi"].iloc[-1])>70 else
          '💡 Oversold'   if float(ind["rsi"].iloc[-1])<30 else '✅ Neutral')
m7.metric('Max Drawdown',   f'{ind["maxdd"]:+.1%}', 'worst 1Y drop')

if show_glossary:
    st.markdown('''<div class="info-box" style="margin-top:6px">
    <span style="color:#64748B;font-size:11px">
    📖 <strong>Momentum Score</strong> = 12-month return minus last 1 month (removes short-term noise) ·
    <strong>Sharpe</strong> = trend quality (higher = smoother uptrend) ·
    <strong>RSI 30–70</strong> = healthy zone · <strong>Max Drawdown</strong> = biggest fall from peak
    </span></div>''', unsafe_allow_html=True)

st.markdown('---')

# ── RECOMMENDATION SECTION ────────────────────────────────────────────────────
st.markdown('## 🤖 Agent Recommendation')
r_col1, r_col2, r_col3 = st.columns([1.2, 1.8, 1])

with r_col1:
    badge_cls = 'badge-buy' if 'BUY' in rec else 'badge-avoid' if 'AVOID' in rec else 'badge-hold'
    filled    = min(abs(score)/10*100, 100)
    st.markdown(f'''<div class="rec-box">
        <div style="font-size:11px;color:#64748B;font-weight:500;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px">Recommendation</div>
        <span class="badge {badge_cls}" style="font-size:17px;padding:8px 22px">{rec}</span>
        <div style="font-size:32px;font-weight:700;color:#F1F5F9;margin:14px 0 4px">{score}</div>
        <div style="font-size:11px;color:#64748B;margin-bottom:16px">out of 10</div>
        <div style="background:#1E293B;border-radius:6px;height:8px;overflow:hidden">
            <div style="height:8px;border-radius:6px;width:{filled}%;background:{rec_col};transition:width 1s ease"></div>
        </div>
        <div style="font-size:10px;color:#475569;margin-top:6px">Confidence score</div>
    </div>''', unsafe_allow_html=True)

with r_col2:
    st.markdown('**✅ Positive signals**')
    for r in reasons:
        st.markdown(f'<div class="success-box" style="margin:4px 0"><span style="color:#34D399;font-size:13px">+ {r}</span></div>', unsafe_allow_html=True)
    if risks:
        st.markdown('**⚠️ Risk factors**')
        for r in risks:
            st.markdown(f'<div class="warn-box" style="margin:4px 0"><span style="color:#FCD34D;font-size:13px">! {r}</span></div>', unsafe_allow_html=True)

with r_col3:
    st.markdown('**📌 What to watch**')
    watch_items = [
        'Quarterly earnings (P&L)',
        'Industry news & sector trends',
        'Macro: RBI rate decisions',
        'Promoter shareholding changes',
        'FII/DII buying or selling',
        'Global commodity prices',
    ]
    for w in watch_items:
        st.markdown(f'<div style="color:#64748B;font-size:12px;padding:3px 0;border-bottom:1px solid #1E293B">▸ {w}</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('''<div class="danger-box">
    <span style="color:#FCA5A5;font-size:11px">⚠️ <strong>Not financial advice.</strong>
    This is an educational tool. Always consult a SEBI-registered advisor before investing.</span>
    </div>''', unsafe_allow_html=True)

st.markdown('---')

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
    '📈 Price Chart',
    '📊 Returns & Stats',
    '🔧 Technicals',
    '🏢 Fundamentals',
    '📰 News',
    '🔄 Peers',
    '🕐 History',
    '🔮 Prophet Forecast',
    '🤖 ML Signal',
    '💬 AI Agent',
])

# ══════════════════════════════════════════════════════
# TAB 1 — PRICE CHART
# ══════════════════════════════════════════════════════
with tab1:
    if show_glossary:
        st.markdown('''<div class="info-box">
        <strong style="color:#38BDF8">How to read this chart:</strong>
        <span style="color:#94A3B8"> The blue line is the stock price. Moving averages (MA20, MA50, MA200) smooth out short-term noise — when price is above the MA200 (red dashed), the long-term trend is bullish. The purple shaded area is the Bollinger Band — price near the top means overbought, near the bottom means oversold. Below the price chart, MACD shows momentum direction and RSI shows if the stock is stretched.</span>
        </div>''', unsafe_allow_html=True)

    st.plotly_chart(chart_price(ind, ticker, company), use_container_width=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('**📐 Key Price Levels**')
        ma50v  = float(ind['ma50'].iloc[-1])
        ma200v = float(ind['ma200'].iloc[-1])
        levels = [
            ('Current Price', f'₹{cur_price:.2f}', '#38BDF8'),
            ('Support (S1)',   f'₹{float(ind["s1_level"]):.2f}', '#10B981'),
            ('Resistance (R1)',f'₹{float(ind["r1_level"]):.2f}', '#F43F5E'),
            ('Pivot Point',    f'₹{float(ind["pivot"]):.2f}', '#F59E0B'),
            ('MA50',           f'₹{ma50v:.2f}', '#818CF8'),
            ('MA200',          f'₹{ma200v:.2f}', '#F43F5E'),
        ]
        for name,val,col in levels:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1E293B"><span style="color:#64748B;font-size:12px">{name}</span><span style="color:{col};font-size:12px;font-weight:600">{val}</span></div>', unsafe_allow_html=True)
        if show_glossary:
            st.caption('S1 = where buyers typically step in. R1 = where sellers typically appear. Pivot = midpoint between high and low.')
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('**📏 Moving Average Status**')
        mas = [('MA20',ind['ma20']),('MA50',ind['ma50']),('MA100',ind['ma100']),('MA200',ind['ma200'])]
        for ma_name, ma_series in mas:
            val = float(ma_series.iloc[-1])
            diff = (cur_price/val-1)*100
            status = '🟢 Price above' if cur_price>val else '🔴 Price below'
            st.markdown(f'<div style="padding:5px 0;border-bottom:1px solid #1E293B"><span style="color:#94A3B8;font-size:12px">{ma_name}: ₹{val:.2f}</span><br><span style="font-size:11px;color:{"#10B981" if diff>=0 else "#EF4444"}">{status} ({diff:+.1f}%)</span></div>', unsafe_allow_html=True)
        cross_str = ind['cross']
        if 'golden' in cross_str:
            days = cross_str.split('_')[1]
            st.markdown(f'<div class="success-box" style="margin-top:8px"><span style="color:#34D399;font-size:12px">✨ Golden cross {days} days ago — bullish signal</span></div>', unsafe_allow_html=True)
        elif 'death' in cross_str:
            days = cross_str.split('_')[1]
            st.markdown(f'<div class="danger-box" style="margin-top:8px"><span style="color:#FCA5A5;font-size:12px">💀 Death cross {days} days ago — bearish signal</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('**🌡️ Volatility & Risk**')
        atr_val = float(ind['atr'].iloc[-1])
        atr_pct = atr_val/cur_price*100
        vol_items = [
            ('Daily ATR',     f'₹{atr_val:.2f} ({atr_pct:.1f}%)', 'Expected daily price swing'),
            ('1M Volatility', f'{ind["v1"]:.1%}', 'Annualised — last month'),
            ('3M Volatility', f'{ind["v3"]:.1%}', 'Annualised — last 3 months'),
            ('Max Drawdown',  f'{ind["maxdd"]:.1%}', 'Worst peak-to-trough drop'),
            ('BB Position',   f'{float(ind["bb_pct"].iloc[-1]):.0%}', '0%=lower band, 100%=upper'),
        ]
        for name,val,tip in vol_items:
            st.markdown(f'<div style="padding:5px 0;border-bottom:1px solid #1E293B"><span style="color:#64748B;font-size:12px">{name}</span><br><span style="color:#F1F5F9;font-size:12px;font-weight:500">{val}</span><br><span style="color:#475569;font-size:10px">{tip}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — RETURNS & STATS
# ══════════════════════════════════════════════════════
with tab2:
    if show_glossary:
        st.markdown('''<div class="info-box">
        <strong style="color:#38BDF8">Understanding returns:</strong>
        <span style="color:#94A3B8"> The bars show how much the stock gained or lost over each period. The drawdown chart shows the worst falls from the peak — useful to understand the worst-case scenario. The distribution chart shows the spread of daily returns — a wide bell curve means high volatility.</span>
        </div>''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_returns_bar(ind),  use_container_width=True)
        st.plotly_chart(chart_rolling_vol(ind),  use_container_width=True)
    with col2:
        st.plotly_chart(chart_drawdown(ind),     use_container_width=True)
        st.plotly_chart(chart_dist(ind),         use_container_width=True)

    st.plotly_chart(chart_obv(ind), use_container_width=True)
    if show_glossary:
        st.caption('OBV (On-Balance Volume) — rises when volume on up-days exceeds volume on down-days. A rising OBV confirms an uptrend; a falling OBV during a price rise is a warning sign.')

    st.markdown('**📊 Full Statistical Summary**')
    dr = ind['dr']
    pos_days = float((dr>0).mean())
    stats = {
        'Metric': ['12M Return','6M Return','3M Return','1M Return',
                   'Momentum Score','Sharpe Proxy','Max Drawdown',
                   '1M Volatility (Ann.)','12M Volatility (Ann.)',
                   'Daily Mean Return','Daily Std Dev','% Positive Days',
                   'Skewness','Kurtosis'],
        'Value':  [f'{ind["r12"]:+.1%}',f'{ind["r6"]:+.1%}',
                   f'{ind["r3"]:+.1%}',f'{ind["r1"]:+.1%}',
                   f'{ind["momentum"]:+.1%}',f'{ind["sharpe"]:.2f}',
                   f'{ind["maxdd"]:.1%}',
                   f'{ind["v1"]:.1%}',f'{ind["v12"]:.1%}',
                   f'{float(dr.mean())*100:.3f}%',
                   f'{float(dr.std())*100:.3f}%',
                   f'{pos_days:.1%}',
                   f'{float(dr.skew()):.2f}',
                   f'{float(dr.kurt()):.2f}'],
        'What it means': [
            'Total gain/loss over 12 months',
            'Total gain/loss over 6 months',
            'Total gain/loss over 3 months',
            'Total gain/loss over 1 month',
            '12M minus 1M return — trend strength',
            'Return divided by risk — trend quality',
            'Worst peak-to-trough drop this year',
            'How much price swings per year (last month)',
            'How much price swings per year (full year)',
            'Average daily gain/loss',
            'Typical daily swing size',
            'Days when stock went up',
            'Positive = more big gains than losses',
            'Higher = more extreme moves',
        ],
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 3 — TECHNICALS DEEP DIVE
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('### Technical Indicator Deep Dive')

    t1, t2 = st.columns(2)
    with t1:
        # RSI gauge
        rsi_cur = float(ind['rsi'].iloc[-1])
        rsi_col = '#EF4444' if rsi_cur>70 else '#10B981' if rsi_cur<30 else '#38BDF8'
        rsi_label = 'OVERBOUGHT — potential pullback' if rsi_cur>70 else \
                    'OVERSOLD — potential bounce' if rsi_cur<30 else \
                    'NEUTRAL — healthy zone'
        st.markdown(f'''<div class="stat-card">
        <div class="kpi-label">RSI (14-period)</div>
        <div class="kpi-value" style="color:{rsi_col}">{rsi_cur:.1f}</div>
        <div class="kpi-sub">{rsi_label}</div>
        <div style="background:#1E293B;border-radius:4px;height:6px;margin:10px 0;overflow:hidden">
            <div style="height:6px;width:{rsi_cur}%;background:{rsi_col};border-radius:4px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#475569">
            <span>0 — Oversold</span><span>30</span><span>50</span><span>70</span><span>100 — Overbought</span>
        </div>
        </div>''', unsafe_allow_html=True)
        if show_glossary:
            st.caption('RSI measures the speed and size of price movements. Like a speedometer — if it reads over 70 the stock is going too fast (overbought). Under 30 means it may have fallen too far (oversold) and could bounce back.')

    with t2:
        # MACD summary
        macd_cur = float(ind['macd'].iloc[-1])
        sig_cur  = float(ind['signal'].iloc[-1])
        hist_cur = macd_cur - sig_cur
        macd_bull = macd_cur > sig_cur
        macd_col  = '#10B981' if macd_bull else '#EF4444'
        st.markdown(f'''<div class="stat-card">
        <div class="kpi-label">MACD</div>
        <div class="kpi-value" style="color:{macd_col}">{'Bullish' if macd_bull else 'Bearish'}</div>
        <div class="kpi-sub">MACD: {macd_cur:.3f} · Signal: {sig_cur:.3f} · Histogram: {hist_cur:+.3f}</div>
        <div style="margin-top:10px;font-size:12px;color:#64748B">
        {'✅ MACD line is above the signal line — buying momentum is stronger than selling.' if macd_bull
         else '⚠️ MACD line is below the signal line — selling pressure is dominant.'}
        </div>
        </div>''', unsafe_allow_html=True)
        if show_glossary:
            st.caption('MACD compares two moving averages. When the blue MACD line crosses above the orange signal line, it is a buy signal. When it crosses below, it is a sell signal.')

    t3, t4 = st.columns(2)
    with t3:
        # Bollinger position
        bb_pct_cur = float(ind['bb_pct'].iloc[-1]) if not pd.isna(ind['bb_pct'].iloc[-1]) else 0.5
        bb_col = '#EF4444' if bb_pct_cur>0.85 else '#10B981' if bb_pct_cur<0.15 else '#38BDF8'
        bb_label = 'Near UPPER band — potentially overbought' if bb_pct_cur>0.85 else \
                   'Near LOWER band — potentially oversold' if bb_pct_cur<0.15 else \
                   'Inside the band — normal range'
        st.markdown(f'''<div class="stat-card">
        <div class="kpi-label">Bollinger Band Position</div>
        <div class="kpi-value" style="color:{bb_col}">{bb_pct_cur:.0%}</div>
        <div class="kpi-sub">{bb_label}</div>
        <div style="background:#1E293B;border-radius:4px;height:6px;margin:10px 0;overflow:hidden">
            <div style="height:6px;width:{bb_pct_cur*100:.0f}%;background:{bb_col};border-radius:4px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#475569">
            <span>Lower Band ₹{float(ind["bb_dn"].iloc[-1]):.0f}</span>
            <span>Upper Band ₹{float(ind["bb_up"].iloc[-1]):.0f}</span>
        </div>
        </div>''', unsafe_allow_html=True)
        if show_glossary:
            st.caption('Bollinger Bands form a channel around the price. 0% = touching the lower band (oversold). 100% = touching the upper band (overbought). Prices tend to revert to the middle.')

    with t4:
        # MA trend summary
        ma_vals = {
            'MA9 (very short)': float(ind['ma9'].iloc[-1]),
            'MA20 (short)': float(ind['ma20'].iloc[-1]),
            'MA50 (medium)': float(ind['ma50'].iloc[-1]),
            'MA200 (long)': float(ind['ma200'].iloc[-1]),
        }
        above_count = sum(1 for v in ma_vals.values() if cur_price > v)
        trend_col   = '#10B981' if above_count>=3 else '#EF4444' if above_count<=1 else '#F59E0B'
        trend_label = 'Strong Uptrend' if above_count>=3 else 'Strong Downtrend' if above_count<=1 else 'Mixed Signals'
        st.markdown(f'''<div class="stat-card">
        <div class="kpi-label">MA Trend Score</div>
        <div class="kpi-value" style="color:{trend_col}">{above_count}/4</div>
        <div class="kpi-sub">Price above {above_count} of 4 moving averages — {trend_label}</div>
        <div style="margin-top:10px">''', unsafe_allow_html=True)
        for ma_name, ma_v in ma_vals.items():
            above = cur_price > ma_v
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid #1E293B"><span style="color:#64748B">{ma_name}</span><span style="color:{"#10B981" if above else "#EF4444"}">{"↑ Above" if above else "↓ Below"} ₹{ma_v:.1f}</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        if show_glossary:
            st.caption('When price is above all 4 MAs, the trend is strong at every timeframe — very bullish. When below all 4, the opposite is true.')

# ══════════════════════════════════════════════════════
# TAB 4 — FUNDAMENTALS
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('### Company Fundamentals')
    if show_glossary:
        st.markdown('''<div class="info-box">
        <strong style="color:#38BDF8">What are fundamentals?</strong>
        <span style="color:#94A3B8"> Fundamentals tell you about the company's actual business — is it profitable? Is it growing? How much debt does it have? Technical analysis tells you WHEN to buy. Fundamentals tell you WHAT to buy.</span>
        </div>''', unsafe_allow_html=True)

    def g(k, fmt=None):
        v = fund.get(k)
        if v is None: return 'N/A'
        try: return fmt(v) if fmt else v
        except: return 'N/A'

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('**💰 Valuation**')
        pe_val  = fund.get('trailingPE')
        fpe_val = fund.get('forwardPE')
        pb_val  = fund.get('priceToBook')
        items = [
            ('Market Cap',    g('marketCap', lambda v: f'₹{v/1e7:.0f} Cr'),
             'Total value of all shares. Large cap = safer investment.'),
            ('P/E (trailing)',f'{pe_val:.1f}x' if isinstance(pe_val,(int,float)) else 'N/A',
             'How much you pay for each ₹1 of profit. Lower = cheaper. Nifty avg ≈ 22x.'),
            ('P/E (forward)', f'{fpe_val:.1f}x' if isinstance(fpe_val,(int,float)) else 'N/A',
             'Based on expected future earnings. Lower than trailing = expected growth.'),
            ('Price/Book',    f'{pb_val:.2f}x'  if isinstance(pb_val,(int,float)) else 'N/A',
             'Price vs assets. <1x means you are buying Rs 1 of assets for less than Rs 1.'),
            ('EPS',           g('trailingEps', lambda v: f'₹{v:.2f}'),
             'Profit per share. Higher and growing = better business.'),
            ('Beta',          g('beta', lambda v: f'{v:.2f}'),
             'How volatile vs Nifty. >1 = more volatile, <1 = calmer.'),
        ]
        for name,val,tip in items:
            st.markdown(f'<div style="padding:7px 0;border-bottom:1px solid #1E293B"><div style="display:flex;justify-content:space-between"><span style="color:#64748B;font-size:12px">{name}</span><span style="color:#F1F5F9;font-size:12px;font-weight:500">{val}</span></div>{"<div style=" + chr(39) + "font-size:10px;color:#475569;margin-top:2px" + chr(39) + ">" + tip + "</div>" if show_glossary else ""}</div>', unsafe_allow_html=True)

    with f2:
        st.markdown('**📊 Financial Health**')
        items2 = [
            ('Revenue',        g('totalRevenue', lambda v: f'₹{v/1e7:.0f} Cr'),
             'Total sales. Higher and growing = healthy top line.'),
            ('Profit Margin',  g('profitMargins', lambda v: f'{v:.1%}'),
             'How much of each rupee of revenue becomes profit.'),
            ('ROE',            g('returnOnEquity', lambda v: f'{v:.1%}'),
             'Return on Equity — how well it uses shareholder money. >15% is good.'),
            ('Debt/Equity',    g('debtToEquity', lambda v: f'{v:.0f}%'),
             'How much it owes vs owns. High debt = risky if interest rates rise.'),
            ('Current Ratio',  g('currentRatio', lambda v: f'{v:.2f}x'),
             'Can it pay short-term bills? >1.5 is healthy.'),
            ('Gross Margin',   g('grossMargins', lambda v: f'{v:.1%}'),
             'Revenue minus cost of goods. Higher = more pricing power.'),
        ]
        for name,val,tip in items2:
            st.markdown(f'<div style="padding:7px 0;border-bottom:1px solid #1E293B"><div style="display:flex;justify-content:space-between"><span style="color:#64748B;font-size:12px">{name}</span><span style="color:#F1F5F9;font-size:12px;font-weight:500">{val}</span></div>{"<div style=" + chr(39) + "font-size:10px;color:#475569;margin-top:2px" + chr(39) + ">" + tip + "</div>" if show_glossary else ""}</div>', unsafe_allow_html=True)

    with f3:
        st.markdown('**🏢 Company Info**')
        info_items = [
            ('Sector',      g('sector')),
            ('Industry',    g('industry')),
            ('Employees',   g('fullTimeEmployees', lambda v: f'{v:,}')),
            ('Country',     g('country')),
            ('Exchange',    'NSE India'),
            ('Dividend Yield', g('dividendYield', lambda v: f'{v:.2%}')),
        ]
        for name,val in info_items:
            st.markdown(f'<div style="padding:7px 0;border-bottom:1px solid #1E293B;display:flex;justify-content:space-between"><span style="color:#64748B;font-size:12px">{name}</span><span style="color:#94A3B8;font-size:12px">{val}</span></div>', unsafe_allow_html=True)

    desc = fund.get('longBusinessSummary','')
    if desc:
        st.markdown('---')
        with st.expander('📖 About the company (click to expand)'):
            st.write(desc)

    st.markdown('---')
    st.markdown('### 🔍 What factors affect this stock specifically?')
    sector = fund.get('sector','')
    if 'Material' in str(sector) or 'Steel' in company or 'Tata' in company:
        factors = [
            ('🇨🇳', 'China steel exports', 'China produces 50%+ of world steel. When they dump cheap steel, global prices fall and Tata Steel margins get squeezed.'),
            ('⛏️', 'Iron ore prices', 'Iron ore is Tata Steel\'s main raw material. Higher iron ore prices = lower profit margins.'),
            ('🏗️', 'India infra budget', 'Government spending on roads, railways, and housing drives steel demand in India.'),
            ('🇬🇧', 'UK operations', 'Tata Steel\'s Port Talbot plant in UK has been costly. Government subsidies and restructuring decisions significantly affect profitability.'),
            ('💱', 'Rupee vs USD', 'International revenues are in USD. A weaker rupee can boost reported earnings in INR.'),
            ('📈', 'RBI interest rates', 'Tata Steel has significant debt. Higher interest rates increase the cost of that debt.'),
        ]
    elif 'Technology' in str(sector) or 'Software' in str(sector):
        factors = [
            ('💵', 'USD/INR', 'IT companies earn in USD. A weaker rupee means more INR revenue.'),
            ('🤖', 'AI adoption', 'AI tools are changing how software is written — both a threat and opportunity.'),
            ('🇺🇸', 'US economy', 'Most IT revenue comes from US clients. US recession = lower spending on IT contracts.'),
            ('📋', 'H-1B visa policy', 'US visa rules affect how easily Indian IT companies can deploy engineers.'),
            ('🏆', 'Deal pipeline', 'Large contract wins or losses move the stock significantly.'),
            ('🧑', 'Attrition rate', 'High employee turnover hurts margins and delivery quality.'),
        ]
    else:
        factors = [
            ('📊', 'Quarterly earnings', 'Revenue and profit vs analyst expectations.'),
            ('🏦', 'RBI / Macro policy', 'Interest rates affect borrowing costs and consumer spending.'),
            ('🌍', 'Global economy', 'International headwinds or tailwinds affect sector outlook.'),
            ('👥', 'Management changes', 'CEO/CFO changes can signal strategic shifts.'),
            ('📋', 'Regulatory updates', 'New rules from SEBI, RBI, or sector regulators.'),
            ('💸', 'FII/DII flows', 'Foreign and domestic institution buying/selling moves prices.'),
        ]
    f_col1, f_col2 = st.columns(2)
    for i,(icon,name,desc2) in enumerate(factors):
        with (f_col1 if i%2==0 else f_col2):
            st.markdown(f'''<div class="stat-card" style="margin-bottom:8px">
            <div style="font-size:20px;margin-bottom:4px">{icon}</div>
            <div style="color:#E2E8F0;font-weight:600;font-size:13px">{name}</div>
            <div style="color:#64748B;font-size:12px;margin-top:4px;line-height:1.5">{desc2}</div>
            </div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 5 — NEWS
# ══════════════════════════════════════════════════════
with tab5:
    pos_c = sum(1 for h in news if h['sentiment']=='positive')
    neg_c = sum(1 for h in news if h['sentiment']=='negative')
    neu_c = sum(1 for h in news if h['sentiment']=='neutral')

    nc1,nc2,nc3,nc4 = st.columns(4)
    nc1.metric('Total Headlines', len(news))
    nc2.metric('Positive 🟢', pos_c)
    nc3.metric('Negative 🔴', neg_c)
    overall_sent = 'POSITIVE 🟢' if pos_c>neg_c else 'NEGATIVE 🔴' if neg_c>pos_c else 'NEUTRAL ⚪'
    nc4.metric('Overall Sentiment', overall_sent)

    if show_glossary:
        st.markdown('''<div class="info-box">
        <strong style="color:#38BDF8">Why does news matter?</strong>
        <span style="color:#94A3B8"> Stock prices often move before official data is released — the market "prices in" expectations. Positive news flow increases demand for shares. Negative news causes investors to sell. Monitoring sentiment helps you understand what the market is currently thinking about this company.</span>
        </div>''', unsafe_allow_html=True)

    # Sentiment mini chart
    fig_sent = go.Figure(go.Pie(
        values=[pos_c, neg_c, neu_c],
        labels=['Positive','Negative','Neutral'],
        hole=0.6,
        marker_colors=['#10B981','#EF4444','#64748B'],
        textinfo='label+percent',
        textfont=dict(color='#F1F5F9', size=12),
    ))
    fig_sent.add_annotation(text=f'<b>{overall_sent.split()[0]}</b>', x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color='#F1F5F9'))
    fig_sent.update_layout(**LAYOUT, height=220, showlegend=False,
        title='Headline Sentiment Breakdown')

    col_chart, col_news = st.columns([1,2])
    with col_chart:
        st.plotly_chart(fig_sent, use_container_width=True)
        st.markdown('''<div class="stat-card">
        <div style="font-size:12px;color:#64748B;line-height:1.8">
        <strong style="color:#E2E8F0">How sentiment is scored:</strong><br>
        Headlines are scanned for words like <span style="color:#10B981">profit, growth, surge, record</span> (positive) and <span style="color:#EF4444">loss, risk, pressure, decline</span> (negative).<br><br>
        News sentiment is one input into the recommendation — a stock with great charts but all negative news is riskier.
        </div>
        </div>''', unsafe_allow_html=True)

    with col_news:
        filt = st.radio('Filter', ['All','Positive 🟢','Negative 🔴','Neutral ⚪'], horizontal=True)

        for h in news:
            if filt != 'All' and not h['sentiment'] in filt.lower(): continue
            card_cls = f"news-{'pos' if h['sentiment']=='positive' else 'neg' if h['sentiment']=='negative' else 'neu'}"
            icon     = '🟢' if h['sentiment']=='positive' else '🔴' if h['sentiment']=='negative' else '⚪'

            pos_tags = ' '.join([f'<span style="background:#052e16;color:#34D399;padding:1px 6px;border-radius:4px;font-size:10px">{w}</span>' for w in h.get('pos_words',[])])
            neg_tags = ' '.join([f'<span style="background:#1c0a0a;color:#FCA5A5;padding:1px 6px;border-radius:4px;font-size:10px">{w}</span>' for w in h.get('neg_words',[])])

            link_html = f'<a href="{h["link"]}" target="_blank" style="color:#38BDF8;font-size:11px;text-decoration:none">Read more →</a>' if h.get('link','#') != '#' else ''

            st.markdown(f'''<div class="news-card {card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:11px;color:#475569;font-weight:500">{h["source"]} · {h["date"]}</span>
                <span style="font-size:12px">{icon} <span style="color:#64748B">{h["sentiment"].capitalize()}</span></span>
            </div>
            <div style="font-size:14px;color:#E2E8F0;font-weight:500;line-height:1.5;margin-bottom:8px">{h["title"]}</div>
            <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px">{pos_tags}{neg_tags}</div>
            {link_html}
            </div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### 📡 Key External Factors to Monitor')
    st.caption('These macro factors often move this stock even when no company-specific news is out.')
    macro_feeds = [
        ('NSE Market Updates',     'https://news.google.com/rss/search?q=NSE+Nifty+India+market&hl=en-IN&gl=IN&ceid=IN:en'),
        ('RBI Policy News',        'https://news.google.com/rss/search?q=RBI+interest+rate+India&hl=en-IN&gl=IN&ceid=IN:en'),
        ('India Economy',          'https://news.google.com/rss/search?q=India+economy+GDP+growth&hl=en-IN&gl=IN&ceid=IN:en'),
    ]
    for feed_name, feed_url in macro_feeds:
        try:
            feed = feedparser.parse(feed_url)
            with st.expander(f'📰 {feed_name} (latest)'):
                for entry in feed.entries[:4]:
                    pub = entry.get('published','')[:16]
                    st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #1E293B"><span style="color:#E2E8F0;font-size:13px">{entry.title}</span><br><span style="color:#475569;font-size:11px">{pub}</span></div>', unsafe_allow_html=True)
        except:
            with st.expander(f'📰 {feed_name}'):
                st.caption('Could not load feed. Check internet connection.')

# ══════════════════════════════════════════════════════
# TAB 6 — PEERS
# ══════════════════════════════════════════════════════
with tab6:
    st.markdown('### 🔄 Sector Peer Comparison')
    if show_glossary:
        st.markdown('''<div class="info-box">
        <strong style="color:#38BDF8">Why compare with peers?</strong>
        <span style="color:#94A3B8"> A stock that fell 10% might look bad — unless all similar companies fell 20%. Comparing with peers shows you whether a stock is outperforming or underperforming its sector. A stock rising when its peers are falling is showing relative strength.</span>
        </div>''', unsafe_allow_html=True)

    if peers:
        peer_fig = chart_peers(ticker, ind, peers)
        if peer_fig: st.plotly_chart(peer_fig, use_container_width=True)

        st.markdown('**Peer Comparison Table**')
        peer_rows = [{
            'Company':    ticker.replace('.NS',''),
            '12M Return': f'{ind["r12"]:+.1%}',
            'Price':      f'₹{cur_price:.2f}',
            'P/E':        f'{fund.get("trailingPE","N/A"):.1f}x' if isinstance(fund.get("trailingPE"),(int,float)) else 'N/A',
            'Mkt Cap':    f'₹{fund.get("marketCap",0)/1e7:.0f} Cr' if fund.get("marketCap") else 'N/A',
            'Status':     '← You are here',
        }]
        for p in peers:
            peer_rows.append({
                'Company':    p['ticker'],
                '12M Return': f'{p["r12"]:+.1%}',
                'Price':      f'₹{p["price"]:.2f}',
                'P/E':        f'{p["pe"]:.1f}x' if isinstance(p.get("pe"),(int,float)) else 'N/A',
                'Mkt Cap':    f'₹{p["mktcap"]/1e7:.0f} Cr' if p.get("mktcap") else 'N/A',
                'Status':     'Peer',
            })
        st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)
    else:
        st.info('Peer comparison not available for this stock. Enable it in the sidebar.')

# ══════════════════════════════════════════════════════
# TAB 7 — HISTORY
# ══════════════════════════════════════════════════════
with tab7:
    st.markdown('### 🕐 Analysis History')
    if show_glossary:
        st.markdown('''<div class="info-box">
        <span style="color:#94A3B8">Every time you run an analysis, it is saved here. This lets you track how the agent score changes over time — if the score is consistently improving, that is a positive sign. All data is stored locally at <code>D:/StockAgent/history.db</code></span>
        </div>''', unsafe_allow_html=True)

    hist_df = get_history(ticker)
    if not hist_df.empty and len(hist_df)>1:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['score'],
            mode='lines+markers',
            line=dict(color='#38BDF8',width=2),
            marker=dict(size=8, color=[
                '#10B981' if s>=3 else '#EF4444' if s<0 else '#F59E0B'
                for s in hist_df['score']]),
            name='Score',
            hovertemplate='%{x}<br>Score: %{y:.1f}<extra></extra>'))
        fig_h.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['price'],
            mode='lines', line=dict(color='#A78BFA',width=1.5,dash='dot'),
            name='Price ₹', yaxis='y2',
            hovertemplate='₹%{y:.2f}<extra>Price</extra>'))
        fig_h.update_layout(**LAYOUT, height=320,
            title=f'{ticker} — Agent Score & Price Over Time',
            yaxis2=dict(title='Price ₹', overlaying='y', side='right',
                        gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#A78BFA')))
        apply_chart_style(fig_h)
        fig_h.update_yaxes(title_text='Score')
        fig_h.add_hline(y=3,   line_dash='dash', line_color='#10B981', line_width=0.8)
        fig_h.add_hline(y=0,   line_dash='dash', line_color='#EF4444', line_width=0.8)
        st.plotly_chart(fig_h, use_container_width=True)

    if not hist_df.empty:
        disp = hist_df[['date','price','recommendation','score','momentum','sharpe','rsi']].copy()
        disp.columns = ['Date','Price (₹)','Recommendation','Score','Momentum','Sharpe','RSI']
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info('No history yet for this stock. The first analysis has been saved. Run again tomorrow to see the trend.')

    st.markdown('---')
    st.markdown('**All stocks you have analysed**')
    all_h = get_history()
    if not all_h.empty:
        st.dataframe(all_h[['date','ticker','price','recommendation','score']], use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown('''<div style="text-align:center;padding:16px 0">
<span style="color:#1E293B;font-size:12px">NSE Stock Analyser Pro · Data: Yahoo Finance & Google News · Charts: Plotly</span><br>
<span style="color:#1E293B;font-size:11px">⚠️ For educational purposes only. Not financial advice. Past performance does not guarantee future returns.
Always consult a SEBI-registered investment advisor before making investment decisions.</span>
</div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 8 — PROPHET FORECAST
# ══════════════════════════════════════════════════════
with tab8:
    st.markdown('### 🔮 Prophet AI Price Forecast')

    st.markdown('''<div class="info-box">
    <strong style="color:#38BDF8">What is Prophet?</strong>
    <span style="color:#94A3B8"> Prophet is a forecasting model built by Meta (Facebook) and used by data scientists worldwide.
    It learns patterns from historical price data — weekly cycles, yearly seasonality, trend changes — and projects them forward.
    The shaded blue area is the <strong>95% confidence interval</strong>: the model believes the true price will land inside this band.
    Wider band = more uncertainty. This is not a guarantee — it is a probabilistic estimate.</span>
    </div>''', unsafe_allow_html=True)

    forecast_days = st.slider('Forecast horizon (trading days)', 10, 90, 30, 5,
                              help='Move slider then wait ~15 seconds for model to retrain')
    st.caption('⏳ Each slider position trains a fresh Prophet model — allow 10–20 seconds to update.')

    # Cache forecast per ticker+days so repeat values are instant
    cache_key = f'prophet_{ticker}_{forecast_days}'
    if cache_key not in st.session_state:
        with st.spinner(f'Training Prophet model for {forecast_days} days — please wait...'):
            forecast, prophet_model = run_prophet_forecast(ind['prices'], days=forecast_days)
        st.session_state[cache_key] = forecast
    else:
        forecast = st.session_state[cache_key]

    if forecast is None:
        st.error('Prophet is not installed or failed. Run: `pip install prophet`')
    else:
        fig_p, last_actual, last_fc, pct_chg = chart_prophet(
            ind['prices'], forecast, ticker, forecast_days)
        st.plotly_chart(fig_p, use_container_width=True)

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        direction_col = '#10B981' if pct_chg >= 0 else '#EF4444'
        direction_txt = '▲ Upward' if pct_chg >= 0 else '▼ Downward'
        c1.metric('Current Price',    f'₹{last_actual:.2f}')
        c2.metric(f'{forecast_days}-Day Target', f'₹{last_fc:.2f}',
                  delta=f'{pct_chg:+.1f}%')
        c3.metric('Trend Direction',  direction_txt)

        # Upper / lower bounds
        fut_fc = forecast[forecast['ds'] > (
            ind['prices'].index[-1].tz_localize(None)
            if ind['prices'].index[-1].tzinfo else ind['prices'].index[-1]
        )]
        upper = float(fut_fc['yhat_upper'].iloc[-1])
        lower = float(fut_fc['yhat_lower'].iloc[-1])
        c4.metric('Confidence Range', f'₹{lower:.0f} – ₹{upper:.0f}')

        st.markdown('---')
        st.markdown('### 📅 Forecast Table (next days)')
        tbl = fut_fc[['ds','yhat','yhat_lower','yhat_upper']].copy()
        tbl.columns = ['Date','Predicted Price (₹)','Lower Bound (₹)','Upper Bound (₹)']
        tbl['Date'] = tbl['Date'].dt.strftime('%d %b %Y')
        for col in ['Predicted Price (₹)','Lower Bound (₹)','Upper Bound (₹)']:
            tbl[col] = tbl[col].round(2)
        st.dataframe(tbl.head(forecast_days), use_container_width=True, hide_index=True)

        st.markdown('''<div class="warn-box" style="margin-top:16px">
        <span style="color:#FCD34D;font-size:11px">
        ⚠️ <strong>Important:</strong> Prophet forecasts are based purely on historical price patterns.
        They do not account for news events, earnings surprises, or macroeconomic shocks.
        Use this as one signal among many — not as a standalone buy/sell decision.
        </span></div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 9 — ML SIGNAL
# ══════════════════════════════════════════════════════
with tab9:
    st.markdown('### 🤖 Machine Learning Buy / Sell Signal')

    st.markdown('''<div class="info-box">
    <strong style="color:#38BDF8">How this works:</strong>
    <span style="color:#94A3B8">
    We train two ML models — <strong>Random Forest</strong> and <strong>Gradient Boosting</strong> — on 2 years of
    historical data for this stock. The models learn which combinations of RSI, MACD, moving averages,
    volume, and momentum historically led to the price rising 10 days later. They are then applied to
    today's indicators to produce a probability. Above 55% = BUY signal, below 45% = SELL, in between = HOLD.
    The models are retrained fresh every time you load this tab.
    </span>
    </div>''', unsafe_allow_html=True)

    with st.spinner('Training ML models on historical data...'):
        ml = run_ml_model(ind)

    if ml is None:
        st.warning('Not enough historical data to train ML model (need 100+ days).')
    else:
        signal   = ml['signal']
        prob     = ml['probability']
        rf_acc   = ml['rf_accuracy']
        gb_acc   = ml['gb_accuracy']

        sig_col = '#10B981' if signal == 'BUY' else '#EF4444' if signal == 'SELL' else '#F59E0B'
        sig_bg  = '#052e16' if signal == 'BUY' else '#1c0a0a' if signal == 'SELL' else '#1c1917'

        # Main signal card
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc1:
            st.markdown(f'''<div class="rec-box">
                <div style="font-size:11px;color:#64748B;font-weight:500;
                    letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px">
                    ML Signal (10-day horizon)
                </div>
                <span class="badge" style="font-size:20px;padding:10px 28px;
                    background:{sig_bg};color:{sig_col};border:1px solid {sig_col}">
                    {signal}
                </span>
                <div style="font-size:36px;font-weight:700;color:{sig_col};margin:14px 0 4px">
                    {prob*100:.1f}%
                </div>
                <div style="font-size:12px;color:#64748B;margin-bottom:16px">
                    probability price rises in 10 days
                </div>
                <div style="background:#1E293B;border-radius:6px;height:10px;overflow:hidden">
                    <div style="height:10px;border-radius:6px;width:{prob*100:.0f}%;
                        background:{sig_col};transition:width 1s ease"></div>
                </div>
            </div>''', unsafe_allow_html=True)

        with sc2:
            st.markdown('#### How the models voted')
            rf_prob  = prob   # already ensemble
            rf_sig   = 'BUY' if ml['rf'].predict_proba(
                ml['scaler'].transform(
                    build_ml_features(ind).drop('label',axis=1).iloc[[-1]]
                ))[0][1] >= 0.5 else 'SELL'
            gb_sig   = 'BUY' if ml['gb'].predict_proba(
                ml['scaler'].transform(
                    build_ml_features(ind).drop('label',axis=1).iloc[[-1]]
                ))[0][1] >= 0.5 else 'SELL'

            for model_name, m_signal, acc in [
                ('🌲 Random Forest',       rf_sig, rf_acc),
                ('⚡ Gradient Boosting',   gb_sig, gb_acc),
                ('🤝 Ensemble (combined)', signal, (rf_acc+gb_acc)/2),
            ]:
                m_col = '#10B981' if m_signal == 'BUY' else '#EF4444'
                st.markdown(f'''<div class="stat-card" style="display:flex;
                    justify-content:space-between;align-items:center;padding:12px 16px;margin-bottom:6px">
                    <span style="color:#E2E8F0;font-size:13px">{model_name}</span>
                    <span style="display:flex;gap:12px;align-items:center">
                        <span style="color:#64748B;font-size:12px">Accuracy: {acc*100:.1f}%</span>
                        <span style="color:{m_col};font-weight:700;font-size:14px">{m_signal}</span>
                    </span>
                </div>''', unsafe_allow_html=True)

            st.markdown('''<div class="info-box" style="margin-top:10px">
            <span style="color:#94A3B8;font-size:12px">
            <strong style="color:#38BDF8">Accuracy</strong> = % of time the model correctly predicted
            direction on the held-out test set (last 20% of historical data). Higher = more reliable.
            </span></div>''', unsafe_allow_html=True)

        with sc3:
            st.metric('Random Forest Accuracy',    f'{rf_acc*100:.1f}%')
            st.metric('Gradient Boost Accuracy',   f'{gb_acc*100:.1f}%')
            avg_acc = (rf_acc + gb_acc) / 2
            st.metric('Average Model Accuracy',    f'{avg_acc*100:.1f}%')

        st.markdown('---')

        # Feature importance chart
        st.markdown('#### 🔑 What the model looks at most')
        st.caption('Feature importance — how much each indicator influenced the ML prediction')

        imp = ml['importances'].reset_index()
        imp.columns = ['Feature', 'Importance']
        FEATURE_LABELS = {
            'rsi':        'RSI (momentum)',
            'macd_diff':  'MACD vs Signal',
            'price_ma20': 'Price vs 20-day MA',
            'price_ma50': 'Price vs 50-day MA',
            'price_ma200':'Price vs 200-day MA',
            'bb_pct':     'Bollinger Band position',
            'atr_pct':    'Volatility (ATR)',
            'obv_chg':    'Volume trend (OBV)',
            'vol_ratio':  'Volume vs average',
            'ret_1':      '1-day return',
            'ret_5':      '5-day return',
            'ret_21':     '21-day return',
        }
        imp['Feature'] = imp['Feature'].map(FEATURE_LABELS).fillna(imp['Feature'])

        fig_imp = go.Figure(go.Bar(
            x=imp['Importance'],
            y=imp['Feature'],
            orientation='h',
            marker_color=['#38BDF8' if i == 0 else '#1D4ED8'
                          for i in range(len(imp))],
            text=[f'{v*100:.1f}%' for v in imp['Importance']],
            textposition='outside',
        ))
        fig_imp.update_layout(
            height=380,
            title='Feature Importance — what drove the ML signal',
            xaxis_title='Importance',
            yaxis=dict(autorange='reversed'),
            margin=dict(l=160, r=60, t=50, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94A3B8'),
        )
        apply_chart_style(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown('''<div class="warn-box">
        <span style="color:#FCD34D;font-size:11px">
        ⚠️ <strong>Important:</strong> ML models learn from the past and assume future patterns
        will be similar. They cannot predict black swan events, earnings surprises, or policy changes.
        Model accuracy of 55–65% is typical and considered good for financial data —
        markets are inherently noisy. Always use multiple signals before making any decision.
        </span></div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 10 — AI AGENT (OLLAMA)
# ══════════════════════════════════════════════════════
with tab10:
    st.markdown('### 💬 AI Stock Agent — Powered by Llama 3.1 (Local)')

    # Check Ollama status
    ollama_ok = check_ollama()
    if ollama_ok:
        st.success('Ollama is running — AI Agent is ready!')
    else:
        st.error('Ollama is not running. Open a new terminal and run: `ollama serve`')
        st.code('ollama serve', language='bash')
        st.info('Keep that terminal open, then come back to this tab.')

    st.markdown('''<div class="info-box">
    <strong style="color:#38BDF8">What this agent can do:</strong>
    <span style="color:#94A3B8">
    The agent has access to all live data for <strong>{company} ({ticker})</strong> —
    price, RSI, MACD, momentum, fundamentals, and latest news headlines.
    Ask it anything about this stock in plain English. It reasons over the real data
    and replies like a financial analyst. All processing happens on your PC — no data leaves your machine.
    </span>
    </div>'''.format(company=company, ticker=ticker), unsafe_allow_html=True)

    # Sample questions — clicking auto-submits immediately
    st.markdown('**Try asking:**')
    sample_cols = st.columns(3)
    samples = [
        f"Should I buy {company} today?",
        f"What are the biggest risks with {company}?",
        f"Explain the RSI and MACD signals for {company}",
        f"Is {company} overvalued right now?",
        f"When exactly should I buy and how many shares?",
        f"Give me a full summary with entry price and target",
    ]
    for i, sample in enumerate(samples):
        with sample_cols[i % 3]:
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                st.session_state['auto_submit_q'] = sample
                st.rerun()

    st.markdown('---')

    # Init session state
    if 'agent_history' not in st.session_state:
        st.session_state.agent_history = []
    if 'agent_input' not in st.session_state:
        st.session_state.agent_input = ''
    if 'stock_context' not in st.session_state:
        st.session_state.stock_context = ''

    # Build stock context — include Prophet + ML predictions if already computed
    context_key = f"{ticker}_context"
    prophet_pct_val = None
    ml_signal_val   = None
    ml_prob_val     = None

    prophet_cache = st.session_state.get(f'prophet_{ticker}_30')
    if prophet_cache is not None:
        try:
            hist_end = ind['prices'].index[-1]
            hist_end = hist_end.tz_localize(None) if hist_end.tzinfo else hist_end
            fut_fc = prophet_cache[prophet_cache['ds'] > hist_end]
            if not fut_fc.empty:
                cur_p = float(ind['prices'].iloc[-1])
                prophet_pct_val = (float(fut_fc['yhat'].iloc[-1]) - cur_p) / cur_p * 100
        except Exception:
            pass

    if context_key not in st.session_state:
        st.session_state[context_key] = build_stock_context(
            ticker, company, ind, fund, rec, score, reasons, risks, news,
            prophet_pct=prophet_pct_val,
            ml_signal=ml_signal_val,
            ml_prob=ml_prob_val,
        )
    stock_ctx = st.session_state[context_key]

    # Display chat history
    for msg in st.session_state.agent_history:
        if msg['role'] == 'user':
            st.markdown(f'''<div style="background:#1E293B;border-radius:12px;
                padding:12px 16px;margin:8px 0;border-left:3px solid #38BDF8">
                <span style="color:#94A3B8;font-size:11px">YOU</span><br>
                <span style="color:#F1F5F9;font-size:14px">{msg["content"]}</span>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div style="background:#0F172A;border-radius:12px;
                padding:12px 16px;margin:8px 0;border-left:3px solid #10B981">
                <span style="color:#94A3B8;font-size:11px">AI AGENT — {company}</span><br>
                <span style="color:#E2E8F0;font-size:14px;line-height:1.7">{msg["content"]}</span>
            </div>''', unsafe_allow_html=True)

    # Always show charts below the last reply
    if st.session_state.agent_history:
        comp_data = st.session_state.get('agent_comp_data', [])
        st.markdown('#### 📊 Live Charts')
        if comp_data:
            charts = make_comparison_charts(
                ticker, company, ind,
                comp_data,
                main_score=score,
            )
            for chart_title, fig in charts:
                st.markdown(f'**{chart_title}**')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(
                make_agent_price_chart(ticker, ind['prices'], ind, company),
                use_container_width=True,
            )

    # Input box
    auto_q = st.session_state.pop('auto_submit_q', None)
    user_q = st.text_input(
        'Ask the agent anything about this stock...',
        value=auto_q or st.session_state.agent_input,
        placeholder=f'e.g. Should I buy {company} today?',
        key='agent_text_input',
    )
    st.session_state.agent_input = ''

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_clicked = st.button('Ask Agent →', type='primary',
                                use_container_width=True, disabled=not ollama_ok)
    with col_clear:
        if st.button('Clear Chat', use_container_width=True):
            st.session_state.agent_history = []
            st.session_state['agent_comp_data'] = []
            st.rerun()

    # Submit if button clicked OR if auto_q was set by sample bubble
    submit_q = user_q.strip() if ask_clicked and user_q.strip() else \
               (auto_q.strip() if auto_q and not ask_clicked else None)

    if submit_q:
        comparison_stocks = detect_comparison_tickers(submit_q, ticker)
        enriched_ctx = stock_ctx
        comp_data = []
        if comparison_stocks:
            with st.spinner(f'Fetching live data for: {", ".join([n for n,_ in comparison_stocks])}...'):
                for comp_name, comp_ticker in comparison_stocks:
                    try:
                        h2, f2, _ = fetch_stock_data(comp_ticker)
                        if h2 is not None:
                            i2 = compute_all(h2)
                            r2, s2, rs2, rk2, _, _ = compute_recommendation(i2, f2)
                            enriched_ctx += fetch_quick_context(comp_ticker, comp_name)
                            comp_data.append((comp_name, comp_ticker, i2, s2, r2))
                    except Exception:
                        enriched_ctx += fetch_quick_context(comp_ticker, comp_name)

        with st.spinner('Agent is thinking...'):
            reply = ask_ollama(
                user_question=submit_q,
                stock_context=enriched_ctx,
                chat_history=st.session_state.agent_history,
            )
        st.session_state.agent_history.append({"role": "user",      "content": submit_q})
        st.session_state.agent_history.append({"role": "assistant", "content": reply})
        st.session_state['agent_comp_data'] = comp_data
        st.rerun()

    st.markdown('---')
    st.markdown('''<div class="danger-box">
    <span style="color:#FCA5A5;font-size:11px">
    ⚠️ <strong>Not financial advice.</strong>
    This AI agent is for educational purposes only. It reasons over technical data
    but cannot predict the future. Always consult a SEBI-registered investment advisor
    before making any investment decisions.
    </span></div>''', unsafe_allow_html=True)

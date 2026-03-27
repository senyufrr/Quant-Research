
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
import logging
import json
import os
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich import box
from rich.columns import Columns

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = ''
SECRET_KEY = ''
BASE_URL   = 'https://paper-api.alpaca.markets'

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = ['NVDA', 'AMD', 'TSLA', 'META', 'MSFT', 'AAPL', 'SPY', 'QQQ']

RISK_CONFIG = {
    'max_session_loss':       -15_000,   # Hard stop for entire session
    'max_position_risk_pct':   0.02,     # Max 2% account at risk per trade
    'max_concurrent_positions': 3,       # Never hold more than 3 at once
    'max_daily_trades':        50,       # Churn ceiling
    'atr_profit_multiplier':   5.0,      # Profit target = ATR * this
    'atr_stop_multiplier':     2.0,      # Stop loss   = ATR * this
    'trailing_stop_pct':       0.008,    # 0.8% trailing stop activation
    'min_signal_score':        65,       # Min confluence score to enter (0-100)
    'volume_surge_threshold':  1.8,      # Volume must be 1.8x average to enter
}

STRATEGY_WEIGHTS = {
    'rsi_reversal':        0.25,
    'bb_squeeze':          0.20,
    'vwap_deviation':      0.20,
    'macd_crossover':      0.20,
    'volume_surge':        0.15,
}

SCAN_INTERVAL_SECS  = 8    # How often to re-scan all tickers
LOG_FILE            = 'apex_trader.log'
STATS_FILE          = 'apex_session_stats.json'

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger('APEX')
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker:      str
    direction:   str          # 'LONG' | 'SHORT' | 'FLAT'
    score:       float        # 0–100 confluence score
    price:       float
    atr:         float
    rsi:         float
    vwap_dev:    float        # % deviation from VWAP
    bb_pct:      float        # Bollinger %B (0 = lower band, 1 = upper)
    macd_hist:   float
    volume_ratio:float        # Current vol / 20-bar avg vol
    strategy:    str          # Dominant strategy that triggered
    timestamp:   datetime = field(default_factory=datetime.now)


@dataclass
class TradeRecord:
    ticker:       str
    side:         str
    qty:          int
    entry_price:  float
    exit_price:   float = 0.0
    pnl:          float = 0.0
    duration_sec: float = 0.0
    strategy:     str = ''
    exit_reason:  str = ''


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorEngine:

    @staticmethod
    def compute(symbol: str) -> Optional[pd.Series]:
        try:
            df = yf.download(
                symbol, period='5d', interval='1m',
                progress=False, multi_level_index=False
            )
            if df.empty or len(df) < 50:
                return None

            c = df['Close']
            h = df['High']
            l = df['Low']
            v = df['Volume']

            # ── RSI (14) ──────────────────────────────────────────────────
            delta = c.diff()
            gain  = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
            loss  = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

            # ── Stochastic RSI (to filter false RSI signals) ───────────────
            rsi_min = df['RSI'].rolling(14).min()
            rsi_max = df['RSI'].rolling(14).max()
            df['StochRSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min + 1e-10)

            # ── Bollinger Bands (10, 1.5σ) — Scalping Tuned ──────────────
            df['MA10']  = c.rolling(10).mean()
            df['BB_std']= c.rolling(10).std()
            df['Upper'] = df['MA10'] + df['BB_std'] * 1.5
            df['Lower'] = df['MA10'] - df['BB_std'] * 1.5
            df['MA20']  = df['MA10']  # alias so downstream refs still work
            # Bollinger %B: 0 = lower band, 1 = upper band
            df['BB_pct'] = (c - df['Lower']) / (df['Upper'] - df['Lower'] + 1e-10)
            # Bandwidth squeeze indicator (low = coiling for breakout)
            df['BB_width'] = (df['Upper'] - df['Lower']) / df['MA20']

            # ── MACD (12/26/9) ─────────────────────────────────────────────
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            df['MACD']      = ema12 - ema26
            df['MACD_sig']  = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_sig']

            # ── ATR (14) ───────────────────────────────────────────────────
            tr = pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs()
            ], axis=1).max(axis=1)
            df['ATR'] = tr.ewm(com=13, adjust=False).mean()

            # ── VWAP (intraday reset) ──────────────────────────────────────
            df['date'] = df.index.date
            df['cum_vol']    = df.groupby('date')['Volume'].cumsum()
            df['cum_tp_vol'] = df.groupby('date').apply(
                lambda g: (((g['High'] + g['Low'] + g['Close']) / 3) * g['Volume']).cumsum()
            ).reset_index(level=0, drop=True)
            df['VWAP'] = df['cum_tp_vol'] / df['cum_vol']
            df['VWAP_dev'] = (c - df['VWAP']) / df['VWAP'] * 100  # % away from VWAP

            # ── EMAs for trend bias ────────────────────────────────────────
            df['EMA9']  = c.ewm(span=9,  adjust=False).mean()
            df['EMA21'] = c.ewm(span=21, adjust=False).mean()

            # ── Volume Surge ───────────────────────────────────────────────
            df['Vol_MA20']    = v.rolling(20).mean()
            df['Volume_ratio']= v / df['Vol_MA20']

            # ── Momentum ───────────────────────────────────────────────────
            df['Momentum'] = c.pct_change(5) * 100  # 5-bar momentum %

            return df.iloc[-1]

        except Exception as e:
            log.error(f"Indicator error for {symbol}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR — Multi-Strategy Confluence Scoring
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:

    @staticmethod
    def score(symbol: str, data: pd.Series) -> Signal:
        """
        Returns a Signal with a confluence score 0–100.
        Positive score = LONG bias. Negative = SHORT bias.
        Score magnitude = confidence.
        """
        long_score  = 0.0
        short_score = 0.0
        dominant    = 'NONE'

        rsi         = float(data['RSI'])
        stoch_rsi   = float(data['StochRSI'])
        bb_pct      = float(data['BB_pct'])
        macd_hist   = float(data['MACD_hist'])
        vwap_dev    = float(data['VWAP_dev'])
        vol_ratio   = float(data['Volume_ratio'])
        momentum    = float(data['Momentum'])
        ema_up      = data['EMA9'] > data['EMA21']
        price       = float(data['Close'])
        atr         = float(data['ATR'])

        # ── Strategy 1: RSI Reversal ───────────────────────────────────────
        # Oversold: RSI < 30 AND StochRSI < 0.2
        # Overbought: RSI > 70 AND StochRSI > 0.8
        rsi_long_sig  = max(0, (35 - rsi) / 35) * (1 if stoch_rsi < 0.25 else 0.4)
        rsi_short_sig = max(0, (rsi - 65) / 35) * (1 if stoch_rsi > 0.75 else 0.4)
        long_score  += rsi_long_sig  * STRATEGY_WEIGHTS['rsi_reversal'] * 100
        short_score += rsi_short_sig * STRATEGY_WEIGHTS['rsi_reversal'] * 100
        if rsi_long_sig > 0.5 or rsi_short_sig > 0.5:
            dominant = 'RSI_REVERSAL'

        # ── Strategy 2: Bollinger Band Extremes + Squeeze ─────────────────
        bb_long_sig  = max(0, 1 - bb_pct * 2)   if bb_pct < 0.15 else 0
        bb_short_sig = max(0, (bb_pct - 0.5) * 2) if bb_pct > 0.85 else 0
        long_score  += bb_long_sig  * STRATEGY_WEIGHTS['bb_squeeze'] * 100
        short_score += bb_short_sig * STRATEGY_WEIGHTS['bb_squeeze'] * 100
        if bb_long_sig > 0.5 or bb_short_sig > 0.5:
            dominant = 'BB_EXTREME'

        # ── Strategy 3: VWAP Deviation Mean-Reversion ─────────────────────
        # -2% from VWAP = oversold vs fair value; +2% = overbought
        vwap_long_sig  = max(0, (-vwap_dev - 0.5) / 2.5) if vwap_dev < -0.5 else 0
        vwap_short_sig = max(0, (vwap_dev  - 0.5) / 2.5) if vwap_dev >  0.5 else 0
        long_score  += vwap_long_sig  * STRATEGY_WEIGHTS['vwap_deviation'] * 100
        short_score += vwap_short_sig * STRATEGY_WEIGHTS['vwap_deviation'] * 100
        if vwap_long_sig > 0.5 or vwap_short_sig > 0.5:
            dominant = 'VWAP_DEVIATION'

        # ── Strategy 4: MACD Crossover Momentum ───────────────────────────
        # Histogram flipping positive/negative = momentum shift
        macd_long_sig  = min(1, max(0,  macd_hist / (atr * 0.1 + 1e-10)))
        macd_short_sig = min(1, max(0, -macd_hist / (atr * 0.1 + 1e-10)))
        long_score  += macd_long_sig  * STRATEGY_WEIGHTS['macd_crossover'] * 100
        short_score += macd_short_sig * STRATEGY_WEIGHTS['macd_crossover'] * 100
        if macd_long_sig > 0.5 or macd_short_sig > 0.5:
            dominant = 'MACD_MOMENTUM'

        # ── Strategy 5: Volume Surge Confirmation ─────────────────────────
        vol_boost = min(1.5, vol_ratio / RISK_CONFIG['volume_surge_threshold'])
        if vol_ratio >= RISK_CONFIG['volume_surge_threshold']:
            long_score  *= vol_boost
            short_score *= vol_boost
            dominant = 'VOLUME_SURGE' if vol_ratio > 2.5 else dominant

        # ── EMA Trend Filter ───────────────────────────────────────────────
        # Only boost signals that align with trend
        if ema_up:
            long_score  *= 1.15
            short_score *= 0.80
        else:
            short_score *= 1.15
            long_score  *= 0.80

        # ── Momentum Confirmation ──────────────────────────────────────────
        if momentum > 0.3:
            long_score  *= 1.1
        elif momentum < -0.3:
            short_score *= 1.1

        # ── Determine Direction ────────────────────────────────────────────
        long_score  = min(100, long_score)
        short_score = min(100, short_score)

        if long_score > short_score and long_score >= RISK_CONFIG['min_signal_score']:
            direction = 'LONG'
            score     = long_score
        elif short_score > long_score and short_score >= RISK_CONFIG['min_signal_score']:
            direction = 'SHORT'
            score     = short_score
        else:
            direction = 'FLAT'
            score     = max(long_score, short_score)

        return Signal(
            ticker=symbol,
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            rsi=rsi,
            vwap_dev=vwap_dev,
            bb_pct=bb_pct,
            macd_hist=macd_hist,
            volume_ratio=vol_ratio,
            strategy=dominant,
        )


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZER — Volatility-Adjusted with Kelly-Inspired Logic
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:

    @staticmethod
    def calc_qty(account_equity: float, price: float, atr: float, score: float) -> int:
        """
        Dynamic qty based on:
        - Max risk % of equity per trade
        - ATR-based stop size (risk per share = atr * stop_multiplier)
        - Signal score as a Kelly-inspired confidence scalar (0.5x–1.5x)
        """
        stop_dist   = atr * RISK_CONFIG['atr_stop_multiplier']
        risk_dollars= account_equity * RISK_CONFIG['max_position_risk_pct']
        base_qty    = int(risk_dollars / max(stop_dist, 0.01))
        # Scale qty with signal confidence (65 score = 0.8x, 100 = 1.5x)
        kelly_scalar = 0.5 + (score / 100) * 1.0
        qty          = max(1, int(base_qty * kelly_scalar))
        # Hard cap: never spend more than 30% of equity on single position
        max_qty = int((account_equity * 0.30) / max(price, 0.01))
        return min(qty, max_qty)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

class SessionState:
    def __init__(self):
        self.lock               = threading.Lock()
        self.realized_pnl       = 0.0
        self.trade_count        = 0
        self.win_count          = 0
        self.total_trades_list  : list[TradeRecord] = []
        self.signals_log        : deque = deque(maxlen=50)
        self.scan_count         = 0
        self.start_time         = datetime.now()
        self.last_signals       : dict[str, Signal] = {}
        self.active_ticker      : Optional[str] = None
        self.active_entry_price = 0.0
        self.active_entry_time  = None
        self.trailing_high      = 0.0
        self.trailing_low       = float('inf')
        self.status_msg         = "⚡ Initializing..."

    def record_trade(self, record: TradeRecord):
        with self.lock:
            self.trade_count     += 1
            self.realized_pnl    += record.pnl
            self.total_trades_list.append(record)
            if record.pnl > 0:
                self.win_count += 1
            log.info(
                f"TRADE CLOSED | {record.ticker} {record.side} x{record.qty} | "
                f"Entry: ${record.entry_price:.2f} Exit: ${record.exit_price:.2f} | "
                f"PnL: ${record.pnl:.2f} | Reason: {record.exit_reason}"
            )

    @property
    def win_rate(self):
        return (self.win_count / max(1, self.trade_count)) * 100

    @property
    def elapsed(self):
        return str(datetime.now() - self.start_time).split('.')[0]


state = SessionState()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD (Rich Live UI)
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(api) -> Panel:
    try:
        account    = api.get_account()
        equity     = float(account.equity)
        cash       = float(account.cash)
        buying_pwr = float(account.buying_power)
        positions  = api.list_positions()
    except Exception:
        equity = cash = buying_pwr = 0
        positions = []

    now = datetime.now().strftime("%H:%M:%S")
    session_elapsed = state.elapsed

    # ── Header ─────────────────────────────────────────────────────────────
    pnl_color  = "green" if state.realized_pnl >= 0 else "red"
    pnl_symbol = "▲" if state.realized_pnl >= 0 else "▼"

    header = Text()
    header.append("  APEX TRADER  ", style="bold white on #1a1a2e")
    header.append(f"  {now}  ", style="dim")
    header.append(f"  Runtime: {session_elapsed}  ", style="cyan")
    header.append(f"  Scans: {state.scan_count}  ", style="dim")

    # ── Account Table ───────────────────────────────────────────────────────
    acct_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold #00d4ff")
    acct_table.add_column("Metric",   style="dim", width=18)
    acct_table.add_column("Value",    justify="right", width=16)
    acct_table.add_row("Equity",      f"[bold]${equity:,.2f}[/]")
    acct_table.add_row("Cash",        f"${cash:,.2f}")
    acct_table.add_row("Buying Power",f"${buying_pwr:,.2f}")
    acct_table.add_row(
        "Session P&L",
        f"[bold {pnl_color}]{pnl_symbol} ${state.realized_pnl:,.2f}[/]"
    )
    acct_table.add_row("Trades",      f"{state.trade_count}")
    acct_table.add_row("Win Rate",    f"[green]{state.win_rate:.1f}%[/]")

    # ── Positions Table ─────────────────────────────────────────────────────
    pos_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold #00d4ff")
    pos_table.add_column("Symbol", width=7)
    pos_table.add_column("Side",   width=6)
    pos_table.add_column("Qty",    justify="right", width=6)
    pos_table.add_column("Entry",  justify="right", width=8)
    pos_table.add_column("Price",  justify="right", width=8)
    pos_table.add_column("P&L",    justify="right", width=10)

    if positions:
        for p in positions:
            upnl = float(p.unrealized_pl)
            col  = "green" if upnl >= 0 else "red"
            side = "LONG" if int(p.qty) > 0 else "SHORT"
            pos_table.add_row(
                f"[bold]{p.symbol}[/]",
                f"[{'green' if side=='LONG' else 'red'}]{side}[/]",
                str(abs(int(p.qty))),
                f"${float(p.avg_entry_price):,.2f}",
                f"${float(p.current_price):,.2f}",
                f"[{col}]${upnl:,.2f}[/]"
            )
    else:
        pos_table.add_row("[dim]─[/]", "[dim]─[/]", "[dim]─[/]", "[dim]─[/]", "[dim]─[/]", "[dim]FLAT[/]")

    # ── Signal Heatmap ──────────────────────────────────────────────────────
    sig_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold #00d4ff")
    sig_table.add_column("Ticker",   width=7)
    sig_table.add_column("Dir",      width=6)
    sig_table.add_column("Score",    justify="right", width=7)
    sig_table.add_column("RSI",      justify="right", width=7)
    sig_table.add_column("VWAP%",    justify="right", width=8)
    sig_table.add_column("Vol×",     justify="right", width=7)
    sig_table.add_column("Strategy", width=16)

    for ticker in WATCHLIST:
        sig = state.last_signals.get(ticker)
        if sig:
            dir_color = "green" if sig.direction=="LONG" else ("red" if sig.direction=="SHORT" else "dim")
            score_bar = "█" * int(sig.score / 10) + "░" * (10 - int(sig.score / 10))
            score_color = "green" if sig.score >= 75 else ("yellow" if sig.score >= 55 else "dim")
            sig_table.add_row(
                f"[bold]{ticker}[/]",
                f"[{dir_color}]{sig.direction}[/]",
                f"[{score_color}]{sig.score:.0f}[/]",
                f"{sig.rsi:.1f}",
                f"{sig.vwap_dev:+.2f}%",
                f"{sig.volume_ratio:.2f}×",
                f"[dim]{sig.strategy}[/]"
            )
        else:
            sig_table.add_row(ticker, "…", "…", "…", "…", "…", "[dim]scanning[/]")

    # ── Status bar ──────────────────────────────────────────────────────────
    status = Panel(
        Text(state.status_msg, style="bold yellow"),
        box=box.SIMPLE,
        style="on #0d0d1a"
    )

    # ── Assemble layout ─────────────────────────────────────────────────────
    top_cols = Columns([
        Panel(acct_table,  title="[bold #00d4ff]ACCOUNT[/]",   border_style="#1e3a5f", expand=True),
        Panel(pos_table,   title="[bold #00d4ff]POSITIONS[/]", border_style="#1e3a5f", expand=True),
    ])

    from rich.console import Group
    content = Group(
        header,
        top_cols,
        Panel(sig_table, title="[bold #00d4ff]SIGNAL MATRIX[/]", border_style="#1e3a5f"),
        status
    )

    return Panel(
        content,
        title="[bold white on #0a0a1a] ⚡ APEX TRADING ENGINE [/]",
        border_style="#00d4ff",
        box=box.HEAVY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER THREAD — Continuously scores all tickers
# ─────────────────────────────────────────────────────────────────────────────

def scanner_thread(api):
    """Runs in background, continuously updating signal scores for all tickers."""
    while True:
        try:
            for ticker in WATCHLIST:
                data = IndicatorEngine.compute(ticker)
                if data is not None:
                    sig = SignalGenerator.score(ticker, data)
                    with state.lock:
                        state.last_signals[ticker] = sig

            with state.lock:
                state.scan_count += 1

        except Exception as e:
            log.error(f"Scanner error: {e}")

        time.sleep(SCAN_INTERVAL_SECS)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION ENGINE — Trade loop
# ─────────────────────────────────────────────────────────────────────────────

def execution_engine(api):
    """Core trade decision loop, runs in main thread."""

    while state.realized_pnl > RISK_CONFIG['max_session_loss']:

        try:
            # ── Guard: Daily trade limit ────────────────────────────────────
            if state.trade_count >= RISK_CONFIG['max_daily_trades']:
                state.status_msg = f"🚫 Daily trade cap ({RISK_CONFIG['max_daily_trades']}) reached. Waiting..."
                time.sleep(30)
                continue

            account   = api.get_account()
            equity    = float(account.equity)
            positions = api.list_positions()

            # ──────────────────────────────────────────────────────────────
            # CASE A: Managing open positions
            # ──────────────────────────────────────────────────────────────
            if positions:
                for p in positions:
                    sym        = p.symbol
                    qty        = abs(int(p.qty))
                    side       = 'LONG' if int(p.qty) > 0 else 'SHORT'
                    entry      = float(p.avg_entry_price)
                    curr_price = float(p.current_price)
                    unrealized = float(p.unrealized_pl)

                    # Fetch current ATR for this position
                    sig = state.last_signals.get(sym)
                    if not sig:
                        continue

                    atr            = sig.atr
                    profit_target  = atr * RISK_CONFIG['atr_profit_multiplier'] * qty
                    stop_loss      = -(atr * RISK_CONFIG['atr_stop_multiplier'] * qty)
                    exit_reason    = None

                    # ── Profit target ──────────────────────────────────────
                    if unrealized >= profit_target:
                        exit_reason = 'PROFIT_TARGET'
                        state.status_msg = f"💎 PROFIT TARGET HIT on {sym} | +${unrealized:.2f}"

                    # ── Hard stop loss ─────────────────────────────────────
                    elif unrealized <= stop_loss:
                        exit_reason = 'STOP_LOSS'
                        state.status_msg = f"🛡️ STOP LOSS on {sym} | ${unrealized:.2f}"

                    # ── Trailing stop ──────────────────────────────────────
                    elif side == 'LONG':
                        if curr_price > state.trailing_high:
                            state.trailing_high = curr_price
                        trail_dist = state.trailing_high * RISK_CONFIG['trailing_stop_pct']
                        if state.trailing_high > 0 and curr_price < state.trailing_high - trail_dist:
                            exit_reason = 'TRAILING_STOP'
                            state.status_msg = f"📉 TRAILING STOP on {sym} (peak was ${state.trailing_high:.2f})"
                    elif side == 'SHORT':
                        if curr_price < state.trailing_low:
                            state.trailing_low = curr_price
                        trail_dist = state.trailing_low * RISK_CONFIG['trailing_stop_pct']
                        if state.trailing_low < float('inf') and curr_price > state.trailing_low + trail_dist:
                            exit_reason = 'TRAILING_STOP'
                            state.status_msg = f"📈 TRAILING STOP on {sym} (bottom was ${state.trailing_low:.2f})"

                    # ── Signal flip: close if signal now opposes position ──
                    elif sig.direction == 'SHORT' and side == 'LONG' and sig.score >= 75:
                        exit_reason = 'SIGNAL_FLIP'
                        state.status_msg = f"🔄 SIGNAL FLIP on {sym} — exiting LONG"
                    elif sig.direction == 'LONG' and side == 'SHORT' and sig.score >= 75:
                        exit_reason = 'SIGNAL_FLIP'
                        state.status_msg = f"🔄 SIGNAL FLIP on {sym} — exiting SHORT"

                    if exit_reason:
                        api.close_position(sym)
                        duration = (
                            (datetime.now() - state.active_entry_time).total_seconds()
                            if state.active_entry_time else 0
                        )
                        state.record_trade(TradeRecord(
                            ticker=sym, side=side, qty=qty,
                            entry_price=entry, exit_price=curr_price,
                            pnl=unrealized, duration_sec=duration,
                            strategy=sig.strategy, exit_reason=exit_reason
                        ))
                        state.trailing_high = 0.0
                        state.trailing_low  = float('inf')
                        log.info(f"CLOSED {sym} | {exit_reason} | PnL: ${unrealized:.2f}")

            # ──────────────────────────────────────────────────────────────
            # CASE B: Looking for new entries
            # ──────────────────────────────────────────────────────────────
            else:
                if len(positions) >= RISK_CONFIG['max_concurrent_positions']:
                    state.status_msg = "⚠️ Max concurrent positions reached."
                    time.sleep(SCAN_INTERVAL_SECS)
                    continue

                # Pick best signal from watchlist
                best_sig = None
                for ticker in WATCHLIST:
                    sig = state.last_signals.get(ticker)
                    if sig and sig.direction != 'FLAT' and sig.score >= RISK_CONFIG['min_signal_score']:
                        if best_sig is None or sig.score > best_sig.score:
                            best_sig = sig

                if best_sig:
                    qty  = PositionSizer.calc_qty(equity, best_sig.price, best_sig.atr, best_sig.score)
                    side = 'buy' if best_sig.direction == 'LONG' else 'sell'

                    emoji = "🟢" if best_sig.direction == 'LONG' else "🔴"
                    state.status_msg = (
                        f"{emoji} ENTERING {best_sig.direction} {best_sig.ticker} "
                        f"| Score: {best_sig.score:.1f} | Qty: {qty} | Strategy: {best_sig.strategy}"
                    )

                    api.submit_order(
                        symbol=best_sig.ticker,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='gtc'
                    )

                    with state.lock:
                        state.active_ticker      = best_sig.ticker
                        state.active_entry_price = best_sig.price
                        state.active_entry_time  = datetime.now()

                    log.info(
                        f"ENTERED {best_sig.direction} {best_sig.ticker} x{qty} "
                        f"@ ${best_sig.price:.2f} | Score: {best_sig.score:.1f} | "
                        f"Strategy: {best_sig.strategy}"
                    )
                else:
                    best_score = max(
                        (s.score for s in state.last_signals.values()), default=0
                    )
                    state.status_msg = (
                        f"🔍 Scanning {len(WATCHLIST)} tickers... "
                        f"Best score: {best_score:.1f} (need {RISK_CONFIG['min_signal_score']})"
                    )

        except Exception as e:
            state.status_msg = f"⚠️ Execution error: {e}"
            log.error(f"Execution loop error: {e}")

        time.sleep(SCAN_INTERVAL_SECS)

    # Session end
    state.status_msg = f"🚨 SESSION TERMINATED — P&L: ${state.realized_pnl:,.2f}"
    log.info(f"SESSION END | P&L: ${state.realized_pnl:.2f} | Trades: {state.trade_count} | Win Rate: {state.win_rate:.1f}%")

    # Save stats
    stats = {
        'date':        datetime.now().isoformat(),
        'realized_pnl': state.realized_pnl,
        'trade_count':  state.trade_count,
        'win_rate':     state.win_rate,
        'runtime':      state.elapsed,
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

    console.print(Panel.fit(
        "[bold #00d4ff]APEX TRADER[/] [white]initializing...[/]\n"
        f"[dim]Watchlist: {', '.join(WATCHLIST)}[/]\n"
        f"[dim]Strategies: RSI·BB·VWAP·MACD·Volume[/]\n"
        f"[dim]Min Signal Score: {RISK_CONFIG['min_signal_score']}[/]",
        border_style="#00d4ff"
    ))
    time.sleep(1)

    # Warm up scanner (pre-populate signals before trading begins)
    console.print("[yellow]⚡ Warming up signal engine...[/]")
    for ticker in WATCHLIST:
        data = IndicatorEngine.compute(ticker)
        if data is not None:
            sig = SignalGenerator.score(ticker, data)
            state.last_signals[ticker] = sig
            console.print(f"  [dim]{ticker}[/] — {sig.direction} | Score: {sig.score:.1f}")

    # Launch background scanner thread
    t = threading.Thread(target=scanner_thread, args=(api,), daemon=True)
    t.start()

    # Launch execution engine in background
    exec_thread = threading.Thread(target=execution_engine, args=(api,), daemon=True)
    exec_thread.start()

    # Live dashboard in main thread
    with Live(build_dashboard(api), refresh_per_second=1, screen=True) as live:
        while exec_thread.is_alive():
            live.update(build_dashboard(api))
            time.sleep(1)

    console.print(
        Panel(
            f"[bold red]SESSION CLOSED[/]\n"
            f"Realized P&L: [bold]${state.realized_pnl:,.2f}[/]\n"
            f"Trades: {state.trade_count}  |  Win Rate: {state.win_rate:.1f}%\n"
            f"Runtime: {state.elapsed}",
            title="APEX TRADER — FINAL REPORT",
            border_style="red"
        )
    )


if __name__ == '__main__':
    main()

"""Metals Quant Engine V1 — research/backtest only. No live order placement.
Input CSV columns: time,open,high,low,close[,spread] (UTC timestamps).
Spread is optional and expressed in price units; if absent, CLI --spread is used.
Assumptions: signals at bar close, entries at NEXT bar open, conservative intrabar
stop/target handling (stop first), fixed risk in USD, one position per instrument.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    equity: float = 50_000.0
    risk_usd: float = 100.0
    hourly_loss_usd: float = 200.0
    daily_loss_usd: float = 500.0
    hourly_profit_goal: float = 250.0
    max_open_risk_usd: float = 200.0
    atr_period: int = 14
    z_window: int = 30
    channel_window: int = 20
    stop_atr: float = 1.5
    reward_risk: float = 1.8
    max_bars_held: int = 24
    commission_per_lot_roundtrip: float = 7.0
    slippage_price: float = 0.02

# Contract size is units of underlying per lot; broker specs MUST be checked.
# Defaults here are illustrative ONLY, not broker-specific.
@dataclass(frozen=True)
class Contract:
    symbol: str
    contract_size: float
    lot_step: float
    min_lot: float
    max_lot: float

CONTRACTS = {
    'XAUUSD': Contract('XAUUSD', 100.0, 0.01, 0.01, 100.0),
    'XAGUSD': Contract('XAGUSD', 5000.0, 0.01, 0.01, 100.0),
}

def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'time', 'open', 'high', 'low', 'close'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='raise')
    df = df.sort_values('time').drop_duplicates('time').set_index('time')
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df.empty or df[['open','high','low','close']].isna().any().any():
        raise ValueError('Empty or invalid OHLC data')
    if ((df['high'] < df[['open','close','low']].max(axis=1)) |
        (df['low'] > df[['open','close','high']].min(axis=1)) |
        (df['low'] <= 0)).any():
        raise ValueError('Inconsistent OHLC bars')
    if 'spread' in df:
        df['spread'] = pd.to_numeric(df['spread'], errors='raise')
        if (df['spread'] < 0).any(): raise ValueError('Spread cannot be negative')
    return df

def features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x['atr'] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
    x['returns'] = np.log(x.close / prev)
    x['momentum_z'] = (x['returns'].rolling(cfg.z_window).sum() /
        (x['returns'].rolling(cfg.z_window).std() * np.sqrt(cfg.z_window)).replace(0, np.nan))
    mean = x.close.rolling(cfg.z_window).mean()
    std = x.close.rolling(cfg.z_window).std()
    x['price_z'] = (x.close - mean) / std.replace(0, np.nan)
    x['channel_high'] = x.high.shift(1).rolling(cfg.channel_window).max()
    x['channel_low'] = x.low.shift(1).rolling(cfg.channel_window).min()
    x['vol_ratio'] = (x.atr / x.close) / (x.atr / x.close).rolling(60).median().replace(0,np.nan)
    return x

def signal(row: pd.Series) -> tuple[int, str]:
    if not np.isfinite(row[['atr','momentum_z','price_z','channel_high','channel_low','vol_ratio']].to_numpy(dtype=float)).all():
        return 0, 'warmup'
    if row.vol_ratio >= 1.25:
        if row.close > row.channel_high + 0.1 * row.atr: return 1, 'breakout'
        if row.close < row.channel_low - 0.1 * row.atr: return -1, 'breakout'
        return 0, 'high_vol_wait'
    if row.vol_ratio < 0.85:
        if row.price_z <= -2.0: return 1, 'mean_reversion'
        if row.price_z >= 2.0: return -1, 'mean_reversion'
        return 0, 'range_wait'
    if row.momentum_z >= 1.5: return 1, 'momentum'
    if row.momentum_z <= -1.5: return -1, 'momentum'
    return 0, 'no_signal'

def size_lots(stop_distance: float, contract: Contract, cfg: Config) -> float:
    if stop_distance <= 0: return 0.0
    raw = cfg.risk_usd / (stop_distance * contract.contract_size)
    steps = np.floor((raw + 1e-12) / contract.lot_step)
    lots = min(float(steps * contract.lot_step), contract.max_lot)
    return round(lots, 8) if lots >= contract.min_lot else 0.0

def backtest(bars: pd.DataFrame, symbol: str, cfg: Config, default_spread: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    if symbol not in CONTRACTS: raise ValueError('Unsupported symbol')
    if default_spread < 0 or cfg.slippage_price < 0: raise ValueError('Costs must be nonnegative')
    contract = CONTRACTS[symbol]
    x = features(bars, cfg)
    equity = cfg.equity
    trades = []
    curve = []
    position = None
    hourly_pnl: dict[pd.Timestamp,float] = {}
    daily_pnl: dict[pd.Timestamp,float] = {}
    pending = None
    # At each bar open, act on the previous bar's close signal; no lookahead.
    for i in range(1,len(x)):
        t = x.index[i]
        bar = x.iloc[i]
        hour = t.floor('h')
        day = t.floor('D')
        hp = hourly_pnl.get(hour,0.0)
        dp = daily_pnl.get(day,0.0)
        spread = float(bar['spread']) if 'spread' in x and pd.notna(bar['spread']) else default_spread
        if position is None and pending is not None:
            side, strategy, atr = pending
            # Do not open if risk limits already reached. Risk is tracked on close.
            if hp > -cfg.hourly_loss_usd and dp > -cfg.daily_loss_usd and hp < cfg.hourly_profit_goal:
                dist = cfg.stop_atr * atr
                lots = size_lots(dist,contract,cfg)
                # Remaining loss budget, incl. estimated transaction costs, must fit.
                cost = lots*cfg.commission_per_lot_roundtrip
                estimated_risk = lots*contract.contract_size*(dist+spread+2*cfg.slippage_price)+cost
                remaining = min(cfg.hourly_loss_usd+hp, cfg.daily_loss_usd+dp, cfg.max_open_risk_usd)
                if lots > 0 and estimated_risk <= remaining:
                    entry = float(bar.open) + side*(spread/2+cfg.slippage_price)
                    position = {'entry_time':t,'entry':entry,'side':side,'lots':lots,
                                'stop':entry-side*dist,'target':entry+side*dist*cfg.reward_risk,
                                'strategy':strategy,'entry_i':i,'entry_day':day,'entry_hour':hour}
        pending = None
        if position is not None:
            p = position
            # Conservative: if stop and target both touched, stop wins.
            # Exit at next available bar price on a gap through stop/target.
            stop_hit = bar.low <= p['stop'] if p['side']==1 else bar.high >= p['stop']
            target_hit = bar.high >= p['target'] if p['side']==1 else bar.low <= p['target']
            expired = i-p['entry_i'] >= cfg.max_bars_held
            if stop_hit or target_hit or expired:
                if stop_hit:
                    raw_exit = min(float(bar.open),p['stop']) if p['side']==1 else max(float(bar.open),p['stop'])
                    reason = 'stop'
                elif target_hit:
                    raw_exit = p['target']  # do not assume favorable gap improvement
                    reason = 'target'
                else:
                    raw_exit = float(bar.close)
                    reason = 'timeout'
                exit_price = raw_exit-p['side']*(spread/2+cfg.slippage_price)
                gross = p['side']*(exit_price-p['entry'])*contract.contract_size*p['lots']
                commission = p['lots']*cfg.commission_per_lot_roundtrip
                net = gross-commission
                equity += net
                # Book realized loss to exit hour/day; entry-hour exposure is not continuously marked.
                hourly_pnl[hour] = hourly_pnl.get(hour,0.0)+net
                daily_pnl[day] = daily_pnl.get(day,0.0)+net
                trades.append({**p,'exit_time':t,'exit':exit_price,'reason':reason,
                               'gross_pnl':gross,'commission':commission,'net_pnl':net,'equity':equity})
                position = None
        # Signal on this bar's close only if flat; action occurs next bar open.
        if position is None and i < len(x)-1:
            side, strategy = signal(bar)
            if side and pd.notna(bar.atr) and bar.atr>0:
                pending = (side,strategy,float(bar.atr))
        # Equity curve is realized-only, not mark-to-market.
        curve.append({'time':t,'realized_equity':equity,'open_position':position is not None})
    # Open positions remain unclosed; no forced end-of-data fill.
    return pd.DataFrame(trades),pd.DataFrame(curve)

def summarize(trades: pd.DataFrame, curve: pd.DataFrame, initial: float) -> dict:
    if trades.empty:
        return {'closed_trades':0,'net_pnl':0.0,'win_rate':None,'max_realized_drawdown':0.0,'end_realized_equity':initial}
    e = pd.concat([pd.Series([initial]),curve.realized_equity],ignore_index=True)
    dd = e/e.cummax()-1
    return {'closed_trades':len(trades),'net_pnl':round(float(trades.net_pnl.sum()),2),
            'win_rate':round(float((trades.net_pnl>0).mean()),4),
            'max_realized_drawdown':round(float(dd.min()),4),
            'end_realized_equity':round(float(curve.realized_equity.iloc[-1]),2)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',required=True)
    ap.add_argument('--symbol',required=True,choices=CONTRACTS)
    ap.add_argument('--spread',type=float,required=True,help='Fallback full bid/ask spread in PRICE units')
    ap.add_argument('--contract-size',type=float,help='Broker-specific units per lot (overrides illustrative default)')
    ap.add_argument('--lot-step',type=float,help='Broker-specific lot increment')
    ap.add_argument('--min-lot',type=float,help='Broker-specific minimum lot')
    ap.add_argument('--max-lot',type=float,help='Broker-specific maximum lot')
    ap.add_argument('--slippage',type=float,default=0.02,help='Per-side price units')
    ap.add_argument('--commission',type=float,default=7.0,help='Roundtrip USD per lot')
    ap.add_argument('--output',default='results')
    args = ap.parse_args()
    original = CONTRACTS[args.symbol]
    contract = Contract(args.symbol,args.contract_size or original.contract_size,
                        args.lot_step or original.lot_step,args.min_lot or original.min_lot,
                        args.max_lot or original.max_lot)
    if min(contract.contract_size,contract.lot_step,contract.min_lot,contract.max_lot)<=0 or contract.min_lot>contract.max_lot:
        ap.error('Invalid contract specification')
    CONTRACTS[args.symbol] = contract
    cfg = Config(slippage_price=args.slippage,commission_per_lot_roundtrip=args.commission)
    trades,curve = backtest(load_bars(args.csv),args.symbol,cfg,args.spread)
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    trades.to_csv(out/'trades.csv',index=False)
    curve.to_csv(out/'equity_curve.csv',index=False)
    print(summarize(trades,curve,cfg.equity))
    print('Research only. No MT5 orders placed.')

if __name__=='__main__': main()

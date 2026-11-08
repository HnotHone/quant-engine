# Metals Quant Engine V1

**Research/backtest only. No broker orders are placed.** Python 3.10+.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Export MT5 OHLC bars to a CSV with `time,open,high,low,close` and optionally `spread` (full bid-ask spread in price units). Use UTC timestamps. M1, M5 or M15 bars can be supplied. Do not mix intervals. Each instrument should be run separately.

## Run

```bash
python quant_engine.py --csv your_XAUUSD_M5.csv --symbol XAUUSD --spread 0.30 --output results_gold
python quant_engine.py --csv your_XAGUSD_M5.csv --symbol XAGUSD --spread 0.04 --output results_silver
```

**Spread values above are illustrative, not actual broker quotes.** Specify broker-specific contract size, lot step, minimum/maximum lots, commission, and slippage using the CLI options. Verify all against MT5 Symbol Specification. Defaults for XAUUSD and XAGUSD contract size are illustrative only.

## Logic

- Computes ATR, normalized momentum, price z-score, prior channel boundaries and relative volatility.
- Chooses breakout under high volatility, mean reversion under low volatility, and momentum otherwise.
- Signals are evaluated on bar close; entry is on the next bar open.
- Position size targets $100 initial stop-distance risk, but total estimated risk includes modeled costs.
- No new positions when realized hourly P&L reaches +$250, -$200, or daily P&L reaches -$500.
- One open position per backtest/instrument. Combined XAU/XAG portfolio risk limits are **not implemented**.

## Critical limitations (do not use for live trading)

- No MT5 connectivity or order execution yet; this is a portable research module compatible with macOS.
- Intrabar bid/ask behavior is approximated from OHLC and fixed/CSV spread; tick-accurate fills are not modeled.
- Stop-loss gaps can exceed intended risk. Hourly/daily loss limits are **entry gates**, not guarantees.
- Realized-equity curve excludes open unrealized P&L; risk limits do not include floating losses.
- Overnight session boundaries, financing/swap, exchange holidays, stale quotes, spread spikes and margin requirements are not modeled.
- Parameters are unvalidated hypotheses. No walk-forward test, statistical significance testing or portfolio aggregation yet.
- Backtest uses static nominal USD risk; equity changes do not automatically alter risk budget.
- Closing a trade at a bar and generating a new signal on that same bar is permitted, but any entry is delayed until next bar.

Next steps: MT5 data export connector, broker metadata validation, portfolio-level risk engine, mark-to-market limits, walk-forward validation, and demo-only execution bridge.

# Nano-L1

High-performance cryptocurrency trading engine and sandbox for strategy development.

![Go](https://img.shields.io/badge/Go-1.23-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Kafka](https://img.shields.io/badge/Kafka-7.4-231F20?logo=apachekafka&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-16-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

## Architecture

The **engine** (Go) is the core — it manages the order book, executes trades, and persists everything to Postgres. It exposes a REST API for order submission and a WebSocket for real-time book/trade updates.

**Feed-sim** replays historical tick data from CSV files into Kafka. The engine consumes these ticks for backtesting scenarios.

**Dashboard** (React) connects to the engine via WebSocket and displays live depth, trades, and P&L.

**Agent** (Python) polls the engine's REST API, runs a trading policy (mean-reversion or ML), and submits orders back.

## Services

| Service | Language | Description |
|---------|----------|-------------|
| `engine-go` | Go | Order matching engine with WebSocket streaming and REST API |
| `dashboard-react` | React | Real-time trading UI with depth chart, trade tape, P&L |
| `backtest-py` | Python | Strategy backtester supporting CSV and synthetic tick generation |
| `agent-py` | Python | Automated trading agent with mean-reversion and ML policies |
| `feed-sim` | Go | Market data replay simulator publishing ticks to Kafka |

## Quick Start

```bash
cd infra/docker
docker-compose up
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5173 |
| Engine API | http://localhost:8080 |
| WebSocket | ws://localhost:8080/ws |

## Backtester Performance

```bash
cd services/backtest-py
python backtest.py --synthetic 2000000
```

```
[backtest] synthetic mode: n=2000000, symbol=TEST, seed=42

[backtest] done
  ticks processed : 2,000,000
  elapsed time    : 1.23 s
  ticks / second  : 1,626,016 / s

  final position  : -1234.00
  last price      : 98.45
  cash PnL        : 123456.78
  MTM PnL         : -121234.56
```

## Trading Agent

Two policies, same interface:

**Mean Reversion** — buys dips, sells rips (5bp threshold), 25% random exploration.

**ML Policy** — LogisticRegression on Binance data, uses price move + volatility features.

```bash
# Mean reversion
cd services/agent-py
python agent.py --engine-url http://localhost:8080

# ML policy (train first)
python train_binance_trades_model.py --csv ../../data/binance/btc_usdt_trades.csv --out model.pkl
python agent.py --engine-url http://localhost:8080 --model-path model.pkl
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/order` | POST | Submit order `{symbol, side, type, price, qty}` |
| `/api/trades/recent?limit=N` | GET | Last N executed trades |
| `/pnl?symbol=X` | GET | Position and P&L for symbol |
| `/ws` | WebSocket | Real-time `book_update` and `trade` events |

## Kafka Topics

| Topic | Producer | Schema |
|-------|----------|--------|
| `ticks` | feed-sim | `{ts, symbol, price, side, qty}` |
| `book_updates` | engine | `{symbol, bestBid, bestAsk, bidQty, askQty}` |
| `trades` | engine | `{id, symbol, price, qty, aggressorSide, ts}` |

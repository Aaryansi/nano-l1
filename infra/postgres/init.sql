@'
CREATE TABLE IF NOT EXISTS trades (
  id SERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  symbol TEXT NOT NULL,
  price DOUBLE PRECISION NOT NULL,
  qty DOUBLE PRECISION NOT NULL,
  aggressor_side TEXT NOT NULL,
  maker_order_id TEXT,
  taker_order_id TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  started_at TIMESTAMP DEFAULT NOW()
);

-- Optional for later: store normalized ticks
CREATE TABLE IF NOT EXISTS ticks (
  id SERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  symbol TEXT NOT NULL,
  price DOUBLE PRECISION NOT NULL,
  qty DOUBLE PRECISION NOT NULL,
  side TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts);
'@ | Set-Content infra/postgres/init.sql

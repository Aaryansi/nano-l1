import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8080";

function buildSeries(trades) {
  if (!trades || trades.length === 0) return { points: [], last: null, changePct: null };

  // sort oldest -> newest
  const sorted = [...trades].sort((a, b) => a.id - b.id);

  const firstPrice = sorted[0].price;
  const lastPrice = sorted[sorted.length - 1].price;

  const points = sorted.map((t, idx) => ({
    x: idx,
    price: t.price,
  }));

  const changePct =
    firstPrice > 0 ? ((lastPrice - firstPrice) / firstPrice) * 100 : 0;

  return { points, last: lastPrice, changePct };
}

export default function PriceHistory() {
  const [points, setPoints] = useState([]);
  const [lastPrice, setLastPrice] = useState(null);
  const [changePct, setChangePct] = useState(null);
  const [error, setError] = useState(null);

  async function fetchPrices() {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/api/trades/recent?limit=200`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const json = await res.json();
      const trades = Array.isArray(json) ? json : json.trades || [];

      const { points, last, changePct } = buildSeries(trades);
      setPoints(points);
      setLastPrice(last);
      setChangePct(changePct);
    } catch (err) {
      console.error("[PriceHistory] fetch error", err);
      setError("Failed to load history");
    }
  }

  useEffect(() => {
    // initial fetch
    fetchPrices();

    // refresh every 2 seconds
    const id = setInterval(fetchPrices, 2000);
    return () => clearInterval(id);
  }, []);

  const minPrice =
    points.length > 0
      ? Math.min(...points.map((p) => p.price))
      : 100.0;
  const maxPrice =
    points.length > 0
      ? Math.max(...points.map((p) => p.price))
      : 100.2;

  const padding = (maxPrice - minPrice || 0.02) * 0.3;
  const lo = minPrice - padding;
  const hi = maxPrice + padding;

  const width = 600;
  const height = 140;
  const leftPad = 20;
  const rightPad = 10;

  const pathD =
    points.length === 0
      ? ""
      : points
          .map((pt, idx) => {
            const x =
              leftPad +
              ((width - leftPad - rightPad) * idx) /
                Math.max(points.length - 1, 1);
            const norm =
              hi === lo ? 0.5 : (pt.price - lo) / (hi - lo);
            const y = height - norm * (height - 20) - 10;
            return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
          })
          .join(" ");

  const pctDisplay =
    changePct === null ? "--" : `${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%`;

  return (
    <section className="card card--wide">
      <div className="card-header">
        <div>
          <h2 className="card-title">Price History</h2>
          <p className="card-subtitle">Last 200 trades (from DB)</p>
        </div>
        <div className="price-history-meta">
          <div className="price-history-last">
            {lastPrice !== null ? lastPrice.toFixed(2) : "--"}
          </div>
          <div
            className={
              changePct > 0
                ? "price-history-change price-history-change--up"
                : changePct < 0
                ? "price-history-change price-history-change--down"
                : "price-history-change"
            }
          >
            {pctDisplay}
          </div>
        </div>
      </div>

      <div className="price-history-chart">
        {error && (
          <div className="card-error">
            {error}
          </div>
        )}
        {!error && points.length === 0 && (
          <div className="card-empty">
            No trades yet – send some orders.
          </div>
        )}
        {!error && points.length > 0 && (
          <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
            <path
              d={pathD}
              fill="none"
              className="price-history-path"
            />
            {/* simple baseline */}
            <line
              x1={leftPad}
              y1={height - 20}
              x2={width - rightPad}
              y2={height - 20}
              className="price-history-baseline"
            />
          </svg>
        )}
      </div>
    </section>
  );
}

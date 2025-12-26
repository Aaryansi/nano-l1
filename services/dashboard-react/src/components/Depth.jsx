import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8080";

export default function Depth() {
  const [book, setBook] = useState(null);
  const [error, setError] = useState(null);

  // Poll the REST endpoint for best bid/ask + last trade
  async function fetchDepth() {
    try {
      const res = await fetch(`${API_BASE}/api/trades/recent?limit=1`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const body = await res.json();
      const trades = Array.isArray(body) ? body : body.trades || [];
      const last = trades[0];

      if (!last) {
        setBook(null);
        return;
      }

      // For now, we approximate top of book from last trade.
      // Later you can wire a dedicated depth endpoint / WS feed.
      setBook({
        bestBid: { price: last.price, qty: last.qty },
        bestAsk: { price: last.price, qty: last.qty },
        lastTradePrice: last.price,
      });
      setError(null);
    } catch (err) {
      console.error("[Depth] fetch error", err);
      setError(err.message);
    }
  }

  useEffect(() => {
    fetchDepth(); // first load
    const id = setInterval(fetchDepth, 3000); // refresh every 3s
    return () => clearInterval(id);
  }, []);

  const bestBid = book?.bestBid || { price: 0, qty: 0 };
  const bestAsk = book?.bestAsk || { price: 0, qty: 0 };
  const lastTradePrice =
    book?.lastTradePrice !== undefined ? book.lastTradePrice : null;

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">Top of Book</div>
        <div className="panel-subtitle">
          Approximated from latest trade (engine_trades)
        </div>
      </div>
      <div className="panel-body">
        {error && (
          <div className="muted-small" style={{ color: "#ff5b7b" }}>
            Failed to load depth: {error}
          </div>
        )}

        {!book && !error && (
          <div className="muted">No trades yet – send an order to see data.</div>
        )}

        {book && (
          <div className="top-of-book-grid">
            <div>
              <div className="metric-label">Best Bid</div>
              <div className="metric-value green">
                {bestBid.price.toFixed(2)}
                <span className="metric-qty">× {bestBid.qty.toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="metric-label">Best Ask</div>
              <div className="metric-value red">
                {bestAsk.price.toFixed(2)}
                <span className="metric-qty">× {bestAsk.qty.toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="metric-label">Last Trade</div>
              <div className="metric-value">
                {lastTradePrice != null ? lastTradePrice.toFixed(2) : "--"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

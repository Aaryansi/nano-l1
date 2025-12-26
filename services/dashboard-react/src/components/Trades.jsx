import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_ENGINE_HTTP || "http://localhost:8080";

export default function Trades() {
  const [trades, setTrades] = useState([]);
  const [error, setError] = useState(null);

  async function fetchTrades() {
    try {
      const res = await fetch(`${API_BASE}/api/trades/recent?limit=20`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const body = await res.json();
      const rows = Array.isArray(body) ? body : body.trades || [];
      setTrades(rows);
      setError(null);
    } catch (err) {
      console.error("[Trades] fetch error", err);
      setError(err.message);
    }
  }

  useEffect(() => {
    fetchTrades();
    const id = setInterval(fetchTrades, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="panel panel--trades">
      <div className="panel-header">
        <div className="panel-title">Trades (latest)</div>
        <div className="panel-subtitle">
          Most recent executions from <code>engine_trades</code> (Postgres)
        </div>
      </div>

      <div className="panel-body panel-body--scroll">
        {error && (
          <div className="muted-small" style={{ color: "#ff5b7b" }}>
            Failed to load trades: {error}
          </div>
        )}

        {trades.length === 0 && !error && (
          <div className="muted">
            No trades yet – send an order to see fills.
          </div>
        )}

        {trades.length > 0 && (
          <div className="table-scroll">
            <table className="table-compact">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Side</th>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>Qty</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t) => (
                  <tr key={t.id}>
                    <td>{t.id}</td>
                    <td
                      className={
                        t.aggressor_side === "buy" ? "green" : "red"
                      }
                    >
                      {t.aggressor_side?.toUpperCase()}
                    </td>
                    <td>{t.symbol}</td>
                    <td>
                      {typeof t.price === "number"
                        ? t.price.toFixed(2)
                        : t.price}
                    </td>
                    <td>{t.qty}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

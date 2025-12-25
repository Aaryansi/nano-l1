import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

export default function PersistedTrades() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const fetchTrades = async () => {
    setLoading(true);
    setErr("");
    try {
      const res = await fetch(`${API_BASE}/api/trades/recent?limit=20`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setRows(data);
    } catch (e) {
      console.error("fetch trades error", e);
      setErr("Failed to load DB trades");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  return (
    <div className="card">
      <div className="card-header-row">
        <h2>Persisted Trades (DB)</h2>
        <button className="btn btn-small" onClick={fetchTrades} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {err && <div className="error">{err}</div>}
      {!err && rows.length === 0 && !loading && (
        <div className="muted">No trades in DB yet.</div>
      )}

      {rows.length > 0 && (
        <div className="table-wrap">
          <table className="trades-table">
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
              {rows.map((t) => (
                <tr key={t.id}>
                  <td>{t.id}</td>
                  <td className={t.aggressorSide === "buy" ? "side-buy" : "side-sell"}>
                    {t.aggressorSide.toUpperCase()}
                  </td>
                  <td>{t.symbol}</td>
                  <td>{t.price.toFixed(2)}</td>
                  <td>{t.qty.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

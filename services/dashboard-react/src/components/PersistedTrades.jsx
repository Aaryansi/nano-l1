import React, { useEffect, useState } from "react";

const ENGINE_HTTP =
  import.meta.env.VITE_ENGINE_HTTP || "http://localhost:8080";

export default function PersistedTrades() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchTrades = async () => {
    setLoading(true);
    setError("");

    try {
      const resp = await fetch(
        `${ENGINE_HTTP}/api/trades/recent?limit=200` // bigger window for DB view
      );

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const data = await resp.json();

      // API shape is { trades: [...] }
      let trades = Array.isArray(data.trades) ? data.trades : null;

      // Fallback if backend ever returns a bare array
      if (!trades && Array.isArray(data)) {
        trades = data;
      }

      if (!trades) trades = [];

      setRows(trades);
    } catch (err) {
      console.error("fetch trades error", err);
      setRows([]);
      setError(err.message || "Failed to load DB trades");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
  <div className="card card--persisted">
    <div className="card__header">
      <div className="card__title">Persisted Trades (DB)</div>
      <button
        className="btn btn--secondary"
        onClick={fetchTrades}
        disabled={loading}
      >
        {loading ? "Loading…" : "Refresh"}
      </button>
    </div>

    {error && (
      <div className="card__error">Failed to load DB trades: {error}</div>
    )}

    {!error && !loading && rows.length === 0 && (
      <div className="card__empty">
        No trades stored yet. Send some orders and refresh.
      </div>
    )}

    {!error && rows.length > 0 && (
      <div className="panel-body panel-body--scroll">
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
              {rows.map((t) => {
                const side = (
                  t.aggressor_side ||
                  t.aggressorSide ||
                  ""
                ).toUpperCase();
                const isBuy = side === "BUY";
                return (
                  <tr key={t.id}>
                    <td>{t.id}</td>
                    <td className={isBuy ? "green" : "red"}>{side || "-"}</td>
                    <td>{t.symbol}</td>
                    <td>
                      {t.price?.toFixed ? t.price.toFixed(2) : t.price}
                    </td>
                    <td>{t.qty}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    )}
  </div>
);

}

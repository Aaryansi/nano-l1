import React, { useEffect, useState } from "react";

const API_BASE =
  import.meta.env.VITE_ENGINE_URL || "http://localhost:8080";

export default function Stats() {
  const [tradesInDB, setTradesInDB] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    async function fetchStats() {
      try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        if (isMounted) {
          setTradesInDB(data.tradesInDB ?? 0);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setError("Failed to load stats");
        }
      }
    }

    // initial + poll every 5s
    fetchStats();
    const id = setInterval(fetchStats, 5000);

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="card stats">
      <div className="card-title">DB Stats</div>
      {error ? (
        <div className="card-error">{error}</div>
      ) : tradesInDB === null ? (
        <div className="card-body small">Loading…</div>
      ) : (
        <div className="card-body">
          <div className="stat-label">Persisted trades</div>
          <div className="stat-value">{tradesInDB}</div>
          <div className="stat-note small">
            Count from <code>engine_trades</code> (Postgres).
          </div>
        </div>
      )}
    </div>
  );
}

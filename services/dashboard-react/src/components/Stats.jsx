// src/components/Stats.jsx
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
          console.error("[Stats] fetch error", err);
          setError("Failed to load stats");
        }
      }
    }

    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">DB Stats</div>
      </div>

      <div className="panel-body panel-body-tight">
        {error ? (
          <div className="muted-small" style={{ color: "#ff5b7b" }}>
            {error}
          </div>
        ) : tradesInDB === null ? (
          <div className="muted-small">Loading…</div>
        ) : (
          <>
            <div className="metric-label">Persisted trades</div>
            <div className="metric-value" style={{ marginTop: 4 }}>
              {tradesInDB}
            </div>
            <div className="muted-small" style={{ marginTop: 8 }}>
              Counting rows in <code>engine_trades</code> (Postgres).
            </div>
          </>
        )}
      </div>
    </div>
  );
}

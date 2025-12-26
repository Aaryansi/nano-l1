// src/components/PnL.jsx
import React, { useEffect, useState } from "react";

const ENGINE_HTTP_URL =
  import.meta.env.VITE_ENGINE_HTTP_URL || "http://localhost:8080";

export default function PnL({ symbol = "TEST" }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | refreshing | loaded | error
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function fetchPnL() {
      try {
        if (cancelled) return;

        setStatus((s) => (s === "loaded" ? "refreshing" : "loading"));

        const res = await fetch(
          `${ENGINE_HTTP_URL}/pnl?symbol=${encodeURIComponent(symbol)}`
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const json = await res.json();

        if (!cancelled) {
          setData(json);
          setStatus("loaded");
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          console.error("[pnl] fetch error", err);
          setError("Failed to load P&L");
          setStatus("error");
        }
      }
    }

    fetchPnL();
    const id = setInterval(fetchPnL, 2000); // poll every 2s

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [symbol]);

  const pnl = data?.totalPnL ?? 0;
  const unrealized = data?.unrealized ?? 0;
  const cash = data?.cash ?? 0;
  const position = data?.position ?? 0;
  const lastPrice = data?.lastPrice ?? 0;
  const tradeCount = data?.tradeCount ?? 0;

  const pnlClass =
    pnl > 0 ? "pnl-positive" : pnl < 0 ? "pnl-negative" : "pnl-flat";

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">P&amp;L (from DB)</div>
          <div className="panel-subtitle">
            {symbol} ·{" "}
            {status === "loading" && "Loading…"}
            {status === "refreshing" && "Refreshing…"}
            {status === "loaded" && "Live"}
            {status === "error" && "Error"}
          </div>
        </div>
      </div>

      <div className="panel-body panel-body-tight">
        {error ? (
          <div className="pnl-error muted-small">{error}</div>
        ) : (
          <>
            {/* Big headline number */}
            <div className={`pnl-total ${pnlClass}`}>
              <span className="metric-label">Total P&amp;L</span>
              <span className="pnl-total-value">
                {pnl.toFixed(2)}
              </span>
            </div>

            {/* 2×3 metric grid */}
            <div className="pnl-metrics-grid">
              <div className="pnl-metric">
                <div className="metric-label">Position</div>
                <div className="metric-value">
                  {position.toFixed(2)}
                </div>
              </div>
              <div className="pnl-metric">
                <div className="metric-label">Last price</div>
                <div className="metric-value">
                  {lastPrice ? lastPrice.toFixed(2) : "-"}
                </div>
              </div>
              <div className="pnl-metric">
                <div className="metric-label">Cash (realized)</div>
                <div className="metric-value">
                  {cash.toFixed(2)}
                </div>
              </div>
              <div className="pnl-metric">
                <div className="metric-label">Unrealized</div>
                <div className="metric-value">
                  {unrealized.toFixed(2)}
                </div>
              </div>
              <div className="pnl-metric">
                <div className="metric-label">Trades counted</div>
                <div className="metric-value">{tradeCount}</div>
              </div>
            </div>

            <div className="pnl-note muted-small">
              P&amp;L is derived from trades for {symbol} in{" "}
              <code>engine_trades</code> (assumes all trades are yours
              for now).
            </div>
          </>
        )}
      </div>
    </div>
  );
}

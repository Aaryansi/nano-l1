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
    <div className="card card-pnl">
      <div className="card-header">
        <div className="card-title">P&amp;L (from DB)</div>
        <div className="card-subtitle">
          {symbol} ·{" "}
          {status === "loading" && "Loading..."}
          {status === "refreshing" && "Refreshing..."}
          {status === "loaded" && "Live"}
          {status === "error" && "Error"}
        </div>
      </div>

      {error ? (
        <div className="error">{error}</div>
      ) : (
        <div className="pnl-body">
          <div className={`pnl-value ${pnlClass}`}>
            {pnl.toFixed(2)}
          </div>

          <div className="pnl-row">
            <span>Position</span>
            <span>{position.toFixed(2)}</span>
          </div>
          <div className="pnl-row">
            <span>Last Price</span>
            <span>{lastPrice ? lastPrice.toFixed(2) : "-"}</span>
          </div>
          <div className="pnl-row">
            <span>Cash (realized)</span>
            <span>{cash.toFixed(2)}</span>
          </div>
          <div className="pnl-row">
            <span>Unrealized</span>
            <span>{unrealized.toFixed(2)}</span>
          </div>
          <div className="pnl-row">
            <span>Trades counted</span>
            <span>{tradeCount}</span>
          </div>

          <div className="pnl-note">
            Note: P&amp;L is computed from <code>engine_trades</code>  
            (assumes all trades are yours for now).
          </div>
        </div>
      )}
    </div>
  );
}

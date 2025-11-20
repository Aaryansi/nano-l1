import React from "react";

export default function PnL({ pnl, pos }) {
  const last = pnl[pnl.length - 1] ?? 0;

  return (
    <div className="card">
      <h2>P&L (MVP)</h2>
      <div className="big">{last.toFixed(2)}</div>
      <div className="small muted">Position: {pos.toFixed(2)}</div>

      <div className="mini">
        {pnl.slice(-30).map((x, i) => (
          <div key={i}>{x.toFixed(1)}</div>
        ))}
      </div>

      <div className="small muted" style={{ marginTop: 8 }}>
        Note: assumes all trades are ours (placeholder).
      </div>
    </div>
  );
}

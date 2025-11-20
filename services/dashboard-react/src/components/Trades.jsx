import React from "react";

export default function Trades({ trades }) {
  return (
    <div className="card">
      <h2>Trades (latest)</h2>
      <div className="tape">
        {trades.length === 0 ? (
          <div className="muted small">No trades yet</div>
        ) : (
          trades.map((t, i) => (
            <div key={i} className={`trade ${t.aggressorSide}`}>
              <span className="side">{t.aggressorSide.toUpperCase()}</span>
              <span>@ {t.price.toFixed(2)}</span>
              <span>x {t.qty.toFixed(2)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

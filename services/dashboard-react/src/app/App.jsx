import React, { useEffect, useMemo, useState } from "react";
import { connectWS } from "../lib/wsClient.js";
import Depth from "../components/Depth.jsx";
import Trades from "../components/Trades.jsx";
import PnL from "../components/PnL.jsx";

const WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8080/ws";

export default function App() {
  const [book, setBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [pos, setPos] = useState(0);
  const [pnl, setPnl] = useState([0]);

  useEffect(() => {
    const ws = connectWS(WS_URL, (msg) => {
      if (msg.eventType === "book_update") {
        setBook(msg.data);
      }

      if (msg.eventType === "trades") {
        const newTrades = msg.data || [];
        setTrades((prev) => [...prev, ...newTrades].slice(-200));

        // naive PnL for MVP: assume all trades are ours
        if (newTrades.length) {
          let p = pnl[pnl.length - 1];
          let position = pos;

          for (const tr of newTrades) {
            if (tr.aggressorSide === "buy") {
              position += tr.qty;
              p -= tr.price * tr.qty;
            } else {
              position -= tr.qty;
              p += tr.price * tr.qty;
            }
          }
          setPos(position);
          setPnl((prev) => [...prev, p]);
        }
      }
    });

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [WS_URL]);

  const lastTrades = useMemo(() => trades.slice(-30).reverse(), [trades]);

  return (
    <div className="wrap">
      <header className="header">
        <h1>Nano-L1 Trading Sandbox</h1>
        <div className="sub">WS: {WS_URL}</div>
      </header>

      {!book ? (
        <div className="loading">Waiting for stream…</div>
      ) : (
        <div className="grid">
          <Depth book={book} />
          <Trades trades={lastTrades} />
          <PnL pnl={pnl} pos={pos} />
        </div>
      )}
    </div>
  );
}

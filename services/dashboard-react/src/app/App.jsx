import React, { useEffect, useMemo, useState } from "react";
import { connectWS } from "../lib/wsClient.js";
import Depth from "../components/Depth.jsx";
import Trades from "../components/Trades.jsx";
import PnL from "../components/PnL.jsx";
import OrderPanel from "../components/OrderPanel";
import PersistedTrades from "../components/PersistedTrades.jsx";

import Stats from "../components/Stats.jsx";


const WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8080/ws";

export default function App() {
  const [book, setBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [pos, setPos] = useState(0);
  const [pnl, setPnl] = useState([0]);
  const [wsStatus, setWsStatus] = useState("connecting");

  useEffect(() => {
    const handleMsg = (msg) => {
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
    };

    const wsHandle = connectWS(WS_URL, handleMsg, setWsStatus);

    return () => {
      wsHandle.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [WS_URL]);

  const lastTrades = useMemo(() => trades.slice(-30).reverse(), [trades]);

  return (
    <div className="wrap">
      <header className="header">
        <h1>Nano-L1 Trading Sandbox</h1>

        <OrderPanel />

        <div className={`sub ws-status ws-status-${wsStatus}`}>
          WS: {WS_URL} &nbsp; <span>({wsStatus})</span>
        </div>


        <div className="sub">WS: {WS_URL}</div>
      </header>

      {!book ? (
        <div className="loading">Waiting for stream…</div>
      ) : (
        <div className="grid">
          <Depth book={book} />
          <Trades trades={lastTrades} />
          <PnL symbol={book?.symbol || "TEST"} />
          <PersistedTrades />
          <Stats />
        </div>
      )}
    </div>
  );
}

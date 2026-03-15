import React, { useState } from "react";

const API_BASE =
  import.meta.env.VITE_ENGINE_URL || "http://localhost:8080";

export default function OrderPanel() {
  const [symbol, setSymbol] = useState("TEST");
  const [side, setSide] = useState("buy");
  const [type, setType] = useState("limit");
  const [price, setPrice] = useState("100.10");
  const [qty, setQty] = useState("5");

  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  // const [lastResp, setLastResp] = useState(null);

  const makeOrderId = () => `o-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;

  async function postOrder(payload) {
    setSending(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/order`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      if (!res.ok) {
        let msg = text || `HTTP ${res.status}`;
        try {
          const j = JSON.parse(text);
          if (j.error) msg = j.error;
        } catch (_) {}
        throw new Error(msg);
      }

      if (text) {
        try {
          const data = JSON.parse(text);
          console.log("[order] success:", data);
        } catch {
          console.log("[order] success (non-JSON):", text);
        }
      } else {
        console.log("[order] success: empty body");
      }
    } catch (err) {
      console.error("send order failed:", err);
      setError(err.message || "Failed to send order");
    } finally {
      setSending(false);
    }
  }

  async function handleSendSingle(e) {
    e.preventDefault();

    if (!symbol.trim()) {
      setError("Symbol is required");
      return;
    }
    const q = Number(qty);
    if (!Number.isFinite(q) || q <= 0) {
      setError("Quantity must be > 0");
      return;
    }
    const p = Number(price);
    if (type === "limit" && (!Number.isFinite(p) || p <= 0)) {
      setError("Price must be > 0 for limit orders");
      return;
    }

    const payload = {
      id: makeOrderId(),
      symbol: symbol.trim(),
      side: side.toLowerCase(),         // buy|sell
      type: type.toLowerCase(),         // limit|market|cancel
      price: type === "limit" ? p : 0,  // engine will ignore price for market
      qty: q,
    };

    await postOrder(payload);
  }

  async function handleSampleCrossing() {
    const q = Number(qty) || 5;
    const p = Number(price) || 100.1;

    const buyPayload = {
      id: makeOrderId(),
      symbol: symbol.trim() || "TEST",
      side: "buy",
      type: "limit",
      price: p,
      qty: q,
    };

    const sellPayload = {
      id: makeOrderId(),
      symbol: symbol.trim() || "TEST",
      side: "sell",
      type: "limit",
      price: p - 0.05, // cross the book
      qty: q,
    };

    await postOrder(buyPayload);
    await postOrder(sellPayload);
  }

  return (
    <section className="order-panel">
      <h2 className="section-title">Manual Order Entry</h2>

      <form className="order-form" onSubmit={handleSendSingle}>
        <div className="field-row">
          <div className="field">
            <label className="label">Symbol</label>
            <input
              className="input"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              disabled={sending}
            />
          </div>

          <div className="field">
            <label className="label">Side</label>
            <select
              className="select"
              value={side}
              onChange={(e) => setSide(e.target.value)}
              disabled={sending}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
        </div>

        <div className="field-row">
          <div className="field">
            <label className="label">Type</label>
            <select
              className="select"
              value={type}
              onChange={(e) => setType(e.target.value)}
              disabled={sending}
            >
              <option value="limit">Limit</option>
              <option value="market">Market</option>
            </select>
          </div>

          <div className="field">
            <label className="label">Price</label>
            <input
              className="input"
              type="number"
              step="0.01"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              disabled={sending || type === "market"}
            />
          </div>
        </div>

        <div className="field-row">
          <div className="field">
            <label className="label">Quantity</label>
            <input
              className="input"
              type="number"
              step="1"
              value={qty}
              onChange={(e) => setQty(e.target.value)}
              disabled={sending}
            />
          </div>
        </div>

        <div className="buttons-row">
          <button
            type="submit"
            className="btn"
            disabled={sending}
          >
            {sending ? "Sending…" : "Send Order"}
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleSampleCrossing}
            disabled={sending}
          >
            Send Sample Crossing Orders
          </button>
        </div>

        {error && <div className="error">Error: {error}</div>}
      </form>

      {/* <div className="last-response">
        <div className="last-response-title">Last Response</div>
        <pre className="last-response-body">
          {lastResp ? JSON.stringify(lastResp, null, 2) : "// no response yet"}
        </pre>
      </div> */}
    </section>
  );
}

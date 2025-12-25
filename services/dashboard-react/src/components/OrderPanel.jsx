import React, { useState } from "react";

const DEFAULT_SYMBOL = "TEST";

export default function OrderPanel() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [side, setSide] = useState("buy");
  const [type, setType] = useState("limit");
  const [price, setPrice] = useState("100.10");
  const [qty, setQty] = useState("5");
  const [lastResponse, setLastResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    setLastResponse(null);

    try {
      const id = "o-" + Date.now(); // simple id

      const body = {
        id,
        symbol: symbol.trim() || DEFAULT_SYMBOL,
        side,
        type, // "limit" or "market"
        qty: Number(qty),
      };

      if (type === "limit") {
        body.price = Number(price);
      }

      const res = await fetch("http://localhost:8080/order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setLastResponse(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const sendSampleCrossingOrders = async () => {
    setError("");
    setLoading(true);
    setLastResponse(null);

    try {
      const orders = [
        {
          id: "sample-buy-" + Date.now(),
          symbol: DEFAULT_SYMBOL,
          side: "buy",
          type: "limit",
          price: 100.1,
          qty: 5,
        },
        {
          id: "sample-sell-" + (Date.now() + 1),
          symbol: DEFAULT_SYMBOL,
          side: "sell",
          type: "limit",
          price: 100.05,
          qty: 5,
        },
      ];

      const results = [];
      for (const o of orders) {
        const res = await fetch("http://localhost:8080/order", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(o),
        });
        const data = await res.json();
        results.push(data);
      }
      setLastResponse({ multiple: true, responses: results });
    } catch (err) {
      console.error(err);
      setError(err.message || "Sample orders failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="order-panel">
      <h2>Manual Order Entry</h2>

      <form onSubmit={handleSubmit} className="order-form">
        <div className="field-row">
          <label>Symbol</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="TEST"
          />
        </div>

        <div className="field-row">
          <label>Side</label>
          <select value={side} onChange={(e) => setSide(e.target.value)}>
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
        </div>

        <div className="field-row">
          <label>Type</label>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="limit">Limit</option>
            <option value="market">Market</option>
          </select>
        </div>

        {type === "limit" && (
          <div className="field-row">
            <label>Price</label>
            <input
              type="number"
              step="0.01"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
        )}

        <div className="field-row">
          <label>Quantity</label>
          <input
            type="number"
            min="1"
            value={qty}
            onChange={(e) => setQty(e.target.value)}
          />
        </div>

        <div className="button-row">
          <button type="submit" disabled={loading}>
            {loading ? "Sending..." : "Send Order"}
          </button>

          <button
            type="button"
            onClick={sendSampleCrossingOrders}
            disabled={loading}
          >
            Send Sample Crossing Orders
          </button>
        </div>
      </form>

      {error && <div className="order-error">Error: {error}</div>}

      {lastResponse && (
        <div className="order-response">
          <h3>Last Response</h3>
          <pre>{JSON.stringify(lastResponse, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

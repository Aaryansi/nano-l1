import React from "react";

export default function Depth({ book }) {
  const bid = book.bestBid || { price: 0, qty: 0 };
  const ask = book.bestAsk || { price: 0, qty: 0 };

  const spread =
    bid.price > 0 && ask.price > 0 ? ask.price - bid.price : 0;

  return (
    <div className="card">
      <h2>Top of Book</h2>

      <div className="row">
        <div className="col">
          <div className="label">Best Bid</div>
          <div className="big bid">{bid.price.toFixed(2)}</div>
          <div className="small">Qty: {bid.qty.toFixed(2)}</div>
        </div>

        <div className="col">
          <div className="label">Best Ask</div>
          <div className="big ask">{ask.price.toFixed(2)}</div>
          <div className="small">Qty: {ask.qty.toFixed(2)}</div>
        </div>
      </div>

      <div className="spread">
        Spread: {spread.toFixed(2)}
      </div>

      <div className="small muted">
        Last Trade: {book.lastTradePrice?.toFixed(2) ?? "-"}
      </div>
    </div>
  );
}

import "../styles/index.css";

import OrderPanel from "../components/OrderPanel.jsx";
import Depth from "../components/Depth.jsx";
import Trades from "../components/Trades.jsx";
import PnL from "../components/PnL.jsx";
import Stats from "../components/Stats.jsx";
import PersistedTrades from "../components/PersistedTrades.jsx";
import PriceHistory from "../components/PriceHistory.jsx";

function AutoTraderStatus() {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">AutoTrader</div>
        <div className="panel-subtitle">python agent</div>
      </div>
      <div className="panel-body">
        <p className="muted">
          Polls engine for trades, runs policy, submits orders.
        </p>
        <ul className="roadmap-list">
          <li>mean reversion - buys dips, sells rips</li>
          <li>ml policy - sklearn model on binance data</li>
        </ul>
        <p className="muted-small">
          python agent.py --model-path model.pkl
        </p>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1 className="app-title">Nano-L1 Trading Sandbox</h1>
          <p className="app-subtitle">
            Go engine · WebSockets · Postgres · Python backtester
          </p>
        </div>
        <div className="app-header-right">
          <div className="badge badge-live">LIVE</div>
          <div className="badge badge-outline">Symbol: TEST</div>
        </div>
      </header>

      <main className="dashboard-grid">
        {/* Row 1: order + top of book + PnL */}
        <section className="grid-item order-area">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">Manual Order Entry</div>
              <div className="panel-subtitle">
                Send limit / market orders into the Go engine
              </div>
            </div>
            <div className="panel-body">
              <OrderPanel />
            </div>
          </div>
        </section>

        <section className="grid-item depth-area">
          <Depth />
        </section>

        <section className="grid-item pnl-area">
          <PnL />
        </section>

        {/* Row 2: tape + DB trades + DB stats */}
        <section className="grid-item trades-area">
          <Trades />
        </section>

        <section className="grid-item persisted-area">
          <PersistedTrades />
        </section>

        <section className="grid-item stats-area">
          <Stats />
        </section>

        {/* Row 3: price chart + RL card */}
        <section className="grid-item chart-area">
          <PriceHistory />
        </section>

        <section className="grid-item rl-area">
          <AutoTraderStatus />
        </section>
      </main>

      <footer className="app-footer">
        <span className="muted-small">WS: ws://localhost:8080/ws</span>
        <span className="muted-small">
          Nano-L1 · Go matching engine · Postgres · Python backtester
        </span>
      </footer>
    </div>
  );
}

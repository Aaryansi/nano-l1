"""smoke-test the resampler against real cached markets before a full ingest.

read-only. prints a per-episode summary at three settings so that an obvious
breakage (inverted spreads, empty grids, stale-forever quotes, imbalance stuck
at zero when a tape was supplied) is visible before committing to a long run.

usage:
    python scripts/smoke_episode.py [--cache data/kalshi] [--n 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.data.episode import build_episode  # noqa: E402
from nano_rl.data.kalshi import KalshiClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../../data/kalshi")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    client = KalshiClient(args.cache)
    markets = {m.ticker: m for m in client.fetch_markets()}

    cached = sorted(p.stem for p in (Path(args.cache) / "trades").glob("*.json"))
    cached = [t for t in cached if t in markets][: args.n]

    if not cached:
        print("no cached tapes found; run build_corpus.py first")
        raise SystemExit(1)

    print(f"smoke-testing {len(cached)} cached markets\n")
    failures = 0

    for ticker in cached:
        market = markets[ticker]
        candles = client.fetch_candles(market)
        trades = client.fetch_trades(ticker)
        print(f"{ticker}  settle={market.settlement_value:.0f}  raw_trades={len(trades)}")

        for step, tape in ((60, None), (60, trades), (10, trades)):
            label = f"  step={step:>2}s tape={'yes' if tape else 'no ':<3}"
            ep = build_episode(market, candles, tape, step_seconds=step)

            if ep is None:
                print(f"{label} -> None (rejected)")
                failures += 1
                continue

            spread = ep.ask - ep.bid
            checks = {
                "spread>=0": bool(np.all(spread >= 0)),
                "px in [0,1]": bool(np.all((ep.mid >= 0) & (ep.mid <= 1))),
                "grid strictly increasing": bool(np.all(np.diff(ep.t_sec) > 0)),
                "ends before close": bool(ep.t_sec[-1] < market.duration_s),
            }
            bad = [k for k, ok in checks.items() if not ok]
            if bad:
                failures += 1

            print(
                f"{label} n={ep.n_steps:>2} "
                f"mid {ep.mid[0]:.3f}->{ep.mid[-1]:.3f} "
                f"spread_med={np.median(spread):.4f} "
                f"vol_med={np.median(ep.volume):>9,.0f} "
                f"imb|max|={np.abs(ep.flow_imbalance).max():.3f} "
                f"stale_max={ep.staleness.max():>5.0f}s "
                f"{'OK' if not bad else 'FAIL ' + ','.join(bad)}"
            )
        print()

    print(f"{'all checks passed' if failures == 0 else f'{failures} FAILURES'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

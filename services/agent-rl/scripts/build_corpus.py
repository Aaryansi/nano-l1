"""build the compact episode corpus from the kalshi api.

supersedes the raw-tape caching in ingest_kalshi.py, which stored ~6.9 MB of
json per market for data that resamples down to a few kilobytes. that ingest
projected to ~44 GB across the full corpus; this one lands at tens of MB.

two modes:

    --mode candles   one api call per market, 1-minute grid, full corpus.
                     no flow imbalance. this is the primary dataset.

    --mode tape      ~20 api calls per market (the tape paginates at 1000),
                     finer grid plus real flow imbalance. measured at
                     0.20 markets/sec, so use it on a subset.

raw responses are NOT retained. only the resampled arrays are written, to a
single compressed npz per split of the corpus.

usage:
    python scripts/build_corpus.py --mode candles --out data/corpus
    python scripts/build_corpus.py --mode tape --episodes 600 --step 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.data.episode import build_episode  # noqa: E402
from nano_rl.data.kalshi import KalshiClient  # noqa: E402

log = logging.getLogger("corpus")


def main() -> None:
    ap = argparse.ArgumentParser(description="build compact kalshi episode corpus")
    ap.add_argument("--cache", default="data/kalshi", help="api cache directory")
    ap.add_argument("--out", default="data/corpus", help="corpus output directory")
    ap.add_argument("--mode", choices=["candles", "tape"], default="candles")
    ap.add_argument("--step", type=int, default=60, help="decision step in seconds")
    ap.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="cap on episodes to build (omit for the whole corpus)",
    )
    ap.add_argument(
        "--keep-raw",
        action="store_true",
        help="retain raw api json (off by default; this is what caused the 44 GB projection)",
    )
    ap.add_argument(
        "--with-spot",
        action="store_true",
        help="download binance 1s klines and add the 5 spot features",
    )
    ap.add_argument("--spot-cache", default="data/binance_klines")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = KalshiClient(cache_dir=args.cache)
    markets = client.fetch_markets()
    markets.sort(key=lambda m: m.open_time)

    if args.episodes:
        # stride across the whole history rather than taking a recent block, so
        # a subset still spans the full 68 days and the walk-forward split
        # remains meaningful.
        stride = max(1, len(markets) // args.episodes)
        markets = markets[::stride][: args.episodes]

    log.info("building %d episodes, mode=%s, step=%ds", len(markets), args.mode, args.step)

    # spot is fetched once for the whole span, then joined per episode with a
    # backward as-of lookup. fetching per episode would redownload the same
    # daily file dozens of times.
    spot_series = None
    if args.with_spot:
        from nano_rl.data.binance import build_spot_features, load_range

        log.info("downloading binance spot for the corpus span (this is ~2.5 MB/day)")
        spot_series = load_range(
            "BTCUSDT",
            markets[0].open_time,
            markets[-1].close_time,
            cache_dir=args.spot_cache,
        )

    built: list[dict] = []
    meta: list[tuple[str, float, float]] = []  # ticker, open_epoch, settlement
    spot_blocks: list[np.ndarray] = []
    skipped = 0
    n_no_spot = 0
    t0 = time.time()

    for i, m in enumerate(markets, 1):
        try:
            candles = client.fetch_candles(m)
            trades = client.fetch_trades(m.ticker) if args.mode == "tape" else None
            ep = build_episode(m, candles, trades, step_seconds=args.step)
        except Exception as exc:
            log.warning("skip %s: %s", m.ticker, str(exc)[:100])
            skipped += 1
            continue

        if ep is None:
            skipped += 1
        else:
            built.append(ep.to_arrays())
            meta.append((m.ticker, m.open_time.timestamp(), m.settlement_value))

            if spot_series is not None:
                block = build_spot_features(
                    spot=spot_series,
                    open_epoch=m.open_time.timestamp(),
                    t_sec=ep.t_sec,
                    implied_prob=ep.mid,
                )
                if not np.any(block):
                    n_no_spot += 1
                spot_blocks.append(block.astype(np.float32))

        if not args.keep_raw:
            # discard the raw json immediately; it is one to two orders of
            # magnitude larger than what we keep.
            for sub in ("trades", "candles"):
                p = Path(args.cache) / sub / f"{m.ticker}.json"
                if p.exists():
                    p.unlink()

        if i % 200 == 0 or i == len(markets):
            rate = i / (time.time() - t0)
            eta = (len(markets) - i) / rate if rate else 0
            print(
                f"  [{i}/{len(markets)}] built={len(built)} skipped={skipped} "
                f"{rate:.1f}/s eta {eta/60:.1f}m",
                flush=True,
            )

    if not built:
        log.error("no episodes built, aborting")
        raise SystemExit(1)

    # every episode has the same length for a fixed step, so we can stack.
    lengths = {len(b["t_sec"]) for b in built}
    if len(lengths) != 1:
        log.warning("ragged episode lengths %s, truncating to the shortest", lengths)
    n = min(lengths)

    stacked = {
        k: np.stack([b[k][:n] for b in built]).astype(np.float32) for k in built[0]
    }
    stacked["settlement"] = np.array([s for _, _, s in meta], dtype=np.float32)
    stacked["open_epoch"] = np.array([o for _, o, _ in meta], dtype=np.float64)

    if spot_blocks:
        stacked["spot"] = np.stack([b[:n] for b in spot_blocks]).astype(np.float32)

    name = f"corpus_{args.mode}_{args.step}s{'_spot' if spot_blocks else ''}.npz"
    path = out_dir / name
    np.savez_compressed(path, **stacked)
    tickers_path = out_dir / f"tickers_{args.mode}_{args.step}s.txt"
    tickers_path.write_text("\n".join(t for t, _, _ in meta))

    size_mb = path.stat().st_size / 1e6
    yes = float(stacked["settlement"].mean())

    print()
    print("=" * 62)
    print(f"  episodes   : {len(built)} ({skipped} skipped)")
    print(f"  steps each : {n}")
    print(f"  transitions: {len(built) * n:,}")
    print(f"  yes rate   : {yes:.4f}")
    if spot_blocks:
        print(f"  spot       : joined, {n_no_spot} episodes without coverage")
    print(f"  written    : {path}  ({size_mb:.1f} MB)")
    print(f"  elapsed    : {(time.time()-t0)/60:.1f} min")
    print("=" * 62)


if __name__ == "__main__":
    main()

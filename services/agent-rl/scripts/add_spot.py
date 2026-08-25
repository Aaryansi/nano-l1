"""augment an existing corpus with binance spot features.

separate from build_corpus.py on purpose. the kalshi ingest is rate-limited and
takes ~27 minutes for the full corpus; rebuilding it just to attach a second
data source would waste that. this reads a built corpus, downloads only the
binance days it spans, joins backward as-of, and writes a new npz.

usage:
    python scripts/add_spot.py --corpus data/corpus/corpus_candles_60s.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.data.binance import build_spot_features, load_range  # noqa: E402
from nano_rl.env.features import SPOT_FEATURES  # noqa: E402

log = logging.getLogger("add_spot")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--cache", default="../../data/binance_klines")
    ap.add_argument("--out", default=None, help="defaults to <corpus>_spot.npz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src = Path(args.corpus)
    data = dict(np.load(src))

    open_epoch = data["open_epoch"]
    t_sec = data["t_sec"]
    bid, ask = data["bid"], data["ask"]
    mid = 0.5 * (bid + ask)

    start = datetime.fromtimestamp(float(open_epoch.min()), timezone.utc)
    end = datetime.fromtimestamp(float(open_epoch.max()) + 900.0, timezone.utc)
    log.info("corpus spans %s .. %s (%d episodes)", start, end, len(open_epoch))

    spot = load_range(args.symbol, start, end, cache_dir=args.cache)

    blocks = np.zeros((len(open_epoch), t_sec.shape[1], len(SPOT_FEATURES)), dtype=np.float32)
    n_empty = 0
    for i in range(len(open_epoch)):
        b = build_spot_features(
            spot=spot,
            open_epoch=float(open_epoch[i]),
            t_sec=t_sec[i].astype(float),
            implied_prob=mid[i].astype(float),
        )
        if not np.any(b):
            n_empty += 1
        blocks[i] = b.astype(np.float32)

    data["spot"] = blocks
    out = Path(args.out) if args.out else src.with_name(src.stem + "_spot.npz")
    np.savez_compressed(out, **data)

    print()
    print("=" * 62)
    print(f"  episodes           : {len(open_epoch)}")
    print(f"  without spot cover : {n_empty}")
    print(f"  spot block shape   : {blocks.shape}")
    for j, name in enumerate(SPOT_FEATURES):
        col = blocks[:, :, j]
        print(f"    {name:<22} mean={col.mean():+.5f} std={col.std():.5f} "
              f"min={col.min():+.4f} max={col.max():+.4f}")
    print(f"  written            : {out} ({out.stat().st_size/1e6:.1f} MB)")
    print("=" * 62)


if __name__ == "__main__":
    main()

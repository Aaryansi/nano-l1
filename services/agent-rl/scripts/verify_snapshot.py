"""Check the pinned corpus, market identifiers, and chronological split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import EpisodeBatch


def verify_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"snapshot hash mismatch: {name}")
    batch = EpisodeBatch.load(str(directory / "corpus.npz"))
    tickers = (directory / "tickers.txt").read_text().splitlines()
    if len(tickers) != len(batch) or len(set(tickers)) != len(tickers):
        raise ValueError("snapshot market identifiers are missing or duplicated")
    if len(batch) != manifest["episodes"] or batch.n_steps != manifest["steps_per_episode"]:
        raise ValueError("snapshot dimensions differ from manifest")
    settings = manifest["split"]
    split = walk_forward_split(batch, train_frac=settings["train_fraction"],
                               val_frac=settings["validation_fraction"],
                               purge=settings["purge_episodes_per_boundary_side"])
    for name, part in (("train", split.train), ("validation", split.val), ("test", split.test)):
        if part.open_epoch.tolist() != settings["membership"][name]:
            raise ValueError(f"snapshot split mismatch: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify_snapshot(args.directory)
    print(f"verified snapshot: {result['episodes']} episodes")

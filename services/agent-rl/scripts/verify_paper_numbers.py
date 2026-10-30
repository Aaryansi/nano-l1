"""Verify generated manuscript inputs and compare numerical claims with a run."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('manuscript_build', ROOT / 'docs/paper/build.py')
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT / 'data/releases/research_v1')
    ap.add_argument('--reports', type=Path,
                    help='compare this complete run with published numerical claim values')
    ap.add_argument('--paper', type=Path, default=ROOT / 'docs/paper')
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data/releases/market_snapshot_v1/corpus.npz')
    ap.add_argument('--banks', type=Path)
    ap.add_argument('--replay-proof', type=Path)
    ap.add_argument('--objectives', type=Path)
    args = ap.parse_args()
    run = args.run if args.reports is None else args.reports.parent
    inputs = paper.replication_inputs(run, args.banks, args.replay_proof, args.objectives,
                                      comparison=args.reports is not None)
    if args.reports is None:
        paper.verify_build(args.run, args.corpus, args.paper, *inputs)
        print('primary and supplemental studies, manuscript statistics, source and PDF hashes verified')
    else:
        current, _ = paper.input_statistics(run, args.corpus, *inputs)
        paper.verify_artifacts(args.paper)
        published = paper.load(args.paper / 'statistics.json')
        paper.compare_claims(paper.numerical_claims(current), paper.numerical_claims(published))
        print('complete comparison run matches the published numerical claim values')


if __name__ == '__main__':
    main()

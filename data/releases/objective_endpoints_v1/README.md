# Objective endpoint replication

This separately identified supplemental study repeats both objectives at 40 and 100 updates using public numerical source [7d14189](https://github.com/Aaryansi/nano-l1/tree/7d14189e442bfe01694954b205778426ce3c9de8). `study/` preserves the completed manifest, comparison and endpoint arrays byte for byte. Five observed agents and 24 blinded references in each of four cells produce 20 observed and 96 reference policies, each evaluated on all 1,284 held-out market episodes.

Both full and masked raw endpoints are retained and reconstructed. All original full episode returns, observed/reference spans, finite diagnostics and decisions match exactly. Original and reconstructed descriptive contrasts are retained in `comparison.json`. Different objective reward units and differing reference widths prevent interpreting these contrasts as an objective-effect significance test.

No checkpoints or training logs were archived for these policies. This evidence is a repeated computation with complete endpoint arrays, not replay of the original objective weights. It does not retroactively add masked arrays to the original release. The manifest identifies the public numerical source and runtime; its original external orchestration wrapper was not archived. The public runner now documents the repeatable workflow.

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/objective_study.py \
  --out data/releases/objective_endpoints_v1/study --verify-only
```

See [supplemental verification](../../../docs/REPLICATION_EVIDENCE.md) for fresh replication and combined reconstruction. The primary release and corpus must also pass their own checks before these studies can supply manuscript results.

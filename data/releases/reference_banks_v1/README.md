# Independent reference-bank study

This immutable supplemental study uses training source [770d484](https://github.com/Aaryansi/nano-l1/tree/770d484cd6baed649de4ea0caa33d368b082990c) and the [declared protocol](../../../docs/REFERENCE_BANK_PROTOCOL.md). It is separate from the original `research_v1` release.

Twenty independently generated banks contain 24 references and 30 observed trials at each of strengths 0, 0.2 and 1. Every policy trains for 40 updates on a fresh 1,200-episode corpus and evaluates on a separate fresh 1,200-episode corpus, with distinct policy and masking seeds. The archive contains all 4,602 original files, including 2,280 checkpoints and their update logs. `bundle/archive.json` indexes exact member sizes and hashes. No files were dropped or rewritten during packaging.

`replay.json` records successful endpoint replay of every checkpoint from the extracted archive. It binds the study manifest and frozen numerical verifier. Reading this proof checks its identity; it does not perform a new replay. Reproduce the replay with a fresh output path:

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/verify_replications.py \
  --banks data/releases/reference_banks_v1/bundle \
  --replay-proof runs/reproductions/new-bank-replay.json --replay-models
```

The [combined verification guide](../../../docs/REPLICATION_EVIDENCE.md) documents extraction, fresh training and publication reconstruction. Git history must contain the recorded source commits. The archive SHA-256 is `8baecb9f86e2ad39d5fb43399a032e4761c609ec65df3b7ca0deec706c9fce73`.

## Interpretation

All three rules flag 17/600, 575/600 and 600/600 trials at strengths 0, 0.2 and 1 respectively, with no unresolved trials or trial-level disagreements. The independent units for marginal uncertainty are 20 banks, not 600 pooled trials. The pointwise bounded-bank interval for the zero-strength rate has upper endpoint 0.24535715395, so these observations do not certify 5% error control. Method agreement is empirical, not equivalence. Signal flag rates combine learning and comparison at the declared budget.

The new study separates masking and policy seeds, unlike the older conditional study. Differences between those studies cannot be attributed solely to bank replication. Neither study supplies exchangeability for market policies.

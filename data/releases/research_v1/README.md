# Selected research release

This directory preserves the completed full-budget experiment used by the manuscript. Training and experiment production used commit [dcc4e3f31c696f818b339df8fd8879534798abc3](https://github.com/Aaryansi/nano-l1/tree/dcc4e3f31c696f818b339df8fd8879534798abc3). It uses the [fixed market snapshot](../market_snapshot_v1/) and the declared [protocol](../../../docs/EXPERIMENT_PROTOCOL.md).

`reports/run_manifest.json` records the source commit, hashes, package versions, numerical runtime, split membership, checkpoints, training logs and complete report membership. `checkpoints/` contains both market objectives, every trained market seed and its log. `reports/` contains raw endpoint arrays, summaries and auxiliary diagnostics. These files are preserved byte for byte from the completed experiment; generated publication inputs are stored separately under [docs/paper](../../../docs/paper/).

From the repository root, with the pinned Python 3.12 environment installed:

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/verify_release.py \
  --run data/releases/research_v1 \
  --corpus data/releases/market_snapshot_v1/corpus.npz
"$PY" services/agent-rl/scripts/verify_paper_numbers.py
```

The first command verifies immutable content, historical source, full budgets, primary raw-summary reconstruction and complete market trajectory parity evidence. The second reconstructs manuscript inputs and checks publication source and output hashes. Git history must contain the recorded training commit. CI also reruns all market checkpoint trajectories against the saved evaluation arrays.

The market replay proof was produced by the public auditor after training and before manifest completion. Its source hash is recorded separately in `reports/rollout_parity.json`; the training commit predates that auditor. The verifier requires the recorded auditor to match the current public implementation.

## Interpretation and reconstruction scope

The primary reference and calibration endpoint archives, market evaluation, state-preservation arrays, control-task endpoints and steering episode returns support raw reconstruction. Some secondary objective/reference diagnostics retain masked means without both endpoint arrays. Their scope is narrower. Correlated control checkpoints and independent training seeds are recorded separately.

Legacy raw diagnostics can contain non-finite standardized values when a reference has zero width. Those are preserved as produced, alongside their unresolved decisions. The generated manuscript statistics use strict JSON and represent undefined standardized quantities as null. Two unresolved results do not count as agreement.

Market versus synthetic references are descriptive comparisons without an established exchangeability assumption. Synthetic calibration rates are conditional on one reused bank. Failed competence gates, assumed execution costs, retained price information in inventory and limited temporal coverage remain limitations. Computational validation does not establish explanation fidelity, absent information or live executable performance.

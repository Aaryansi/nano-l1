# Research implementation

The binary-market environment, PPO trainer, explanation methods, reference constructions and tests used by the paper live here. The full experiment entry point is [reproduce.sh](../../reproduce.sh) at the repository root. Computation uses the CPU with pinned Python dependencies.

| Directory | Purpose |
|---|---|
| `nano_rl/env/` | Market accounting, observations, synthetic and shuffled-outcome corpora |
| `nano_rl/agents/` | Market PPO and actor-critic network |
| `nano_rl/explain/` | Return interventions, Shapley and IG targets, ranking diagnostics and reference rules |
| `nano_rl/envs/` | Control-task wrappers and independent control trainer |
| `scripts/` | Experiment producers, release verification and publication reconstruction |
| `tests/` | Accounting, trajectory parity, exact attribution games, statistical states and pipeline regressions |

## Entry points

Run from the repository root after installing [requirements.txt](requirements.txt) into a Python 3.12 environment:

```sh
PY=services/agent-rl/.venv/bin/python
(cd services/agent-rl && .venv/bin/python -m pytest)
"$PY" services/agent-rl/scripts/verify_release.py \
  --run data/releases/research_v1 \
  --corpus data/releases/market_snapshot_v1/corpus.npz
"$PY" services/agent-rl/scripts/verify_paper_numbers.py
```

`run_manifest.py` validates experiment outputs and records provenance. `audit_market_rollouts.py` compares complete scalar/vector observations, actions, accounting and saved returns for every market checkpoint. `verify_release.py` checks selected evidence and declared budgets; `--allow-quick` explicitly limits its result to execution evidence. `publication_statistics.py` reconstructs numerical manuscript inputs, with undefined standardized quantities represented by null.

New `positive_control.py` runs retain full and masked episode returns for both objective arms, every observed agent and every blinded reference at each budget. They record corpus and archive hashes, train-only normalization, episode identities and policy/masking seeds. The producer rejects a corpus changed during execution and runs `verify_objective_control.py` before returning successfully. To repeat this read-only check, use `"$PY" services/agent-rl/scripts/verify_objective_control.py --reports RUN/reports --corpus CORPUS`. It reconstructs spans, reference widths, decisions and descriptive paired contrasts. It checks recorded identities and computations, without replaying training or saved policies. Existing objective evidence is refused rather than overwritten. The selected `research_v1` predates this schema and retains its documented narrower reconstruction scope.

`verify_paper_numbers.py` checks the selected experiment against the generated manuscript and all publication hashes. Its `--reports PATH` mode compares a separate completed full run with the published numerical claims. It never falls back to another run's results. [docs/paper/build.py](../../docs/paper/build.py) builds all profiles from the shared article, using verified full-budget data.

## Reference interpretation

`calibrate_reference_banks.py` implements the separate [independent-bank study](../../docs/REFERENCE_BANK_PROTOCOL.md). It uses fresh reference and observed corpora per bank, archives both endpoints and saved policies, and verifies hashes, seed membership, decisions and bank-level summaries. Its `--verify-only --replay-models` mode replays every archived policy. `--quick` is an execution check; the full declared study requires committed clean source. `bank_calibration_statistics.py` computes pointwise bounded-mean intervals across banks and paired method contrasts, without pooling dependent trials into a binomial interval.

`nano_rl/explain/sanity.py` retains the legacy reference-centered rank/normal diagnostic with outside, within and unresolved outcomes. A zero-width reference or insufficient rank resolution is unresolved under this rule. Standard upper-tail and pooled-symmetric rank comparators are evaluated on the same synthetic calibration trials. Their marginal rank guarantees require exchangeability; a metadata match alone does not prove that assumption.

Each endpoint span concerns a declared input intervention. Probability attribution, critic attribution, return intervention, feature-level fidelity and task utility are separate targets. See the [protocol](../../docs/EXPERIMENT_PROTOCOL.md), [implemented model](../../docs/MDP.md) and [shared manuscript](../../docs/paper/article.tex).

## Auxiliary tools

Historical ingestion and exploratory training tools remain available for new experiments, including `verify_kalshi_data.py`, `verify_kalshi_candles.py`, `smoke_episode.py`, `diagnose_null.py`, `diagnose_ppo.py` and `sweep_abstention.py`. They are outside the fixed release protocol and do not establish the manuscript's numerical claims. Downloading a new API window produces a new dataset.

The CI smoke script writes isolated reduced-budget outputs, exercises the full verification path and explicitly reports that scientific budgets were not checked. It never updates the selected release.

# agent-rl

everything the paper runs. the environment, the PPO agent, the attribution
implementations, the four null constructions, and the tests that pin them.

nothing here needs a GPU. the full pipeline is driven from `reproduce.sh` at
the repository root, not from this directory.

## layout

| | |
|---|---|
| `nano_rl/env/` | binary-market env, features, synthetic corpora, permuted-outcome corpora |
| `nano_rl/agents/` | PPO and the actor-critic network |
| `nano_rl/explain/` | Shapley, integrated gradients, trajectory attribution, masking, the decision rule |
| `nano_rl/envs/` | gym wrappers for the control tasks, including the blinded-observation wrapper |
| `scripts/` | one script per experiment; see below |
| `tests/` | 312 tests, run with `pytest` |
| `runs/` | trained checkpoints, five seeds |

## the decision rule

`nano_rl/explain/sanity.py` is the module every verdict in the paper comes
from. it is small and worth reading first. it has already shipped two bugs of
the kind that do not announce themselves, both now pinned by tests: a degenerate
null returning `z = 0.0` and silently converting an observation outside the
reference into a null result, and a rank statistic whose floor of `1/(n+1)`
made rejection impossible at the budget then in use.

## scripts

most scripts correspond to a section of the paper and are called in order by
`reproduce.sh`. run that rather than invoking them individually.

six are **not** called by `reproduce.sh`. they are kept because they are the
provenance of claims made in `docs/AGENT_NOTES.md` and `docs/REPORT.md`, and
deleting them would leave those claims unsourced:

| | |
|---|---|
| `diagnose_null.py` | why tabular Q churns on a corpus with no signal |
| `diagnose_ppo.py` | why PPO collapsed to always-flat on a corpus with real signal |
| `sweep_abstention.py` | can PPO learn to abstain on noise, and under what settings |
| `smoke_episode.py` | read-only check of the resampler against cached markets |
| `verify_kalshi_data.py` | the original de-risking spike: is a causal dataset buildable |
| `verify_kalshi_candles.py` | do the exchange's candles carry bid/ask history |

## running

```sh
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest              # 312 tests
./.venv/bin/python scripts/verify_paper_numbers.py   # 186 claims vs artifacts
```

`verify_paper_numbers.py` reads `reports/*.json` and asserts every numerical
claim in `docs/paper/main.tex` against the artifact that produced it. it exits
non-zero on a mismatch and runs as the last stage of `reproduce.sh`.

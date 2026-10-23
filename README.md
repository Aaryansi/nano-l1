# nano-l1

A reproducible audit of reference distributions for observation-replacement return spans in reinforcement learning. The experiments use a binary-contract simulator, synthetic populations with declared signal, and six control-task configurations.

[Full paper](docs/paper/main.pdf) | [Core profile](docs/paper/core/main.pdf) | [Short profile](docs/paper/workshop.pdf) | [Shared source](docs/paper/article.tex)

## What the study supports

A return span measures the effect of a specified input intervention on a particular policy. Its interpretation depends on the replacement distribution, financial state, reference training procedure and evaluation population. The endpoint sum alone cannot establish feature-level explanation fidelity.

The corrected release retains negative market performance, training-seed variation, failed competence gates and unresolved references. Market comparisons with synthetic policies are descriptive. Synthetic calibration is conditional on one reference bank, and the control configurations do not establish a universal ranking of reference constructions. The paper includes an exact finite-pool counterexample to the legacy reference-centered rank/normal rule and states the assumptions of standard exchangeable ranks.

Tables, numerical paragraphs and publication figures are generated from one verified statistics artifact. The three manuscript profiles share one article. They are general article layouts, without a claim of compliance with a particular venue's submission format.

## Selected evidence

| Location | Contents |
|---|---|
| [Research release](data/releases/research_v1/) | Immutable full-run reports, raw episode arrays, market checkpoints and training logs |
| [Generated statistics](docs/paper/statistics.json) | Reconstructed manuscript inputs, explicit undefined values and verification scope |
| [Manuscript manifest](docs/paper/manuscript_manifest.json) | Publication source and artifact hashes |
| [Fixed market snapshot](data/releases/market_snapshot_v1/) | Corpus, episode identifiers, hashes and chronological split membership |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Budgets, replication units, competence gates and interpretation limits |
| [Implemented model](docs/MDP.md) | Observations, transitions, rewards and assumed execution costs |
| [Artifact guide](docs/REPORT.md) | Verification commands and limitations |
| [Research code](services/agent-rl/) | Environment, trainers, explanations, experiment scripts and tests |

The historical market window was used during development. The selected rerun is retrospective and does not create a prospective temporal holdout. Simulated costs and candle-derived quotes do not establish live executable profit or exact historical transaction costs.

## Verify the selected release

Use Python 3.12 and the pinned dependencies. The runner uses `uv` to create the environment when needed.

```sh
uv venv --python 3.12 services/agent-rl/.venv
uv pip install --python services/agent-rl/.venv/bin/python -r services/agent-rl/requirements.txt
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/verify_release.py \
  --run data/releases/research_v1 \
  --corpus data/releases/market_snapshot_v1/corpus.npz
"$PY" services/agent-rl/scripts/verify_paper_numbers.py
```

Verification checks the recorded training commit in Git history, corpus and artifact hashes, declared full budgets, raw-summary reconstruction and complete market replay evidence. A normal Git clone includes that history. The publication verifier also checks every generated file and its source hash.

To rebuild the three PDFs, install Tectonic and Poppler, then run:

```sh
services/agent-rl/.venv/bin/python docs/paper/build.py
```

The build stages compilation before replacing outputs and refuses incomplete or reduced-budget evidence. Temporary layout fixtures cannot verify a publication.

## Run a new experiment

```sh
./reproduce.sh --tests-only
./reproduce.sh --print-config
./reproduce.sh --quick
./reproduce.sh
```

Each invocation uses a new directory under `runs/reproductions/`. `RUN_ROOT` selects a named output location; nonempty output directories are refused. The fixed snapshot is verified before training. `CORPUS=/path/to/corpus.npz` selects a different dataset for a separately identified experiment. Source, corpus, splits, package versions, runtime settings, checkpoints and outputs are recorded in its manifest.

Quick and CI smoke runs check execution with deliberately insufficient scientific budgets. They cannot support publication claims. `./reproduce.sh --verify-paper` compares a newly completed full run with the selected numerical claims. A different runtime or dataset can produce different results and must remain separately identified.

`CHECKPOINTS=/path/to/run/checkpoints ./reproduce.sh --explain-only` runs auxiliary explanations into fresh output directories. It does not repeat the full protocol. Publication artifact promotion is separate from experiment execution.

## Next scientific tests

1. Repeat calibration across independent reference banks, with enough null trials to constrain rare error rates.
2. Compare reference procedures using competent agents and matched learning budgets, retaining failures.
3. Validate sequential conditional interventions in a simulator with a known joint path/outcome law.
4. Freeze a new temporal market holdout before further model selection.

## Matching engine

The repository also contains a Go matching engine, sklearn agent, backtester, dashboard and feed simulator. The research paper evaluates the reinforcement-learning experiments. Matching-engine architecture, APIs and benchmarks are documented in [docs/ENGINE.md](docs/ENGINE.md).

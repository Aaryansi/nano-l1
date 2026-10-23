# Research artifacts and verification

The current results are in the [full manuscript](paper/main.pdf), generated from the [selected research release](../data/releases/research_v1/). The [statistics artifact](paper/statistics.json) supplies its numerical tables, paragraphs and figures. This guide avoids maintaining a second hand-written set of result values.

The [experiment protocol](EXPERIMENT_PROTOCOL.md) declares budgets, competence gates, replication units and uncertainty methods. The [implemented model](MDP.md) describes the financial accounting and observation interventions. The [research-code guide](../services/agent-rl/README.md) describes the executable entry points.

## Verification

From the repository root, after installing the pinned research dependencies:

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/verify_release.py \
  --run data/releases/research_v1 \
  --corpus data/releases/market_snapshot_v1/corpus.npz
"$PY" services/agent-rl/scripts/verify_paper_numbers.py
```

The release verifier checks source against the recorded training commit, hashes, membership, full budgets, primary endpoint summaries and market trajectory replay evidence. The manuscript verifier reconstructs its inputs and checks publication source, tables, figures and PDF hashes. Rebuilding the PDFs uses `"$PY" docs/paper/build.py` with Tectonic and Poppler installed.

## Limitations

Computational checks establish the stated verification scope. They do not establish feature-level fidelity, market efficiency, absent information, behavioural equivalence or a universal null distribution. Market policies and synthetic references lack an established exchangeability relationship. Conditional calibration intervals omit variation across independent reference banks.

The outcome attribution is a repeated empirical input-replacement intervention. It is inspired by the study of performance explanations, but does not reproduce SVERL's conditional occupancy construction or its local partial-information policy. The manuscript describes the distinction and cites the primary framework.

The financial model uses assumed costs, coarse quotes and full fills. Retained inventory coordinates can reveal price; donor rows or paths do not guarantee consistency with the recipient's financial state. Failed control-task competence and unresolved references remain evidence rather than being counted as agreement or low power.

The selected window was used during development. A new temporal holdout is needed for prospective robustness. Some secondary diagnostics retain masked spans as means without both per-episode endpoint arrays; those records have a narrower reconstruction scope than the primary, calibration, state and control endpoints.

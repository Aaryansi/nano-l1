# Research artifacts and verification

The current results are in the [full manuscript](paper/main.pdf), generated from the [selected primary release](../data/releases/research_v1/), [independent-bank study](../data/releases/reference_banks_v1/) and [objective replication](../data/releases/objective_endpoints_v1/). The [statistics artifact](paper/statistics.json) supplies its numerical tables, paragraphs and figures. This guide avoids maintaining a second hand-written set of result values.

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

The release verifier checks source against the recorded training commit, hashes, membership, full budgets, primary endpoint summaries and market trajectory replay evidence. The manuscript verifier reconstructs all three studies, checks the recorded bank replay proof identity and checks publication source, tables, figures and PDF hashes. CI performs fresh endpoint replay of all 2,280 archived bank checkpoints. The [supplemental guide](REPLICATION_EVIDENCE.md) documents local replay and new-study commands. Rebuilding the PDFs uses `"$PY" docs/paper/build.py` with Tectonic and Poppler installed.

## Limitations

Computational checks establish the stated verification scope. They do not establish feature-level fidelity, market efficiency, absent information, behavioural equivalence or a universal null distribution. Market policies and synthetic references lack an established exchangeability relationship. Original conditional calibration intervals omit bank variation. The separate 20-bank study measures that variation with banks as independent units; its pointwise conservative intervals remain too wide to certify 5% error control. Agreement among all three rules is empirical, not equivalence.

The outcome attribution is a repeated empirical input-replacement intervention. It is inspired by the study of performance explanations, but does not reproduce SVERL's conditional occupancy construction or its local partial-information policy. The manuscript describes the distinction and cites the primary framework.

The financial model uses assumed costs, coarse quotes and full fills. Retained inventory coordinates can reveal price; donor rows or paths do not guarantee consistency with the recipient's financial state. Failed control-task competence and unresolved references remain evidence rather than being counted as agreement or low power.

The selected window was used during development. A new temporal holdout is needed for prospective robustness. Some original secondary diagnostics retain masked means without both endpoint arrays. The objective supplement now retains both endpoints and reproduces every original return/span comparison exactly, without retroactively filling the old archive. Its policies have no archived checkpoints or training logs. Other original secondary records retain their narrower reconstruction scope.

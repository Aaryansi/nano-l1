# Supplemental replication evidence

The independent-bank study and objective endpoint replication are separate from the immutable `research_v1` release. Their completed outputs must pass their own source, membership, raw-result and budget checks before publication. Execution checks use explicit reduced budgets and cannot verify the scientific protocol.

## Independent reference banks

Follow [REFERENCE_BANK_PROTOCOL.md](REFERENCE_BANK_PROTOCOL.md) to generate a fresh study from clean committed source. Preserve all banks and policies. A compact archive retains every original file, including model weights and training logs:

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/study_archive.py pack \
  --source runs/reproductions/reference-banks-v1 --out runs/reproductions/bank-bundle
"$PY" services/agent-rl/scripts/verify_replications.py \
  --banks runs/reproductions/bank-bundle \
  --replay-proof runs/reproductions/bank-replay.json --replay-models
```

The archive index records every relative name, byte count and SHA-256. Extraction verifies archive and member hashes, requires canonical regular-file members, refuses existing destinations and installs a complete verified tree. Packaging preserves bytes; it does not certify the scientific content by itself. The study verifier reconstructs that content after extraction.

`--replay-models` regenerates each agent's data and normalization, loads its saved weights, and reproduces both endpoints. The fresh external proof binds the completed study manifest and frozen verifier source to that replay. Subsequent reading of the proof verifies its identity without claiming a new checkpoint replay. To extract a persistent copy, run `study_archive.py unpack --source BUNDLE --out FRESH_DIRECTORY`.

## Objective endpoint replication

Repeat both original objectives at 40 and 100 updates, with five observed agents and 24 blinded references in each cell:

```sh
"$PY" services/agent-rl/scripts/objective_study.py \
  --out runs/reproductions/objective-endpoints-v1
"$PY" services/agent-rl/scripts/objective_study.py \
  --out runs/reproductions/objective-endpoints-v1 --verify-only
```

The runner records clean public source, package/runtime settings and corpus identity, then checks that they stayed fixed. It verifies both raw endpoints and compares every original unmasked episode return, observed/reference span, finite diagnostic and decision with the selected release, allowing only 1e-9 absolute numeric roundoff. Undefined zero-scale quantities are represented by null. Original and reconstructed descriptive contrasts are both retained for review. No checkpoint or training-log archive is claimed for these objective policies. The source snapshot identifies the public numerical implementation; the original supplemental run used an external launch wrapper, while this public runner provides the repeatable command.

The original release retains its narrower raw-evidence scope. A successful replication supports the repeated quantities; it does not retroactively create original masked arrays or establish feature fidelity. Changed data or an unmatching result is a separate experiment, not a verified numerical reproduction. Failed runs remain failed and cannot pass the publication gate.

## Combined reconstruction

```sh
"$PY" services/agent-rl/scripts/verify_replications.py \
  --banks runs/reproductions/bank-bundle \
  --replay-proof runs/reproductions/bank-replay.json \
  --objectives runs/reproductions/objective-endpoints-v1 \
  --out runs/reproductions/replication-statistics.json
```

The numerical artifact records independent bank rates and pointwise bounds, unresolved rates, paired method contrasts and trial-level outcome disagreements. It also records the objective replication and its comparison with the original release. Trial agreement is checked directly; equal aggregate rates alone do not imply equal trial outcomes. The artifact identifies each input manifest, the replay proof and the analysis sources.

CI exercises the same workflow with reduced budgets and an explicitly supplied matching execution reference. `objective_study.py --quick` and verification with `--allow-quick` remain execution-only checks. A manuscript must require both full studies explicitly and must not substitute a selected cached study for a missing new experiment.

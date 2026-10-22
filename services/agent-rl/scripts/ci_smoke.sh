#!/usr/bin/env bash
# Execution checks with deliberately insufficient budgets for scientific claims.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="${SMOKE_OUT:-$(mktemp -d)}"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "refusing nonempty smoke output: $OUT"
  exit 1
fi
CORPUS="$ROOT/data/releases/market_snapshot_v1/corpus.npz"
cd "$ROOT/services/agent-rl"
"$PY" scripts/verify_snapshot.py "$ROOT/data/releases/market_snapshot_v1"
"$PY" scripts/run_manifest.py --corpus "$CORPUS" --checkpoints "$OUT/checkpoints" \
  --out "$OUT/reports" --phase started --quick 1
"$PY" scripts/train_ppo.py --corpus "$CORPUS" --seeds 2 --updates 2 \
  --out "$OUT/checkpoints/ppo"
"$PY" scripts/train_ppo.py --corpus "$CORPUS" --seeds 2 --updates 2 \
  --frictionless --out "$OUT/checkpoints/ppo_nofric"
"$PY" scripts/evaluate.py --corpus "$CORPUS" --runs "$OUT/checkpoints/ppo" \
  --runs-nofric "$OUT/checkpoints/ppo_nofric" --out "$OUT/reports"
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/calibrate_detection.py \
  --out "$OUT/reports" --n-null 2 --updates 1 --trials 1 --strengths 0 1
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/sanity_check_explanations.py \
  --corpus "$CORPUS" --runs "$OUT/checkpoints/ppo" --out "$OUT/reports/headline" \
  --n-null 2 --updates 1 --episodes 8 --coalitions 8
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/calibrate_detection.py \
  --out "$OUT/reports/reused-bank" --n-null 2 --updates 1 --trials 1 --strengths 0 1 \
  --reference-json "$OUT/reports/headline/sanity_test.json"
"$PY" scripts/state_masking.py --corpus "$CORPUS" --runs "$OUT/checkpoints/ppo" \
  --out "$OUT/reports" --seeds 2 --updates 1 --episodes 64
"$PY" scripts/horizon_scaling.py --out "$OUT/reports/horizon" \
  --horizons 2 3 4 --n-null 2 --updates 1 --episodes 32
"$PY" scripts/stability.py --corpus "$CORPUS" --runs "$OUT/checkpoints/ppo" \
  --out "$OUT/reports" --n-states 2 --outcome-coalitions 8 --outcome-episodes 8
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/null_corpus_check.py --corpus "$CORPUS" \
  --runs "$OUT/checkpoints/ppo" --out "$OUT/reports" --n-null 2 --updates 1
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/sverl_targets.py --corpus "$CORPUS" \
  --runs "$OUT/checkpoints/ppo" --out "$OUT/reports" --n-null 2 --updates 1 --states 8
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/second_method.py --corpus "$CORPUS" \
  --runs "$OUT/checkpoints/ppo" --out "$OUT/reports" --n-null 2 --updates 1 --n-states 2
"$PY" scripts/parameter_randomization.py --corpus "$CORPUS" \
  --runs "$OUT/checkpoints/ppo" --out "$OUT/reports" --n-states 2 --seeds 1 --updates 1 \
  --outcome-seeds 1 --outcome-coalitions 8 --outcome-episodes 8
"$PY" - "$OUT/reports" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
corpus = json.loads((root / "null_corpus_check.json").read_text())
assert corpus["constructions_agree"] is None
assert corpus["decision_summary"]["n_unresolved"] == 2
targets = json.loads((root / "sverl_targets.json").read_text())
assert targets["targets_agree_under_signal_free_null"] is None
for target in targets["targets"].values():
    expected = np.std(target["null_blinded"]["spans"], ddof=1) <= 1e-12
    assert target["blinded_degenerate"] == expected
method = json.loads((root / "second_method.json").read_text())
assert method["cross_family_return_test_comparison"]["performed"] is False
assert "families_agree" not in method
assert len(method["checkpoints"]) == 2
def reject_constant(token):
    raise ValueError(f"nonfinite JSON diagnostic: {token}")
parameter = json.loads((root / "parameter_randomization.json").read_text(),
                       parse_constant=reject_constant)
assert "separates_agents" not in parameter
assert "outcome_target_agrees_with_behaviour" not in parameter
PY
ALLOW_UNRESOLVED_NULLS=1 "$PY" scripts/generalize_gym.py --envs FrozenLake-v1 CliffWalking-v0 --jobs 2 \
  --trainer sb3 --steps 512 --null-steps 512 --n-null 2 --agent-seeds 2 \
  --attr-episodes 2 --out "$OUT/reports/controls"
"$PY" scripts/run_manifest.py --corpus "$CORPUS" --checkpoints "$OUT/checkpoints" \
  --out "$OUT/reports" --phase complete --quick 1
echo "smoke execution verified; reduced budgets do not support publication claims"

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
"$PY" scripts/state_masking.py --corpus "$CORPUS" --runs "$OUT/checkpoints/ppo" \
  --out "$OUT/reports" --seeds 2 --updates 1 --episodes 64
"$PY" scripts/run_manifest.py --corpus "$CORPUS" --checkpoints "$OUT/checkpoints" \
  --out "$OUT/reports" --phase complete --quick 1
echo "smoke execution verified; reduced budgets do not support publication claims"

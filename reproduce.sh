#!/usr/bin/env bash
#
# regenerate every result in docs/REPORT.md from a clean checkout.
#
# no api keys and no accounts. both data sources are public:
#   kalshi     api.elections.kalshi.com   read-only, unauthenticated
#   binance    data.binance.vision        static daily files
#
# runtime is dominated by the kalshi ingest, which is rate limited. everything
# is cached, so a second run skips straight to training.
#
# usage:
#   ./reproduce.sh              full pipeline
#   ./reproduce.sh --quick      fewer seeds and updates, for a smoke test
#   ./reproduce.sh --tests-only just run the test suite

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
RL="$ROOT/services/agent-rl"
PY="$RL/.venv/bin/python"

SEEDS=5
UPDATES=100
EXPLAIN_UPDATES=60
QUICK=0
TESTS_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --quick)      QUICK=1; SEEDS=2; UPDATES=30; EXPLAIN_UPDATES=20 ;;
    --tests-only) TESTS_ONLY=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- environment
step "environment"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. install it: https://docs.astral.sh/uv/getting-started/"
  echo "(uv is used because it pins the python version as well as the deps)"
  exit 1
fi

if [ ! -x "$PY" ]; then
  echo "creating venv at $RL/.venv"
  uv venv --python 3.12 "$RL/.venv"
fi
uv pip install --quiet --python "$PY" -r "$RL/requirements.txt"
echo "python: $("$PY" --version)"

# --------------------------------------------------------------------- tests
step "test suite"
( cd "$RL" && "$PY" -m pytest -q )

if [ "$TESTS_ONLY" = "1" ]; then
  echo "tests only, stopping here"
  exit 0
fi

# ---------------------------------------------------------------------- data
CORPUS="$ROOT/data/corpus/corpus_candles_60s_spot.npz"

if [ -f "$CORPUS" ]; then
  step "corpus already built, skipping ingest"
else
  step "building the kalshi corpus (rate limited, ~30 min on a cold cache)"
  ( cd "$RL" && "$PY" scripts/build_corpus.py \
      --mode candles --step 60 \
      --cache "$ROOT/data/kalshi" --out "$ROOT/data/corpus" )

  step "joining binance spot (~170 MB of 1s klines)"
  ( cd "$RL" && "$PY" scripts/add_spot.py \
      --corpus "$ROOT/data/corpus/corpus_candles_60s.npz" \
      --cache "$ROOT/data/binance_klines" )
fi

step "validating the corpus end to end"
( cd "$RL" && "$PY" scripts/validate_corpus.py --corpus "$CORPUS" )

# ------------------------------------------------------------ sanity checks
step "sanity: does the learning loop work on problems with known answers?"
( cd "$RL" && "$PY" scripts/sanity_tabular.py \
    --learn-episodes $([ "$QUICK" = 1 ] && echo 800 || echo 3000) \
    --null-episodes  $([ "$QUICK" = 1 ] && echo 4000 || echo 20000) )
( cd "$RL" && "$PY" scripts/sanity_ppo.py --updates $((UPDATES / 2)) )

# ------------------------------------------------------------------ training
step "training ppo, $SEEDS seeds, $UPDATES updates"
( cd "$RL" && "$PY" scripts/train_ppo.py \
    --corpus "$CORPUS" --seeds "$SEEDS" --updates "$UPDATES" \
    --entropy-coef 0.01 --out runs/ppo )

step "training the zero-cost ablation"
( cd "$RL" && "$PY" scripts/train_ppo.py \
    --corpus "$CORPUS" --seeds "$SEEDS" --updates "$UPDATES" \
    --entropy-coef 0.01 --frictionless --out runs/ppo_nofric )

# ---------------------------------------------------------------- evaluation
step "evaluating on the held-out test split"
( cd "$RL" && "$PY" scripts/evaluate.py \
    --corpus "$CORPUS" --runs runs/ppo --runs-nofric runs/ppo_nofric \
    --out "$ROOT/reports" )

# ------------------------------------------------------------ explainability
step "shapley explanations, with the ground-truth validation"
( cd "$RL" && "$PY" scripts/explain.py \
    --corpus "$CORPUS" --runs runs/ppo \
    --out "$ROOT/reports" --updates "$EXPLAIN_UPDATES" )

step "attribution stability across seeds"
( cd "$RL" && "$PY" scripts/stability.py \
    --corpus "$CORPUS" --runs runs/ppo --out "$ROOT/reports" )

step "null-model test: is the explanation distinguishable from nothing?"
( cd "$RL" && "$PY" scripts/sanity_check_explanations.py \
    --corpus "$CORPUS" --runs runs/ppo --out "$ROOT/reports" \
    --n-null $([ "$QUICK" = 1 ] && echo 8 || echo 24) )

step "faithfulness: the decoy experiment and deletion curves"
( cd "$RL" && "$PY" scripts/faithfulness.py \
    --out "$ROOT/reports" --updates "$EXPLAIN_UPDATES" )

# -------------------------------------------------------------------- done
step "done"
echo "figures and json in $ROOT/reports:"
ls -1 "$ROOT/reports"
echo
echo "read docs/REPORT.md for what they mean."

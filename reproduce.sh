#!/usr/bin/env bash
#
# regenerate every figure and number in docs/PAPER.md and docs/REPORT.md from a
# clean checkout.
#
# no api keys and no accounts. every data source is public:
#   kalshi     api.elections.kalshi.com   read-only, unauthenticated
#   binance    data.binance.vision        static daily files
#   gymnasium  CartPole-v1, Acrobot-v1    installed with the deps
#
# note: kalshi serves a moving window, so a rebuild on a different day returns
# a slightly different set of markets. verified: overlapping episodes are
# byte-identical, only membership changes. expect figures close to the
# published ones rather than identical.
#
# runtime is dominated by the kalshi ingest, which is rate limited to about 27
# minutes on a cold cache. everything is cached, so a second run skips to
# training. the full pipeline is roughly 2 hours cold, 1 hour warm.
#
# usage:
#   ./reproduce.sh              full pipeline
#   ./reproduce.sh --quick      reduced seeds and budgets, for a smoke test
#   ./reproduce.sh --tests-only just the test suite
#   ./reproduce.sh --explain-only  skip data and training, reuse checkpoints

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
PIPELINE_START=$(date +%s)
RL="$ROOT/services/agent-rl"
PY="$RL/.venv/bin/python"
REPORTS="$ROOT/reports"

SEEDS=5
UPDATES=100
EXPLAIN_UPDATES=60
N_NULL=24
STEER_SEEDS=3
GYM_STEPS=400000
QUICK=0
TESTS_ONLY=0
EXPLAIN_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --quick)
      QUICK=1; SEEDS=2; UPDATES=30; EXPLAIN_UPDATES=20
      N_NULL=8; STEER_SEEDS=2; GYM_STEPS=60000 ;;
    --tests-only)   TESTS_ONLY=1 ;;
    --explain-only) EXPLAIN_ONLY=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- environment
step "environment"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found: https://docs.astral.sh/uv/getting-started/"
  echo "(used because it pins the python version as well as the dependencies)"
  exit 1
fi
[ -x "$PY" ] || uv venv --python 3.12 "$RL/.venv"
uv pip install --quiet --python "$PY" -r "$RL/requirements.txt"
echo "python: $("$PY" --version)"

# --------------------------------------------------------------------- tests
step "test suite"
( cd "$RL" && "$PY" -m pytest -q )
[ "$TESTS_ONLY" = "1" ] && { echo "tests only, stopping"; exit 0; }

CORPUS="$ROOT/data/corpus/corpus_candles_60s_spot.npz"

if [ "$EXPLAIN_ONLY" = "0" ]; then
  # ------------------------------------------------------------------- data
  if [ -f "$CORPUS" ]; then
    step "corpus already built, skipping ingest"
  else
    step "building the kalshi corpus (rate limited, ~27 min cold)"
    ( cd "$RL" && "$PY" scripts/build_corpus.py --mode candles --step 60 \
        --cache "$ROOT/data/kalshi" --out "$ROOT/data/corpus" )
    step "joining binance spot (~170 MB of 1s klines)"
    ( cd "$RL" && "$PY" scripts/add_spot.py \
        --corpus "$ROOT/data/corpus/corpus_candles_60s.npz" \
        --cache "$ROOT/data/binance_klines" )
  fi

  step "validating the corpus end to end"
  ( cd "$RL" && "$PY" scripts/validate_corpus.py --corpus "$CORPUS" )

  # ---------------------------------------------------------- sanity checks
  step "sanity: does the learning loop work where the answer is known?"
  ( cd "$RL" && "$PY" scripts/sanity_tabular.py \
      --learn-episodes $([ "$QUICK" = 1 ] && echo 800 || echo 3000) \
      --null-episodes  $([ "$QUICK" = 1 ] && echo 4000 || echo 20000) )
  ( cd "$RL" && "$PY" scripts/sanity_ppo.py --updates $((UPDATES / 2)) )

  # ---------------------------------------------------------------- training
  step "training ppo, $SEEDS seeds"
  ( cd "$RL" && "$PY" scripts/train_ppo.py --corpus "$CORPUS" \
      --seeds "$SEEDS" --updates "$UPDATES" --entropy-coef 0.01 --out runs/ppo )
  step "training the zero-cost ablation"
  ( cd "$RL" && "$PY" scripts/train_ppo.py --corpus "$CORPUS" \
      --seeds "$SEEDS" --updates "$UPDATES" --entropy-coef 0.01 \
      --frictionless --out runs/ppo_nofric )

  # -------------------------------------------------------------- evaluation
  step "evaluating on the held-out test split (touched once)"
  ( cd "$RL" && "$PY" scripts/evaluate.py --corpus "$CORPUS" \
      --runs runs/ppo --runs-nofric runs/ppo_nofric --out "$REPORTS" )
fi

# ------------------------------------------------------------ explainability
step "shapley explanations, with the ground-truth validation"
( cd "$RL" && "$PY" scripts/explain.py --corpus "$CORPUS" --runs runs/ppo \
    --out "$REPORTS" --updates "$EXPLAIN_UPDATES" )

step "faithfulness: the decoy experiment and deletion curves"
( cd "$RL" && "$PY" scripts/faithfulness.py --out "$REPORTS" \
    --updates "$EXPLAIN_UPDATES" )

step "attribution stability across seeds"
( cd "$RL" && "$PY" scripts/stability.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" )

# ---------------------------------------------------------- the null-model test
step "null-model test: is the explanation distinguishable from nothing?"
( cd "$RL" && "$PY" scripts/sanity_check_explanations.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" --n-null "$N_NULL" )

step "power curve, and estimation certainty vs validity"
( cd "$RL" && "$PY" scripts/power_curve.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" )

step "can an explanation be steered at fixed performance?"
( cd "$RL" && "$PY" scripts/steer_explanation.py --corpus "$CORPUS" \
    --out "$REPORTS" --seeds "$STEER_SEEDS" --updates "$EXPLAIN_UPDATES" )

step "a second attribution family, and per-feature values across all seeds"
( cd "$RL" && "$PY" scripts/second_method.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" --n-null "$N_NULL" \
    --updates "$EXPLAIN_UPDATES" )

step "does the verdict depend on the credit-assignment scheme?"
( cd "$RL" && "$PY" scripts/scheme_robustness.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 16) )

step "horizon scaling: why does the weight null fail?"
( cd "$RL" && "$PY" scripts/horizon_scaling.py --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 16) )

step "is the span an artefact of off-manifold masking?"
( cd "$RL" && "$PY" scripts/manifold_masking.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 12) )

step "does it generalise? CartPole and Acrobot"
( cd "$RL" && "$PY" scripts/generalize_gym.py --out "$REPORTS" \
    --steps "$GYM_STEPS" --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 12) )

# -------------------------------------------------------------------- done
step "redrawing the paper's figures at publication size"
( cd "$RL" && "$PY" scripts/paper_figures.py --reports "$REPORTS" \
    --out "$ROOT/docs/paper/figures" )

step "verifying every number in docs/paper/main.md against the artifacts"
( cd "$RL" && "$PY" scripts/verify_paper_numbers.py )

# the paper states a wall-clock figure. record the real one rather than leaving
# it to memory of an earlier run.
PIPELINE_SECONDS=$(( $(date +%s) - PIPELINE_START ))
cat > "$REPORTS/pipeline_timing.json" <<JSON
{
  "seconds": $PIPELINE_SECONDS,
  "minutes": $(( PIPELINE_SECONDS / 60 )),
  "quick": $QUICK,
  "explain_only": $EXPLAIN_ONLY,
  "cpu": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)",
  "cores": $(sysctl -n hw.ncpu 2>/dev/null || echo 0)
}
JSON

step "done in $(( PIPELINE_SECONDS / 60 ))m $(( PIPELINE_SECONDS % 60 ))s"
echo "figures and json in $REPORTS:"
ls -1 "$REPORTS"
echo
echo "docs/PAPER.md   the interpretability findings"
echo "docs/REPORT.md  the trading project"

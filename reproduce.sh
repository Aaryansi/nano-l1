#!/usr/bin/env bash
#
# regenerate every figure and number in docs/paper/ and docs/REPORT.md from a
# clean checkout.
#
# no api keys and no accounts. every data source is public:
#   kalshi     api.elections.kalshi.com   read-only, unauthenticated
#   binance    data.binance.vision        static daily files
#   gymnasium  four classic-control tasks  installed with the deps
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

# the same test with every null matched to its own corpus, which is what
# section 3 actually defines. the step above is kept because the paper reports
# what was tried before as well as what it settled on.
step "the null-model test again, with every null matched to its own corpus"
( cd "$RL" && "$PY" scripts/matched_null_test.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" --n-null "$N_NULL" )

# the third construction: permute the outcomes and leave everything else, so
# the null agents stay sighted and the corpus stays fixed. this is the one the
# paper adopts; the two above are what it rejected on the way.
# the stratified variant, and whether any bucket width makes it usable. costs
# seconds because it trains nothing: if no window exists there is nothing to
# run a null test on.
# does the null-construction failure hold across SVERL's three explanatory
# targets, or only for the outcome-level statistic the rest of the paper uses?
step "the three SVERL targets against both null constructions"
( cd "$RL" && "$PY" scripts/sverl_targets.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 16) )

step "is there a stratified permutation that works?"
( cd "$RL" && "$PY" scripts/stratified_sweep.py --corpus "$CORPUS" \
    --out "$REPORTS" )

step "what the outcome permutation actually removes"
( cd "$RL" && "$PY" scripts/permutation_calibration.py --corpus "$CORPUS" \
    --out "$REPORTS" )

step "the outcome-permutation null: sighted agents, fixed corpus"
( cd "$RL" && "$PY" scripts/permuted_null_test.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" --n-null "$N_NULL" )

# the matched null's width depends on how converged its blind agents are, so
# the margin is measured against the budget rather than assumed independent.
step "is the matched null's width a property of the task or of the budget?"
( cd "$RL" && "$PY" scripts/null_budget_check.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 12) \
    --budgets $([ "$QUICK" = 1 ] && echo "10 20" || echo "20 40 80 160") )

# the headline null is 24 draws. rerun at 99 to show the verdicts do not move,
# written to a separate artifact so the paper's budgeted numbers stay put.
step "does the headline survive a four-times larger null?"
( cd "$RL" && "$PY" scripts/sanity_check_explanations.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS/n99" \
    --n-null $([ "$QUICK" = 1 ] && echo 12 || echo 99) )
cp "$REPORTS/n99/sanity_test.json" "$REPORTS/sanity_test_n99.json"

step "power curve, and estimation certainty vs validity"
( cd "$RL" && "$PY" scripts/power_curve.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" )

step "can an explanation be steered at fixed performance?"
( cd "$RL" && "$PY" scripts/steer_explanation.py --corpus "$CORPUS" \
    --out "$REPORTS" --seeds "$STEER_SEEDS" --updates "$EXPLAIN_UPDATES" )

# steering holds return fixed, which is weaker than holding behaviour fixed.
# measured rather than asserted, because the paper's claim depends on which.
step "are the steered and unsteered agents the same policy?"
( cd "$RL" && "$PY" scripts/behavioural_equivalence.py --corpus "$CORPUS" \
    --out "$REPORTS" --seeds "$STEER_SEEDS" --updates "$EXPLAIN_UPDATES" \
    --n-states $([ "$QUICK" = 1 ] && echo 512 || echo 4096) )

# the canonical adebayo check, as distinct from using parameter randomisation
# to generate reference draws. the two answer different questions and the
# paper needs to have run the one it discusses.
step "the canonical parameter-randomization sanity check"
( cd "$RL" && "$PY" scripts/parameter_randomization.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" --updates "$EXPLAIN_UPDATES" \
    --seeds $([ "$QUICK" = 1 ] && echo 2 || echo 5) \
    --n-states $([ "$QUICK" = 1 ] && echo 8 || echo 25) \
    --outcome-seeds $([ "$QUICK" = 1 ] && echo 1 || echo 3) \
    --outcome-coalitions $([ "$QUICK" = 1 ] && echo 16 || echo 64) \
    --outcome-episodes $([ "$QUICK" = 1 ] && echo 40 || echo 150) )

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

step "positive control: does the test fire on a learnable REAL task?"
( cd "$RL" && "$PY" scripts/positive_control.py --corpus "$CORPUS" \
    --out "$REPORTS" --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 12) )

# the paper's headline verdict turned out to depend on this, so it is measured
# rather than assumed. same agent, same measurement corpus, two null
# constructions.
step "which null construction does the market verdict depend on?"
( cd "$RL" && "$PY" scripts/null_corpus_check.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 32) )

step "is the span an artefact of off-manifold masking?"
( cd "$RL" && "$PY" scripts/manifold_masking.py --corpus "$CORPUS" \
    --runs runs/ppo --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 12) )

# four classic-control tasks, not two. box2d environments are deliberately not
# used: they need a system swig binary to build, which would trade the
# one-command property for one more feature dimension.
step "does it generalise? four control tasks"
( cd "$RL" && "$PY" scripts/generalize_gym.py --out "$REPORTS" \
    --envs CartPole-v1 Acrobot-v1 MountainCar-v0 Pendulum-v1 \
    --steps "$GYM_STEPS" --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 12) )

# -------------------------------------------------------------------- done
step "confidence intervals on every reported z-score"
( cd "$RL" && "$PY" scripts/bootstrap_z.py --reports "$REPORTS" \
    --out "$REPORTS" )

step "redrawing the paper's figures at publication size"
( cd "$RL" && "$PY" scripts/paper_figures.py --reports "$REPORTS" \
    --out "$ROOT/docs/paper/figures" )

step "verifying every number in docs/paper/main.tex against the artifacts"
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
echo "docs/paper/main.pdf   the paper"
echo "docs/REPORT.md  the trading project"

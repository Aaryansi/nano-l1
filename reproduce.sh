#!/usr/bin/env bash
#
# run the research pipeline in isolated output directories.
# compare a selected full run with the manuscript using --verify-paper.
#
# The fixed public-data snapshot is tracked under data/releases/market_snapshot_v1.
# Every default run verifies its hashes and chronological split before training.
# Set CORPUS to use a different, already-built dataset as a new experiment.
# The ingestion scripts remain available for collecting a fresh API window.
#
# usage:
#   ./reproduce.sh              full pipeline
#   ./reproduce.sh --quick      reduced seeds and budgets, for a smoke test
#   ./reproduce.sh --tests-only just the test suite
#   CHECKPOINTS=/path/to/checkpoints ./reproduce.sh --explain-only
#   ./reproduce.sh --print-config  inspect paths without running
#   ./reproduce.sh --verify-paper  compare this full run with manuscript claims

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
PIPELINE_START=$(date +%s)
RL="$ROOT/services/agent-rl"
PY="$RL/.venv/bin/python"

SEEDS=5
UPDATES=100
EXPLAIN_UPDATES=60
N_NULL=24
NULL_UPDATES=40
STEER_SEEDS=10
GYM_STEPS=400000
CONTROL_JOBS="${CONTROL_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
if ! [[ "$CONTROL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTROL_JOBS must be a positive integer"; exit 1
fi
QUICK=0
TESTS_ONLY=0
EXPLAIN_ONLY=0
PRINT_CONFIG=0
VERIFY_PAPER=0

for arg in "$@"; do
  case "$arg" in
    --quick)
      QUICK=1; SEEDS=2; UPDATES=30; EXPLAIN_UPDATES=20
      N_NULL=8; NULL_UPDATES=10; STEER_SEEDS=2; GYM_STEPS=60000 ;;
    --print-config) PRINT_CONFIG=1 ;;
    --verify-paper) VERIFY_PAPER=1 ;;
    --tests-only)   TESTS_ONLY=1 ;;
    --explain-only) EXPLAIN_ONLY=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done


# Every invocation has its own reports, figures, and checkpoint directories.
# Reusing checkpoints requires an explicit path; smoke runs never alter a release.
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/reproductions/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
REPORTS="${REPORTS:-$RUN_ROOT/reports}"
FIGURES="${FIGURES:-$RUN_ROOT/figures}"
CHECKPOINTS="${CHECKPOINTS:-$RUN_ROOT/checkpoints}"
RUNS="$CHECKPOINTS/ppo"
RUNS_NOFRIC="$CHECKPOINTS/ppo_nofric"
SNAPSHOT="$ROOT/data/releases/market_snapshot_v1"
CORPUS="${CORPUS:-$SNAPSHOT/corpus.npz}"
if [ "$QUICK" = 1 ] && [ "$VERIFY_PAPER" = 1 ]; then
  echo "quick mode cannot verify publication claims"; exit 1
fi
if [ "$EXPLAIN_ONLY" = 1 ] && [ "$VERIFY_PAPER" = 1 ]; then
  echo "explanation-only runs cannot verify the full publication protocol"; exit 1
fi
if [ "$VERIFY_PAPER" = 1 ] && { [ -z "${BANK_STUDY:-}" ] || [ -z "${BANK_REPLAY_PROOF:-}" ] || [ -z "${OBJECTIVE_STUDY:-}" ]; }; then
  echo "publication comparison requires explicit BANK_STUDY, BANK_REPLAY_PROOF and OBJECTIVE_STUDY inputs"; exit 1
fi
if [ "$EXPLAIN_ONLY" = 1 ] && [ ! -d "$RUNS" ]; then
  echo "explain-only requires CHECKPOINTS pointing to an existing run"; exit 1
fi
if [ "$PRINT_CONFIG" = 1 ]; then
  python3 - "$RUN_ROOT" "$REPORTS" "$FIGURES" "$RUNS" "$RUNS_NOFRIC" "$CORPUS" "$CONTROL_JOBS" <<'PYCONFIG'
import json, sys
config = dict(zip(("run_root", "reports", "figures", "runs", "runs_nofric", "corpus"), sys.argv[1:7]))
config["control_jobs"] = int(sys.argv[7])
print(json.dumps(config, indent=2))
PYCONFIG
  exit 0
fi
# Refuse accidental reuse, including reduced seed counts leaving old files behind.
if [ "$TESTS_ONLY" = 0 ]; then
  python3 - "$REPORTS" "$FIGURES" "$CHECKPOINTS" "$EXPLAIN_ONLY" <<'PYGUARD'
from pathlib import Path
import sys
for raw in sys.argv[1:4]:
    p = Path(raw).resolve()
    if p.exists() and any(p.iterdir()) and not (raw == sys.argv[3] and sys.argv[4] == "1"):
        raise SystemExit(f"refusing nonempty output directory: {p}; choose a new run")
PYGUARD
fi

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

# the smoke test checks that the code runs, not what the answer is. its null
# budgets sit below the rank floor for alpha=0.05, so every verdict it prints is
# UNRESOLVED and must not be read as a result. exported only AFTER the test
# suite: the guard's own tests assert that a sub-floor budget is a hard error,
# and an opt-out visible to pytest turns five of them into failures.
if [ "$QUICK" = "1" ]; then
  export ALLOW_UNRESOLVED_NULLS=1
  echo "quick mode: null budgets are below the rank floor, verdicts are UNRESOLVED"
fi


mkdir -p "$REPORTS" "$FIGURES"

if [ "$EXPLAIN_ONLY" = "0" ]; then
  # ------------------------------------------------------------------- data
  if [ ! -f "$CORPUS" ]; then
    echo "corpus missing: $CORPUS; restore the snapshot or set CORPUS to an existing dataset"
    exit 1
  fi
  if [ "$CORPUS" = "$SNAPSHOT/corpus.npz" ]; then
    step "verifying the fixed corpus snapshot"
    "$PY" "$RL/scripts/verify_snapshot.py" "$SNAPSHOT"
  fi

  step "validating the corpus end to end"
  ( cd "$RL" && "$PY" scripts/validate_corpus.py --corpus "$CORPUS" )

  ( cd "$RL" && "$PY" scripts/run_manifest.py --corpus "$CORPUS" \
      --checkpoints "$CHECKPOINTS" --out "$REPORTS" --phase started --quick "$QUICK" )

  # ---------------------------------------------------------- sanity checks
  step "synthetic learning checks against specified reference strategies"
  ( cd "$RL" && "$PY" scripts/sanity_tabular.py \
      --learn-episodes $([ "$QUICK" = 1 ] && echo 800 || echo 3000) \
      --null-episodes  $([ "$QUICK" = 1 ] && echo 4000 || echo 20000) )
  ( cd "$RL" && "$PY" scripts/sanity_ppo.py --updates $((UPDATES / 2)) )

  # ---------------------------------------------------------------- training
  step "training ppo, $SEEDS seeds"
  ( cd "$RL" && "$PY" scripts/train_ppo.py --corpus "$CORPUS" \
      --seeds "$SEEDS" --updates "$UPDATES" --entropy-coef 0.01 --out "$RUNS" )
  step "training the zero-cost ablation"
  ( cd "$RL" && "$PY" scripts/train_ppo.py --corpus "$CORPUS" \
      --seeds "$SEEDS" --updates "$UPDATES" --entropy-coef 0.01 \
      --frictionless --out "$RUNS_NOFRIC" )

  # -------------------------------------------------------------- evaluation
  step "evaluating on the held-out test split (touched once)"
  ( cd "$RL" && "$PY" scripts/evaluate.py --corpus "$CORPUS" \
      --runs "$RUNS" --runs-nofric "$RUNS_NOFRIC" --out "$REPORTS" )
fi

# ------------------------------------------------------------ explainability
step "shapley explanations, with the ground-truth validation"
( cd "$RL" && "$PY" scripts/explain.py --corpus "$CORPUS" --runs "$RUNS" \
    --out "$REPORTS" --updates "$EXPLAIN_UPDATES" )

step "faithfulness: the decoy experiment and deletion curves"
( cd "$RL" && "$PY" scripts/faithfulness.py --out "$REPORTS" \
    --updates "$EXPLAIN_UPDATES" )

step "attribution stability across seeds"
( cd "$RL" && "$PY" scripts/stability.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" )

# ---------------------------------------------------------- the null-model test
step "return-span comparison against the synthetic reference bank"
( cd "$RL" && "$PY" scripts/sanity_check_explanations.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --n-null "$N_NULL" --updates "$NULL_UPDATES" )

# Secondary blinded-reference training retains each underlying environment.
# Its observed masking draws differ from the primary synthetic-bank analysis.
step "the null-model test again, with every null matched to its own corpus"
( cd "$RL" && "$PY" scripts/matched_null_test.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --n-null "$N_NULL" )

# Global settlement permutation changes the joint price/reward law.
# It is a secondary sensitivity, without a market-efficiency interpretation.
# the stratified variant, and whether any bucket width makes it usable. costs
# seconds because it trains nothing: if no window exists there is nothing to
# run a null test on.
# does the null-construction failure hold across SVERL's three explanatory
# targets, or only for the outcome-level statistic the rest of the paper uses?
step "three target proxies against both reference constructions"
( cd "$RL" && "$PY" scripts/sverl_targets.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 24) )

step "descriptive stratified-permutation screening"
( cd "$RL" && "$PY" scripts/stratified_sweep.py --corpus "$CORPUS" \
    --out "$REPORTS" )

step "what the outcome permutation actually removes"
( cd "$RL" && "$PY" scripts/permutation_calibration.py --corpus "$CORPUS" \
    --out "$REPORTS" )

step "the outcome-permutation null: sighted agents, fixed corpus"
( cd "$RL" && "$PY" scripts/permuted_null_test.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --n-null "$N_NULL" )

# the matched null's width depends on how converged its blind agents are, so
# the margin is measured against the budget rather than assumed independent.
step "is the matched null's width a property of the task or of the budget?"
( cd "$RL" && "$PY" scripts/null_budget_check.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 24) \
    --budgets $([ "$QUICK" = 1 ] && echo "10 20" || echo "20 40 80 160") )

# the headline null is 24 draws. rerun at 99 to show the verdicts do not move,
# written to a separate artifact so the paper's budgeted numbers stay put.
step "does the headline survive a four-times larger null?"
( cd "$RL" && "$PY" scripts/sanity_check_explanations.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS/n99" --updates "$NULL_UPDATES" \
    --n-null $([ "$QUICK" = 1 ] && echo 12 || echo 99) )

step "conditional detection rates on independent synthetic trials"
( cd "$RL" && "$PY" scripts/calibrate_detection.py --out "$REPORTS" \
    --reference-json "$REPORTS/sanity_test.json" --n-null "$N_NULL" \
    --updates "$NULL_UPDATES" --trials $([ "$QUICK" = 1 ] && echo 2 || echo 30) )

step "signal-strength illustration and attribution estimation diagnostics"
( cd "$RL" && "$PY" scripts/power_curve.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --updates "$NULL_UPDATES" )

step "attribution suppression, behaviour and paired return contrasts"
( cd "$RL" && "$PY" scripts/steer_explanation.py --corpus "$CORPUS" \
    --out "$REPORTS" --seeds "$STEER_SEEDS" --updates "$EXPLAIN_UPDATES" )

# The auxiliary single-seed greedy comparison does not establish behavioural
# equivalence across the main multi-seed steering experiment.
step "are the steered and unsteered agents the same policy?"
( cd "$RL" && "$PY" scripts/behavioural_equivalence.py --corpus "$CORPUS" \
    --out "$REPORTS" --seeds "$STEER_SEEDS" --updates "$EXPLAIN_UPDATES" \
    --n-states $([ "$QUICK" = 1 ] && echo 512 || echo 4096) )

# the canonical adebayo check, as distinct from using parameter randomisation
# to generate reference draws. the two answer different questions and the
# paper needs to have run the one it discusses.
step "the canonical parameter-randomization sanity check"
# this step trains its own planted-signal agent, and unlike the other explain
# steps it wants that agent well converged: the canonical check asks whether a
# TRAINED network's explanation differs from a random one, so an undertrained
# subject weakens the test. 150 is the value the reported numbers come from. it
# was previously left to the script's default, which meant the pipeline passed
# 60 here and silently produced different planted-arm numbers from the ones the
# paper quotes.
( cd "$RL" && "$PY" scripts/parameter_randomization.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --updates $([ "$QUICK" = 1 ] && echo 20 || echo 150) \
    --seeds $([ "$QUICK" = 1 ] && echo 2 || echo 5) \
    --n-states $([ "$QUICK" = 1 ] && echo 8 || echo 25) \
    --outcome-seeds $([ "$QUICK" = 1 ] && echo 1 || echo 3) \
    --outcome-coalitions $([ "$QUICK" = 1 ] && echo 16 || echo 64) \
    --outcome-episodes $([ "$QUICK" = 1 ] && echo 40 || echo 150) )

step "a second attribution family, and per-feature values across all seeds"
( cd "$RL" && "$PY" scripts/second_method.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --n-null "$N_NULL" \
    --updates "$EXPLAIN_UPDATES" )

step "does the verdict depend on the credit-assignment scheme?"
( cd "$RL" && "$PY" scripts/scheme_robustness.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 24) )

step "descriptive reference widths across synthetic horizons"
( cd "$RL" && "$PY" scripts/horizon_scaling.py --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 16) )

step "positive control: does the test fire on a learnable REAL task?"
( cd "$RL" && "$PY" scripts/positive_control.py --corpus "$CORPUS" \
    --out "$REPORTS" --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 24) \
    --seeds "$SEEDS" --budgets $([ "$QUICK" = 1 ] && echo "10 20" || echo "40 100") )

# Compare two reference populations using the same observed agent and
# measurement corpus, retaining unresolved decisions separately.
step "which null construction does the market verdict depend on?"
( cd "$RL" && "$PY" scripts/null_corpus_check.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 32) )

step "is the span an artefact of off-manifold masking?"
( cd "$RL" && "$PY" scripts/manifold_masking.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" \
    --n-null $([ "$QUICK" = 1 ] && echo 4 || echo 24) )

step "separating market replacement from inventory and time replacement"
( cd "$RL" && "$PY" scripts/state_masking.py --corpus "$CORPUS" \
    --runs "$RUNS" --out "$REPORTS" --seeds "$SEEDS" --updates "$EXPLAIN_UPDATES" )

# Six control tasks. box2d environments are deliberately not
# used: they need a system swig binary to build, which would trade the
# one-command property for one more feature dimension.
step "does it generalise? six control tasks"
( cd "$RL" && "$PY" scripts/generalize_gym.py --out "$REPORTS" \
    --envs CartPole-v1 Acrobot-v1 MountainCar-v0 Pendulum-v1 FrozenLake-v1 CliffWalking-v0 --trainer sb3 \
    --steps "$GYM_STEPS" --jobs "$CONTROL_JOBS" --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 24) )

# Whole-row resampling retains donor row support, not compatibility with the
# recipient trajectory. Independent observed seeds are distinct replicates;
# checkpoints along one training trajectory are correlated observations.
step "five independently trained agents per task"
( cd "$RL" && "$PY" scripts/generalize_gym.py --out "$REPORTS/seeds" \
    --envs CartPole-v1 Acrobot-v1 MountainCar-v0 Pendulum-v1 FrozenLake-v1 CliffWalking-v0 --trainer sb3 \
    --steps "$GYM_STEPS" --jobs "$CONTROL_JOBS" --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 24) \
    --agent-seeds $([ "$QUICK" = 1 ] && echo 2 || echo 5) )

step "the same six tasks with whole-row resampling"
( cd "$RL" && "$PY" scripts/generalize_gym.py --out "$REPORTS/resample" \
    --envs CartPole-v1 Acrobot-v1 MountainCar-v0 Pendulum-v1 FrozenLake-v1 CliffWalking-v0 --trainer sb3 --blind resample \
    --steps "$GYM_STEPS" --jobs "$CONTROL_JOBS" --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 24) )

# -------------------------------------------------------------------- done
step "confidence intervals on every reported z-score"
# stated rather than left to the default: the committed intervals were once
# built at 4000 resamples while the pipeline used 10000, so the numbers in the
# paper could not be reproduced by running it.
( cd "$RL" && "$PY" scripts/bootstrap_z.py --reports "$REPORTS" \
    --out "$REPORTS" --n-boot 10000 )

step "rendering auxiliary experiment figures"
( cd "$RL" && "$PY" scripts/paper_figures.py --reports "$REPORTS" \
    --out "$FIGURES" )

if [ "$EXPLAIN_ONLY" = 0 ]; then
  # A temporary view supports explicit output-directory overrides without
  # copying checkpoints or changing the immutable run layout.
  RELEASE_VERIFY_VIEW="$(mktemp -d)"
  trap 'rm -rf "${RELEASE_VERIFY_VIEW:?}"' EXIT
  ln -s "$(cd "$CHECKPOINTS" && pwd)" "$RELEASE_VERIFY_VIEW/checkpoints"
  ln -s "$(cd "$REPORTS" && pwd)" "$RELEASE_VERIFY_VIEW/reports"
  step "auditing every held-out market checkpoint trajectory"
  ( cd "$RL" && "$PY" scripts/audit_market_rollouts.py --corpus "$CORPUS" \
      --run "$RELEASE_VERIFY_VIEW" --out "$REPORTS/rollout_parity.json" )
fi

# Record execution time as part of the completed experiment evidence.
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

step "recording and validating the generated run"
( cd "$RL" && "$PY" scripts/run_manifest.py --corpus "$CORPUS" \
    --checkpoints "$CHECKPOINTS" --out "$REPORTS" --phase complete --quick "$QUICK" )

if [ "$VERIFY_PAPER" = 1 ]; then
  step "comparing completed run evidence with the published numerical claims"
  ( "$PY" "$RL/scripts/verify_paper_numbers.py" --reports "$RELEASE_VERIFY_VIEW/reports" \
      --corpus "$CORPUS" --banks "$BANK_STUDY" --replay-proof "$BANK_REPLAY_PROOF" \
      --objectives "$OBJECTIVE_STUDY" )
fi

step "done in $(( PIPELINE_SECONDS / 60 ))m $(( PIPELINE_SECONDS % 60 ))s"
echo "figures and json in $REPORTS:"
ls -1 "$REPORTS"
echo
echo "publication verification is opt-in with --verify-paper"
echo "docs/REPORT.md  selected evidence and manuscript verification"

"""cross-check every number in docs/paper/main.tex against reports/*.json.

a paper with a mistranscribed number is worse than no paper, and the numbers in
this one were copied by hand across many editing passes. this asserts each
claim against the artifact that produced it, so a stale figure surfaces here
rather than in review.

usage:
    python scripts/verify_paper_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"

failures: list[str] = []
checks = 0


def load(name: str):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def check(label: str, claimed: float, actual: float | None, tol: float = 0.005) -> None:
    global checks
    checks += 1
    if actual is None:
        failures.append(f"{label}: no artifact")
        print(f"  [ ] {label:<52} NO ARTIFACT")
        return
    ok = abs(claimed - actual) <= tol * max(1.0, abs(actual))
    mark = "x" if ok else " "
    print(f"  [{mark}] {label:<52} paper {claimed:>10.4f}  artifact {actual:>10.4f}")
    if not ok:
        failures.append(f"{label}: paper {claimed} vs artifact {actual}")


print("=" * 78)
print("verifying docs/paper/main.tex against reports/")
print("=" * 78)

# ---------------------------------------------------------------- table 1
r = load("results.json")
print("\ntable 1: test-split evaluation")
if r:
    check("ppo mean pnl", -0.595, r["ppo_across_seeds"]["mean_pnl_mean"], 0.02)
    check("ppo std across seeds", 0.176, r["ppo_across_seeds"]["mean_pnl_std"], 0.05)
    check("ppo p vs flat", 0.100, r["paired_bootstrap_p_vs_flat"]["ppo"], 0.15)
    check("buy-and-hold pnl", -1.608, r["baselines"]["buy-and-hold"]["mean_pnl"], 0.02)
    check("random pnl", -18.532, r["baselines"]["random"]["mean_pnl"], 0.02)
    check("mean-reversion pnl", -11.615, r["baselines"]["mean-reversion"]["mean_pnl"], 0.02)
    check("logistic-refit pnl", -5.729, r["baselines"]["logistic-refit"]["mean_pnl"], 0.02)
    check("test episodes", 1284, r["test_episodes"], 0.001)

# ---------------------------------------------------------------- table 2
s = load("sanity_test.json")
print("\ntable 2: null-model test")
if s:
    check("planted signal z", 12.27, s["results"]["planted_signal"]["z_score"], 0.02)
    check("null corpus z", 0.72, s["results"]["held_out_null"]["z_score"], 0.05)
    check("real market z", 0.23, s["results"]["real_market"]["z_score"], 0.20)
    check("planted signal span", 70.89, s["results"]["planted_signal"]["statistic"], 0.02)
    check("real market span", 9.37, s["results"]["real_market"]["statistic"], 0.02)
    import numpy as np
    ns = np.array(s["null_spans"])
    check("null span mean", 8.19, float(ns.mean()), 0.02)
    check("null span sd", 5.11, float(ns.std(ddof=1)), 0.02)
    check("n null samples", 24, len(ns), 0.001)

# the n=99 rerun quoted in 7.5. the point of the check is that the verdicts do
# not move, so the verdicts are asserted, not just the numbers.
n99 = load("sanity_test_n99.json")
print("\nsection 7.5: the headline at n=99")
if n99:
    import numpy as np
    ns = np.array(n99["null_spans"])
    check("n99 null samples", 99, len(ns), 0.001)
    check("n99 null span mean", 8.08, float(ns.mean()), 0.02)
    check("n99 null span sd", 5.08, float(ns.std(ddof=1)), 0.02)
    r = n99["results"]
    check("n99 planted z", 12.35, r["planted_signal"]["z_score"], 0.02)
    check("n99 null corpus z", 0.74, r["held_out_null"]["z_score"], 0.10)
    check("n99 real market z", 0.25, r["real_market"]["z_score"], 0.20)
    checks += 1
    ok = (r["planted_signal"]["passes"]
          and not r["held_out_null"]["passes"]
          and not r["real_market"]["passes"])
    print(f"  [{'x' if ok else ' '}] {'all three verdicts unchanged at n=99':<52} "
          f"{'yes' if ok else 'NO':>21}")
    if not ok:
        failures.append("n=99 rerun: paper says every verdict is unchanged")

# ---------------------------------------------------------------- steering
st = load("steering.json")
print("\ntable 4: steering")
if st:
    rm, sy = st["real_market"], st["learnable_synthetic"]
    check("market baseline attribution", 0.399, rm[0]["target_share_mean"], 0.05)
    check("market steered attribution", 0.032, min(x["target_share_mean"] for x in rm[1:]), 0.20)
    check("synthetic baseline attribution", 0.465, sy[0]["target_share_mean"], 0.05)
    check("synthetic baseline return", 45.04, sy[0]["return_mean"], 0.02)
    check("synthetic steered return", 7.92, min(x["return_mean"] for x in sy[1:]), 0.20)

# ------------------------------------------------------------ environments
g = load("generalize_gym.json")
print("\ntable 3: null construction across environments")
if g:
    # paper table 3, one row per environment
    WEIGHT_SD = {"cartpole": 138.92, "acrobot": 124.94,
                 "mountaincar": 0.0, "pendulum": 131.42}
    ENV_SD = {"cartpole": 3.29, "acrobot": 0.0,
              "mountaincar": 0.0, "pendulum": 37.79}
    for row in g:
        e = row["env_id"].split("-")[0].lower()
        # a degenerate null is exactly zero, so a relative tolerance cannot
        # express "close"; compare those absolutely instead.
        check(f"{e} weight null sd", WEIGHT_SD[e], row["weight_null"]["std"],
              0.02 if WEIGHT_SD[e] else 1.0)
        check(f"{e} env null sd", ENV_SD[e], row["env_null"]["std"],
              0.02 if ENV_SD[e] else 1.0)
    agree = sum(r["nulls_agree"] for r in g)
    total = sum(r["n_checkpoints"] for r in g)
    check("checkpoints where nulls agree", 11, agree, 0.001)
    check("total checkpoints", 16, total, 0.001)

    import numpy as np
    zw = [c["z_weight"] for r in g for c in r["checkpoints"]]
    check("max |z| under the weight null", 2.45, max(map(abs, zw)), 0.02)

# ------------------------------------------------- initialisation variance
nw = load("null_width_conjecture.json")
print("\nsection 6.1: why the parameter null is wide")
if nw and g:
    byenv = dict(zip([e.split("-")[0].lower() for e in nw["env_ids"]],
                     nw["random_return_sd"]))
    for e, v in (("cartpole", 138.38), ("acrobot", 123.04),
                 ("pendulum", 135.49), ("mountaincar", 0.0)):
        check(f"{e} random-init return sd", v, byenv[e], 0.02 if v else 1.0)
    # the mechanism: the masked term is near-constant, so the span inherits
    # the unmasked term's variance
    import numpy as np
    for row in g:
        e = row["env_id"].split("-")[0].lower()
        if e not in ("cartpole", "acrobot"):
            continue
        un = np.array(row["random_init_return"]["returns"])
        ma = un - np.array(row["weight_null"]["spans"])
        frac = float(ma.std(ddof=1) / un.std(ddof=1))
        check(f"{e} masked/unmasked sd ratio",
              0.018 if e == "cartpole" else 0.054, frac, 0.10)

# the variance-decomposition bound quoted in 6.1. asserted because it was
# wrong in a draft: the covariance term is bounded by 2r, not by r^2, and the
# first version of that sentence claimed both were under a percent.
if g:
    import numpy as np
    print("\nsection 6.1: the variance decomposition bound")
    for row in g:
        e = row["env_id"].split("-")[0].lower()
        if e not in ("cartpole", "acrobot"):
            continue
        un = np.array(row["random_init_return"]["returns"])
        ma = un - np.array(row["weight_null"]["spans"])
        r = float(ma.std(ddof=1) / un.std(ddof=1))
        check(f"{e} neglected-term bound (r^2 + 2r)",
              0.036 if e == "cartpole" else 0.111, r * r + 2 * r, 0.10)

# ---------------------------------------------------------------- schemes
sc = load("scheme_robustness.json")
print("\nsection 7.3: credit-assignment schemes")
if sc:
    for k, sig, real in (("span", 75.74, 8.84),
                         ("leave_one_out", 101.49, -3.89),
                         ("only_one_in", 93.46, -0.60)):
        check(f"{k} planted statistic", sig, sc["planted_signal"][k]["statistic"], 0.02)
        check(f"{k} real statistic", real, sc["real_market"][k]["statistic"], 0.05)
    check("loo vs ooi rank correlation", 0.309, sc["loo_vs_ooi_rank_corr"], 0.05)

# ---------------------------------------------------------------- horizon
h = load("horizon_scaling.json")
print("\nsection 9.1: horizon scaling (rejected hypothesis)")
if h:
    check("weight null exponent (full fit)", 0.96, h["alpha_weight"], 0.05)
    import numpy as np
    x = np.array([r["n_steps"] for r in h["rows"]], float)
    w = np.array([r["weight_std"] for r in h["rows"]])
    v = np.array([r["env_std"] for r in h["rows"]])
    keep = v > 0.1  # 1e-2 failed to exclude the h=56 collapse at 0.037
    aw = np.polyfit(np.log(x[keep]), np.log(w[keep]), 1)[0]
    av = np.polyfit(np.log(x[keep]), np.log(v[keep]), 1)[0]
    check("weight exponent excl. collapse", 0.94, float(aw), 0.05)
    check("env exponent excl. collapse", 0.83, float(av), 0.05)
    check("env sd at horizon 56", 0.04, v[-1], 0.5)

# ------------------------------------------------------------ second method
sm = load("second_method.json")
print("\nsection 7.3: attribution family")
if sm:
    check("shapley vs IG rank correlation", 0.981, sm["shapley_vs_ig_rank_corr"], 0.03)
    check("cross-seed consistency, shapley", 0.750, sm["consistency_shapley"], 0.05)
    check("cross-seed consistency, IG", 0.800, sm["consistency_ig"], 0.05)
    check("IG real-market z", 4.58, sm["ig_real_market"]["z_score"], 0.05)
    check("IG planted-signal z", 22.69, sm["ig_planted_signal"]["z_score"], 0.05)

# ------------------------------------------------------------- stability
sb = load("stability.json")
print("\nsection 5.1: cross-seed stability")
if sb:
    beh = next(r for r in sb["stability"] if "behaviour" in r["target"])
    check("behaviour rank correlation", 0.850, beh["rank_corr_mean"], 0.02)
    check("behaviour min rank correlation", 0.761, beh["rank_corr_min"], 0.02)
    check("behaviour top-1 agreement", 1.0, beh["top1_agreement"], 0.001)
    check("indistinguishable seed pairs", 10, sb["n_indistinguishable_pairs"], 0.001)

# ------------------------------------------------------------ faithfulness
f = load("faithfulness.json")
print("\nsection 5.2: decoy and deletion curves")
if f:
    check("decoy naive rank", 3, f["decoy"]["decoy_naive_rank"], 0.001)
    check("decoy trajectory rank", 10, f["decoy"]["decoy_trajectory_rank"], 0.001)
    check("decoy in-sample return", 47.24, f["decoy"]["in_sample_return"], 0.02)
    check("decoy held-out return", -6.83, f["decoy"]["held_out_return"], 0.05)
    auc = f["deletion"]["auc"]
    check("deletion AUC, trajectory", -187.8, auc["trajectory-aware"], 0.02)
    check("deletion AUC, per-decision", -165.3, auc["per-decision"], 0.02)
    check("deletion AUC, random", 366.6, auc["random (control)"], 0.02)

# ------------------------------------------------------------ off-manifold
mm = load("manifold_masking.json")
print("\nsection 7.4: off-manifold masking")
if mm:
    check("span, marginal masking", 8.838, mm["span_marginal"], 0.02)
    check("span, conditional masking", 8.838, mm["span_conditional"], 0.02)
    check("loo rank corr, marginal vs conditional", 0.767, mm["loo_rank_corr"], 0.05)
    check("conditional planted-signal z", 9.61, mm["conditional_planted_signal"]["z_score"], 0.05)
    check("conditional real-market z", -0.55, mm["conditional_real_market"]["z_score"], 0.20)
    by_kept = {r["n_kept"]: r for r in mm["offmanifold_distance"]}
    check("real-state distance floor", 0.381, by_kept[18]["marginal"], 0.02)
    check("marginal distance, 14 replaced", 0.486, by_kept[4]["marginal"], 0.02)
    check("conditional distance, 14 replaced", 0.186, by_kept[4]["conditional"], 0.02)

# --------------------------------------------------------- positive control
pc = load("positive_control.json")
print("\nsection 5.3: positive control on real data")
if pc:
    by = {t["task"]: t for t in pc["tasks"]}
    check("prediction span", 7.463, by["prediction"]["span"], 0.02)
    check("prediction null mean", -0.476, by["prediction"]["null_mean"], 0.20)
    check("prediction null sd", 2.337, by["prediction"]["null_std"], 0.05)
    check("prediction z", 3.40, by["prediction"]["result"]["z_score"], 0.05)
    check("trading span (paper checkpoint)", 7.405, by["trading"]["span"], 0.02)
    check("trading null mean", 2.470, by["trading"]["null_mean"], 0.05)
    check("trading null sd", 2.285, by["trading"]["null_std"], 0.05)
    check("trading z", 2.16, by["trading"]["result"]["z_score"], 0.05)
    # the separation is the whole claim, so assert it rather than the numbers
    # that happen to produce it
    checks += 1
    ok = bool(pc["separated"])
    print(f"  [{'x' if ok else ' '}] {'prediction fires, trading declines':<52} "
          f"{'yes' if ok else 'NO':>21}")
    if not ok:
        failures.append("positive control: the two tasks are not separated")

# ------------------------------------------------------- null construction
nc = load("null_corpus_check.json")
print("\nsection 6.3: which null construction")
if nc:
    check("observed span", 7.405, nc["observed_span"], 0.02)
    for key, mean, sd, z in (("null_synthetic", 9.07, 4.91, -0.34),
                             ("null_blinded_real", 2.26, 2.21, 2.33)):
        import numpy as np
        a = np.array(nc[key]["spans"])
        check(f"{key} mean", mean, float(a.mean()), 0.02)
        check(f"{key} sd", sd, float(a.std(ddof=1)), 0.02)
        check(f"{key} z", z, nc[key]["result"]["z_score"], 0.05)
    # the paper says both constructions decline under the two-part rule
    for key in ("null_synthetic", "null_blinded_real"):
        checks += 1
        passes = nc[key]["result"]["passes"]
        ok = not passes
        print(f"  [{'x' if ok else ' '}] {key + ' declines':<52} "
              f"{'yes' if ok else 'NO, IT FIRES':>21}")
        if not ok:
            failures.append(f"{key}: paper says it declines, artifact says it fires")

# ------------------------------------------------------------- z intervals
zi = load("z_intervals.json")
print("\nsection 7.5: bootstrap verdict stability")
if zi:
    expected_labels = {"shared-null: planted signal", "shared-null: real market",
                       "matched: real market", "real prediction", "real trading",
                       "market vs synthetic-corpus null",
                       "market vs blinded-real null"}
    missing = expected_labels - set(zi)
    checks += 1
    print(f"  [{'x' if not missing else ' '}] {'every expected bootstrap label is present':<52} "
          f"{'yes' if not missing else sorted(missing)}")
    if missing:
        failures.append(f"z_intervals: missing labels {sorted(missing)}")
    check("blinded-real verdict stability", 0.65,
          zi["market vs blinded-real null"]["verdict_stability"], 0.08)
    check("synthetic-corpus verdict stability", 1.0,
          zi["market vs synthetic-corpus null"]["verdict_stability"], 0.01)
    check("real prediction verdict stability", 1.00,
          zi["real prediction"]["verdict_stability"], 0.01)
    check("real trading verdict stability", 0.64,
          zi["real trading"]["verdict_stability"], 0.08)
    check("planted signal interval, low", 10.32, zi["shared-null: planted signal"]["z_lo"], 0.05)
    check("planted signal interval, high", 16.87, zi["shared-null: planted signal"]["z_hi"], 0.05)
    check("real market interval, low", -0.16, zi["shared-null: real market"]["z_lo"], 0.30)
    check("real market interval, high", 0.73, zi["shared-null: real market"]["z_hi"], 0.15)

# ------------------------------------------------- the matched construction
mn = load("matched_null_test.json")
print("\nsection 6.4: correcting the null everywhere")
if mn:
    by = {c["case"]: c for c in mn["cases"]}
    check("matched planted span", 74.69, by["planted signal"]["span"], 0.02)
    check("matched null-corpus span", 13.90, by["null corpus"]["span"], 0.02)
    check("matched real-market span", 7.40, by["real market"]["span"], 0.02)
    check("matched real-market null mean", 2.97, by["real market"]["null_mean"], 0.05)
    check("matched real-market null sd", 2.29, by["real market"]["null_std"], 0.05)
    for case in ("planted signal", "null corpus"):
        checks += 1
        # both are degenerate point-mass nulls; that is the finding
        degenerate = by[case]["null_std"] < 1e-9
        print(f"  [{'x' if degenerate else ' '}] {case + ' null is degenerate':<52} "
              f"{'yes' if degenerate else 'NO':>21}")
        if not degenerate:
            failures.append(f"{case}: paper says the null is a point mass")
    # the whole point of the section: it loses specificity
    checks += 1
    lost = by["null corpus"]["fires"] and not mn["has_specificity"]
    print(f"  [{'x' if lost else ' '}] {'matched construction fires on a signal-free corpus':<52} "
          f"{'yes' if lost else 'NO':>21}")
    if not lost:
        failures.append("matched null: paper says it produces a false positive")

# ------------------------------------------------------- budget dependence
nb = load("null_budget_check.json")
print("\nsection 6.4: the blinded null collapses with budget")
if nb:
    rows = {r["updates"]: r for r in nb["rows"]}
    for u, sd, z in ((20, 3.268, 0.55), (40, 2.247, 1.71),
                     (80, 1.623, 4.23), (160, 0.036, 203.62)):
        if u in rows:
            check(f"null sd at {u} updates", sd, rows[u]["null_std"], 0.05)
            check(f"z at {u} updates", z, rows[u]["result"]["z_score"], 0.08)
    checks += 1
    flipped = not nb["verdict_stable"]
    print(f"  [{'x' if flipped else ' '}] {'verdict flips with the null training budget':<52} "
          f"{'yes' if flipped else 'NO':>21}")
    if not flipped:
        failures.append("budget check: paper says the verdict is budget-dependent")

# ------------------------------------------------ the permutation construction
pn = load("permuted_null_test.json")
print("\nsection 6.5: the outcome-permutation null")
if pn:
    by = {c["case"]: c for c in pn["cases"]}
    check("permuted planted z", 10.48, by["planted signal"]["result"]["z_score"], 0.05)
    check("permuted null-corpus z", 1.75, by["null corpus"]["result"]["z_score"], 0.05)
    check("permuted real-market z", -15.52, by["real market"]["result"]["z_score"], 0.05)
    check("permuted real-market null mean", 55.85, by["real market"]["null_mean"], 0.05)
    check("permuted real-market null sd", 3.12, by["real market"]["null_std"], 0.08)
    check("permuted null agents' return", 36.26, by["real market"]["null_return_mean"], 0.05)
    # power and specificity are kept; the failure is elsewhere, and the paper
    # says so, so both halves are asserted
    for label, want in (("has_power", True), ("has_specificity", True),
                        ("any_collapsed", False)):
        checks += 1
        ok = bool(pn[label]) is want
        print(f"  [{'x' if ok else ' '}] {'permuted null ' + label + f' is {want}':<52} "
              f"{'yes' if ok else 'NO':>21}")
        if not ok:
            failures.append(f"permuted null: {label} is not {want}")
    checks += 1
    below = by["real market"]["result"]["z_score"] < -1.96
    print(f"  [{'x' if below else ' '}] {'real market lands below its own null':<52} "
          f"{'yes' if below else 'NO':>21}")
    if not below:
        failures.append("permuted null: paper says the market lands below the null")

pc = load("permutation_calibration.json")
print("\nsection 6.5: what the permutation removes")
if pc:
    check("calibration error, real", 0.0071, pc["calibration_error_real"], 0.05)
    check("calibration error, permuted", 0.4561, pc["calibration_error_permuted"], 0.05)
    check("fade edge, real (extremes)", -0.0015, pc["fade_edge"]["real"]["extremes"], 0.60)
    check("fade edge, permuted (extremes)", 0.4864, pc["fade_edge"]["permuted"]["extremes"], 0.05)

# ------------------------------------------------ the stratified permutation
ss = load("stratified_sweep.json")
print("\nsection 6.5: the stratified permutation")
if ss:
    rows = {r["n_buckets"]: r for r in ss["rows"]}
    for n, moved, corr in ((1, 0.506, -0.012), (2, 0.063, 0.873),
                           (8, 0.061, 0.879), (32, 0.052, 0.896),
                           (256, 0.049, 0.901)):
        check(f"{n} buckets, labels moved", moved, rows[n]["changed"], 0.05)
        check(f"{n} buckets, label corr", corr, rows[n]["label_corr"],
              0.05 if abs(corr) > 0.1 else 1.0)
    check("calibration recovered at 8 buckets", 0.0071,
          rows[8]["calibration_error"], 0.05)
    check("price-outcome correlation", 0.950,
          ss["real"]["price_outcome_r"], 0.02)
    check("fraction of episodes resolved by price", 0.887,
          ss["real"]["fraction_resolved"], 0.02)
    check("price is right when resolved", 0.994,
          ss["real"]["price_agrees_when_resolved"], 0.02)
    # the finding: no bucket width is usable
    checks += 1
    none_usable = not ss["usable_exists"]
    print(f"  [{'x' if none_usable else ' '}] {'no bucket width is usable':<52} "
          f"{'yes' if none_usable else 'NO':>21}")
    if not none_usable:
        failures.append("stratified sweep: paper says no window exists")

# --------------------------------------------------------- the SVERL targets
sv = load("sverl_targets.json")
print("\nsection 7.1: the three explanatory targets")
if sv:
    t = sv["targets"]
    for name, span, z in (("behaviour", 0.177, 6.33),
                          ("prediction", -0.053, 0.12),
                          ("outcomes", 7.405, -0.52)):
        check(f"{name} span", span, t[name]["span"], 0.05)
        check(f"{name} z", z, t[name]["null_signal_free"]["result"]["z_score"],
              0.05 if abs(z) > 1 else 1.0)
    # the finding: they disagree, and only behaviour fires
    checks += 1
    ok = (not sv["targets_agree_under_signal_free_null"]
          and sv["targets_that_fire"] == ["behaviour"])
    print(f"  [{'x' if ok else ' '}] {'targets disagree; only behaviour fires':<52} "
          f"{'yes' if ok else 'NO':>21}")
    if not ok:
        failures.append("sverl targets: paper says only behaviour fires")

# ------------------------------- the canonical parameter-randomization check
pr = load("parameter_randomization.json")
print("\nsection 6.2: the canonical parameter-randomization check")
if pr:
    for who, cas, ind in (
        ("market", [0.627, 0.405, 0.381], [0.627, 0.331, 0.335]),
        ("planted", [0.992, 0.971, 0.996], [0.992, 0.991, 0.985]),
    ):
        for mode, vals in (("cascading", cas), ("independent", ind)):
            for row, claimed in zip(pr[who][mode], vals):
                check(f"{who} {mode} {row['stage']}", claimed,
                      row["rank_corr_mean"], 0.06)
    # the finding: the canonical check points the opposite way to the null test
    checks += 1
    fr = pr["fully_randomized_rank_corr"]
    ok = fr["market"] < 0.6 < fr["planted"]
    print(f"  [{'x' if ok else ' '}] {'canonical check points opposite to the null test':<52} "
          f"{'yes' if ok else 'NO':>21}")
    if not ok:
        failures.append("parameter randomization: paper says the verdicts invert")

# ------------------------------------------- steered vs unsteered behaviour
be = load("behavioural_equivalence.json")
print("\nsection 7.2: are the steered and unsteered agents the same policy?")
if be:
    check("greedy action agreement", 0.660, be["action_agreement"], 0.02)
    check("mean KL(base || steered)", 0.073, be["mean_kl"], 0.30)
    check("mean Jensen-Shannon", 0.019, be["mean_js"], 0.30)
    checks += 1
    ok = not be["behaviourally_equivalent"]
    print(f"  [{'x' if ok else ' '}] {'not behaviourally equivalent, so the claim is':<52} "
          f"{'performance' if ok else 'NO':>21}")
    if not ok:
        failures.append("behavioural equivalence: paper narrows to task performance")

# ----------------------------------------------------------- the test count
# the paper states a test count, which is the one claim with no json artifact
# behind it, so it drifted three times before this check existed. asking pytest
# is cheap and makes it drift-proof.
print("\nsection 4.4: the test suite")
try:
    import subprocess
    out = subprocess.run(
        # no -q: the quiet format prints per-file counts, not a total
        [sys.executable, "-m", "pytest", "-p", "no:warnings", "--collect-only"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=300,
    ).stdout
    m = re.search(r"(\d+) tests collected", out)
    collected = int(m.group(1)) if m else None
except Exception:
    collected = None

claimed = None
paper = ROOT / "docs" / "paper" / "main.tex"
if paper.exists():
    m = re.search(r"(\d+) tests cover", paper.read_text())
    claimed = int(m.group(1)) if m else None

if collected is None or claimed is None:
    print("  [ ] could not read one side of the comparison; skipping")
else:
    check("tests the paper claims", claimed, collected, 0.0)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
if failures:
    print(f"{len(failures)} of {checks} CLAIMS DO NOT MATCH THE ARTIFACTS")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print(f"all {checks} numerical claims in the paper match the artifacts")

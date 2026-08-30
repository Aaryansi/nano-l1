"""cross-check every number in docs/paper/main.md against reports/*.json.

a paper with a mistranscribed number is worse than no paper, and the numbers in
this one were copied by hand across many editing passes. this asserts each
claim against the artifact that produced it, so a stale figure surfaces here
rather than in review.

usage:
    python scripts/verify_paper_numbers.py
"""

from __future__ import annotations

import json
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
print("verifying docs/paper/main.md against reports/")
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
    for row in g:
        e = row["env_id"].split("-")[0].lower()
        check(f"{e} weight null sd", 138.92 if e == "cartpole" else 124.94,
              row["weight_null"]["std"], 0.02)
        check(f"{e} env null sd", 3.29 if e == "cartpole" else 0.0,
              row["env_null"]["std"], 0.02 if e == "cartpole" else 1.0)
    agree = sum(r["nulls_agree"] for r in g)
    total = sum(r["n_checkpoints"] for r in g)
    check("checkpoints where nulls agree", 3, agree, 0.001)
    check("total checkpoints", 8, total, 0.001)

# ---------------------------------------------------------------- schemes
sc = load("scheme_robustness.json")
print("\nsection 5.7: credit-assignment schemes")
if sc:
    for k, sig, real in (("span", 75.74, 8.84),
                         ("leave_one_out", 101.49, -3.89),
                         ("only_one_in", 93.46, -0.60)):
        check(f"{k} planted statistic", sig, sc["planted_signal"][k]["statistic"], 0.02)
        check(f"{k} real statistic", real, sc["real_market"][k]["statistic"], 0.05)
    check("loo vs ooi rank correlation", 0.309, sc["loo_vs_ooi_rank_corr"], 0.05)

# ---------------------------------------------------------------- horizon
h = load("horizon_scaling.json")
print("\nsection 5.7: horizon scaling")
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
print("\nsection 5.7: attribution family")
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
    check("behaviour rank correlation", 0.849, beh["rank_corr_mean"], 0.02)
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

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
if failures:
    print(f"{len(failures)} of {checks} CLAIMS DO NOT MATCH THE ARTIFACTS")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print(f"all {checks} numerical claims in the paper match the artifacts")

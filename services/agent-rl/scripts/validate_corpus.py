"""end-to-end validation of the built corpus.

runs the full stack on real data: load, walk-forward split, env rollout,
accounting checks. unit tests use synthetic fixtures on purpose, so this is the
only place the real corpus is exercised end to end.

it also measures the thing the rl agent will ultimately be compared against:
**is the kalshi market itself calibrated?** if a contract priced at 0.70
resolves yes 70% of the time, the market's implied probability is already a
well-calibrated forecast, and any edge must come from somewhere other than
predicting resolution better than the price does.

usage:
    python scripts/validate_corpus.py --corpus data/corpus/corpus_candles_60s.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import ACTION_NAMES, BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.costs import CostModel  # noqa: E402
from nano_rl.env.features import FEATURE_NAMES  # noqa: E402


def rollout(env: BinaryMarketEnv, policy, n_episodes: int, seed: int = 0):
    """run a fixed policy and collect per-episode totals."""
    rng = np.random.default_rng(seed)
    returns, fees, trades, telescope_err = [], [], [], []

    for ep in range(min(n_episodes, len(env.batch))):
        obs, _ = env.reset(seed=seed, options={"episode": ep})
        total, info = 0.0, {}
        for _ in range(env.n_steps):
            a = policy(obs, rng)
            obs, r, term, _, info = env.step(a)
            total += r
            if term:
                break
        returns.append(total)
        fees.append(info["fees"])
        trades.append(info["trades"])
        # the telescoping invariant, checked on real data
        telescope_err.append(abs(total - info["equity"]))

    return (
        np.array(returns),
        np.array(fees),
        np.array(trades),
        float(np.max(telescope_err)),
    )


def calibration(prices: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """reliability of a probability forecast against realised outcomes."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (prices >= edges[i]) & (prices < edges[i + 1])
        if m.sum() < 20:
            continue
        rows.append((0.5 * (edges[i] + edges[i + 1]), float(prices[m].mean()),
                     float(outcomes[m].mean()), int(m.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    print("=" * 70)
    print(f"corpus: {args.corpus}")
    print("=" * 70)
    print(f"  episodes    : {len(batch)}")
    print(f"  steps/ep    : {batch.n_steps}")
    print(f"  transitions : {len(batch) * batch.n_steps:,}")
    print(f"  duration    : {batch.duration_s:.0f}s")
    print(f"  yes rate    : {batch.settlement.mean():.4f}")
    print(f"  spot joined : {batch.has_spot}")

    spread = batch.ask - batch.bid
    mid_all = 0.5 * (batch.bid + batch.ask)
    print(f"  spread      : median={np.median(spread):.4f} mean={spread.mean():.4f}")
    print(f"  mid range   : {mid_all.min():.3f} .. {mid_all.max():.3f}")
    print(f"  mid at open : {mid_all[:, 0].mean():.4f} (coin-flip prior is 0.5)")

    # ---------------------------------------------------------------- splits
    split = walk_forward_split(batch)
    print("\nwalk-forward split")
    print(split.summary())
    assert split.train.open_epoch.max() < split.val.open_epoch.min()
    assert split.val.open_epoch.max() < split.test.open_epoch.min()
    print("  ordering    : OK (train < val < test, disjoint)")

    # ------------------------------------------------------- market calibration
    print("\nis the market itself calibrated?")
    print("  (if yes, beating it by predicting resolution is not available)")
    mid = 0.5 * (batch.bid + batch.ask)
    # use the price at the midpoint of each episode as the forecast
    mid_idx = batch.n_steps // 2
    prices = mid[:, mid_idx]
    rows = calibration(prices, batch.settlement)
    print(f"  {'bin':>8} {'mean price':>11} {'realised':>10} {'n':>6}  {'err':>7}")
    errs = []
    for centre, mean_p, realised, n in rows:
        err = realised - mean_p
        errs.append(abs(err) * n)
        print(f"  {centre:>8.2f} {mean_p:>11.3f} {realised:>10.3f} {n:>6}  {err:>+7.3f}")
    if rows:
        total_n = sum(r[3] for r in rows)
        print(f"  weighted mean |calibration error| = {sum(errs)/total_n:.4f}")

    # ------------------------------------------------------ lead-lag hypothesis
    if batch.has_spot:
        print("\nlead-lag: does spot carry information the kalshi price has not "
              "yet absorbed?")
        print("  (this is the project's one testable alpha hypothesis)")

        # spot_implied_gap is feature index 9+4 = 13 in MARKET_FEATURES order;
        # read it from the raw spot block instead to avoid index drift.
        gap = batch.spot[:, :, 4]  # spot_implied_gap
        ret = batch.spot[:, :, 0]  # spot_ret_since_open
        outcome = batch.settlement[:, None]

        for name, sig in (("spot_implied_gap", gap), ("spot_ret_since_open", ret)):
            # correlation between the signal and the eventual binary outcome,
            # pooled over all steps
            flat_sig = sig.reshape(-1)
            flat_out = np.repeat(outcome, batch.n_steps, axis=1).reshape(-1)
            if flat_sig.std() < 1e-12:
                print(f"  {name:<22} degenerate (no variance)")
                continue
            corr = float(np.corrcoef(flat_sig, flat_out)[0, 1])

            # and the incremental version: does the signal help ON TOP of price?
            price = mid.reshape(-1)
            resid_out = flat_out - price  # what the price already fails to explain
            corr_resid = float(np.corrcoef(flat_sig, resid_out)[0, 1])
            print(f"  {name:<22} corr(signal, outcome)={corr:+.4f}   "
                  f"corr(signal, outcome - price)={corr_resid:+.4f}")

        print("  the second column is the one that matters: correlation with what")
        print("  the price has ALREADY priced in is not tradeable.")

    # ------------------------------------------------------------- rollouts
    print("\npolicy rollouts on TEST split (never used for tuning)")
    env = BinaryMarketEnv(
        split.test, normalizer=split.normalizer, random_episode_order=False
    )
    free_env = BinaryMarketEnv(
        split.test,
        cost_model=CostModel(enabled=False),
        normalizer=split.normalizer,
        random_episode_order=False,
    )

    policies = {
        "always FLAT": lambda o, r: 1,
        "always LONG": lambda o, r: 2,
        "always SHORT": lambda o, r: 0,
        "random": lambda o, r: int(r.integers(0, 3)),
    }

    print(f"  {'policy':>13} {'mean pnl':>10} {'std':>9} {'fees':>9} {'trades':>7}")
    for name, pol in policies.items():
        rets, fees, trades, err = rollout(env, pol, args.episodes)
        print(
            f"  {name:>13} {rets.mean():>10.2f} {rets.std():>9.2f} "
            f"{fees.mean():>9.2f} {trades.mean():>7.1f}"
        )
        assert err < 1e-6, f"telescoping violated for {name}: max err {err}"

    print("\n  same policies with costs DISABLED (isolates friction)")
    print(f"  {'policy':>13} {'mean pnl':>10} {'std':>9}")
    for name, pol in policies.items():
        rets, _, _, err = rollout(free_env, pol, args.episodes)
        print(f"  {name:>13} {rets.mean():>10.2f} {rets.std():>9.2f}")
        assert err < 1e-6, f"telescoping violated for {name} (free): {err}"

    print("\n  telescoping invariant holds on real data for every policy")

    # --------------------------------------------------------------- features
    print(f"\nobservation: {len(FEATURE_NAMES)} features")
    obs, _ = env.reset(seed=0, options={"episode": 0})
    assert obs.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(obs))
    print(f"  actions: {list(ACTION_NAMES)}")
    print("  all finite: OK")
    print("\nvalidation passed")


if __name__ == "__main__":
    main()

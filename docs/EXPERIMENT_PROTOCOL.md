**Evaluation protocol for observation-ablation reference distributions**

This protocol defines the corrected experiment release. Earlier runs are exploratory and do not become prospective experiments by being rerun. Changes to this protocol after inspecting release outcomes must be recorded in the release manifest and described as exploratory.

The primary quantity is the expected episode-return difference between an unmodified policy and that same policy under a specified observation intervention:

`span = mean(return with all inputs) - mean(return with masked inputs)`.

This is a property of the policy, environment, masking intervention, and evaluation distribution. It does not measure the accuracy of a feature-level decomposition. Failure to distinguish a span from a candidate reference does not establish that the policy learned nothing, that the environment has no signal, or that an explanation is meaningless.

**Primary questions and comparisons**

1. How do the center, width, and empirical tail rank change when the reference is built by parameter randomization, training on signal-free data, training with blinded observations, or permuting settlement outcomes?
2. Does the reference change the conclusion for the same observed policy, rollout distribution, background, and evaluation episodes?
3. On synthetic data with known population information structure, how often does each rule flag an independently trained null or signal policy?
4. How much does suppressing one feature change attribution magnitude, behavior, and held-out task return? These are three separate effects.

Market comparisons between synthetic and real corpora are descriptive because exchangeability is not established. A normal z-score is a standardized distance within a reference, not a multiplier of statistical power. The two-part rank/normal rule remains a candidate diagnostic; empirical calibration must be reported for its specific setting. A zero-variance or under-budget reference is unresolved, not evidence for either scientific hypothesis.

**Data and evaluation**

- Keep the chronological market split and train-only normalizer. Save corpus hashes and split membership with the run.
- Use the tracked `data/releases/market_snapshot_v1` snapshot by default, verifying its content and membership before training. A newly downloaded API window is a new dataset, not an exact reproduction of the fixed snapshot.
- Generate synthetic training and evaluation corpora independently, using separate recorded corpus seeds. Fit normalizers on training only. A corpus held out from the reference bank is not automatically held out from the observed policy's training.
- Use all held-out market episodes for the primary endpoint comparison. If computational limits require a subset for feature attribution, record its membership and report that result as a subset estimate.
- Use the same evaluation episodes and masking randomness within a paired policy comparison. Keep episode/time-block identity when comparing training seeds.
- Report every observed training seed. Do not select the seed closest to a mean as the observed statistic. Report per-seed comparisons and summarize their variation.
- Label full-observation replacement separately from market-only replacement, which preserves the agent's true inventory state. Empirical row resampling preserves single-row support; it does not prove preservation of trajectory or hidden-state consistency.
- Compare the existing full-observation replacement with two market-input interventions that retain the real clock and all four inventory features. The first draws a fresh donor episode at each matching decision step; the second fixes a donor episode for the whole market trajectory. Use all evaluation episodes and all declared policy seeds, with independent synthetic training corpora and a common held-out synthetic evaluation corpus. Retained entry prices and unrealized P&L can reveal market information; these are state-preservation controls, not complete information-removal claims. Record masking seeds and treat uncertainty as conditional on those draws.

**Budgets and controls**

The market release uses five independent training seeds, 100 PPO updates per seed, and a 24-draw primary reference. The larger-reference sensitivity uses 99 draws. The main synthetic endpoint comparisons use 40 updates for both observed and reference agents, independently generated training/evaluation data, and the same architecture, normalization procedure, episode count, and frictions. These budgets are algorithm choices and must not be hidden as implementation details.

The real-data objective control trains both prediction and trading agents from scratch at 40 and 100 updates with five independent seeds per cell, using matched seeds and identical state-transition code. Reward differs; observation construction, action effects, split, training steps, and model architecture do not. Report both arms and their direct contrast. Significance in one arm and non-significance in the other is not a test of their difference.

For steering, use ten independent training seeds and fix the reported penalty at 20 before the release, retain all penalty settings rather than selecting the best after evaluation, and use independent synthetic evaluation data. Report the complete suppression/return tradeoff, raw paired episode results, seed variation, and confidence intervals. A non-significant difference is not equivalence. Any practical noninferiority margin must be justified and recorded before applying a noninferiority test; otherwise report intervals without an equivalence claim.

**Replication and uncertainty**

Training runs are independent seed replicates; shared evaluation episodes are another source of variation. For paired seed comparisons, resample common episode blocks across all selected seeds, rather than independently resampling the same evaluation corpus inside each seed. Use UTC-day blocks for the market and individual episodes for independent synthetic corpora. Report the number of available blocks and the limitations of a small seed/block count.

Synthetic calibration uses 30 independent trials at each of signal strengths 0, 0.2, and 1, conditional on the primary 24-draw reference bank. Report rejection rates, trial counts, and pointwise interval estimates; this budget has limited precision for rare false positives. If a reference bank is shared across trials, describe rates as conditional on that bank and report the bank's construction and uncertainty. Do not present one successful planted signal and one null non-rejection as measured power and specificity. A signal-strength sweep with one training run per strength is an effect-size illustration, not a detection-probability curve.

Before the full corrected release, add two standard rank comparators to the same synthetic trials: an upper-tail rank of the raw span and a two-sided rank of absolute deviations from the mean of the pooled observed and reference spans. Include the observation and count ties as at least as extreme. Pooling the center symmetrically is essential. Under joint exchangeability of the statistics and this permutation-equivariant scoring, each p-value is super-uniform marginally over the bank and observed draw. This is the established finite-rank argument, not a new theorem about reinforcement learning; see [Kuchibhotla](https://arxiv.org/abs/2005.06095v3).

The matching assumption is plausible only for zero-strength synthetic trials generated, trained and evaluated by the declared common procedure. Record and check generator parameters, episode/horizon counts, training configuration, normalization, evaluation policy, background size, masking-seed coupling, implementation hashes and numerical runtime when reusing a bank. A metadata match does not prove exchangeability. Market comparisons do not meet an established exchangeability assumption. The fixed bank's conditional false-positive rate may exceed alpha even when the marginal rank guarantee holds. Keep conditional rates and intervals, both tails, the rank floor, and whether that floor can resolve alpha. Do not use low-budget non-rejections as sensitivity evidence.

The legacy rule centers on references alone and also requires a normal tail threshold. It has no general exchangeability guarantee: uniformly designate one observation in the fixed pool `[-1, +1, 0, ..., 0]` with 18 zeros. Either extreme has legacy rank 0.05 and absolute standardized distance about 4.59, so the rule rejects 2 of 20 possible observed indices at alpha 0.05. The symmetric pooled two-sided rank assigns each extreme 0.10. This exact finite exchangeability example is distinct from an error estimate for the iid synthetic benchmark or the market data.

**Competence and validity gates**

Every explanation rollout must agree with scalar evaluation on actions, observations, costs, inventory, and terminal return. Non-finite rollout returns fail validation. Undefined standardized distances from zero-width references remain explicitly unresolved.

Control tasks contribute evidence about detection only when agents demonstrate both improvement over the declared baseline and nonzero observation dependence. Report failed training and degenerate rewards explicitly. More tasks on which the agent does not learn do not provide stronger evidence about sensitivity. Separate checkpoints along one training trajectory from independently trained agents.

The corrected control release uses the independent Stable Baselines3 2.5.0 PPO implementation, with separate policy/value trunks, two 64-unit actor layers, four independently seeded environment copies, 10 optimization epochs, minibatches of 64, and zero entropy coefficient. Total rollout size is 2,048 interactions across the copies. The native implementation remains selectable for comparison. Attribution exports only the trained categorical actor; the exported container's unused value head is zero and is not a calibrated critic. Exporting snapshots must not consume the training RNG stream.

Use 400,000 requested interactions per observed control agent and 30,000 per blinded reference agent, 24 reference draws, five independent observed training seeds per task, and 20 endpoint evaluation episodes. Competence is measured on 25 unmasked episodes. Keep both endpoint return arrays and the competence episode returns with their reset seeds, and validate all means and reference widths from those arrays. Separate checkpoint and whole-row-resampling sweeps remain descriptive sensitivity analyses.

The six control settings are CartPole, Acrobot, MountainCar, Pendulum with nine discrete torque choices, deterministic 4-by-4 FrozenLake, and CliffWalking capped at 200 steps. Grid observations are two normalized coordinates. Declare convergence thresholds of 475, -100, -110, -300, 0.8, and -20 respectively, together with nonzero observation dependence. These are evaluation gates, not claims that training will succeed. Keep all failed cases in the report. Record requested and actual interaction budgets, since complete 2,048-step rollout batches can exceed the requested count.

Calibration of the PPO critic compares predicted policy return with realized policy return. Converting a state value into a settlement probability requires a separate justified model and is not used here. Market calibration is descriptive; marginal calibration and small linear residual correlations do not prove absence of exploitable conditional information. Report entry/hold costs separately from open/close round-trip costs.

**Release requirements**

Keep configurations, raw per-seed episode outputs, corpus/split identities, source and checkpoint hashes, dependency versions, and artifact hashes, including the timing artifact written before manifest completion. Smoke runs have reduced budgets and cannot validate scientific conclusions. The selected full run must finish successfully before any manuscript numbers or publication figures are regenerated from it. Report failed comparisons and revised conclusions as readily as successful controls.

The headline and calibration endpoint archives retain both full-input and masked-input episode returns, episode identities, and keys for every observed/reference policy. Reconstruct spans, diagnostic decisions and calibration summaries from these archives before completing the manifest. A reused calibration bank copies its endpoint records and identifies the source JSON by content hash. The 99-draw sensitivity keeps its JSON and archive together under `reports/n99`; do not detach the JSON from its sidecar.

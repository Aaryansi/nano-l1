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
- Generate synthetic training and evaluation corpora independently, using separate recorded corpus seeds. Fit normalizers on training only. A corpus held out from the reference bank is not automatically held out from the observed policy's training.
- Use all held-out market episodes for the primary endpoint comparison. If computational limits require a subset for feature attribution, record its membership and report that result as a subset estimate.
- Use the same evaluation episodes and masking randomness within a paired policy comparison. Keep episode/time-block identity when comparing training seeds.
- Report every observed training seed. Do not select the seed closest to a mean as the observed statistic. Report per-seed comparisons and summarize their variation.
- Label full-observation replacement separately from market-only replacement, which preserves the agent's true inventory state. Empirical row resampling preserves single-row support; it does not prove preservation of trajectory or hidden-state consistency.

**Budgets and controls**

The market release uses five independent training seeds, 100 PPO updates per seed, and a 24-draw primary reference. The larger-reference sensitivity uses 99 draws. The main synthetic endpoint comparisons use 40 updates for both observed and reference agents, independently generated training/evaluation data, and the same architecture, normalization procedure, episode count, and frictions. These budgets are algorithm choices and must not be hidden as implementation details.

The real-data objective control trains both prediction and trading agents from scratch at each selected budget, using matched seeds and identical state-transition code. Reward differs; observation construction, action effects, split, training steps, and model architecture do not. Report both arms and their direct contrast. Significance in one arm and non-significance in the other is not a test of their difference.

For steering, fix the reported penalty at 20 before the release, retain all penalty settings rather than selecting the best after evaluation, and use independent synthetic evaluation data. Report the complete suppression/return tradeoff, raw paired episode results, seed variation, and confidence intervals. A non-significant difference is not equivalence. Any practical noninferiority margin must be justified and recorded before applying a noninferiority test; otherwise report intervals without an equivalence claim.

**Replication and uncertainty**

Training runs are independent seed replicates; shared evaluation episodes are another source of variation. For paired seed comparisons, resample common episode blocks across all selected seeds, rather than independently resampling the same evaluation corpus inside each seed. Use UTC-day blocks for the market and individual episodes for independent synthetic corpora. Report the number of available blocks and the limitations of a small seed/block count.

Synthetic calibration reports rejection rates, trial counts, and interval estimates. If a reference bank is shared across trials, describe rates as conditional on that bank and report the bank's construction and uncertainty. Do not present one successful planted signal and one null non-rejection as measured power and specificity. A signal-strength sweep with one training run per strength is an effect-size illustration, not a detection-probability curve.

**Competence and validity gates**

Every explanation rollout must agree with scalar evaluation on actions, observations, costs, inventory, and terminal return. Non-finite results fail validation.

Control tasks contribute evidence about detection only when agents demonstrate both improvement over the declared baseline and nonzero observation dependence. Report failed training and degenerate rewards explicitly. More tasks on which the agent does not learn do not provide stronger evidence about sensitivity. Separate checkpoints along one training trajectory from independently trained agents.

Calibration of the PPO critic compares predicted policy return with realized policy return. Converting a state value into a settlement probability requires a separate justified model and is not used here. Market calibration is descriptive; marginal calibration and small linear residual correlations do not prove absence of exploitable conditional information. Report entry/hold costs separately from open/close round-trip costs.

**Release requirements**

Keep configurations, raw per-seed episode outputs, corpus/split identities, source and checkpoint hashes, dependency versions, and artifact hashes. Smoke runs have reduced budgets and cannot validate scientific conclusions. The selected full run must finish successfully before any manuscript numbers or publication figures are regenerated from it. Report failed comparisons and revised conclusions as readily as successful controls.

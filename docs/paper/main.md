# Sanity Checks for Reinforcement Learning Explanations

*Draft. Target: 8 pages, two-column ML format. Anonymise before submission.*

---

## Abstract

Feature attribution is increasingly proposed as a basis for auditing reinforcement
learning agents. We show that an attribution can be stable across training runs,
precisely estimated, semantically plausible, and carry no information whatsoever.
On a prediction market that is empirically efficient, a PPO agent whose return is
statistically indistinguishable from abstention produces attributions with a
cross-seed rank correlation of 0.849, unanimous agreement on the most important
feature, and a top-5 ranking whose adjacent gaps exceed their own standard
errors. Every conventional check passes. We introduce a null-model test, built on
the efficiency identity that attribution schemes already satisfy, which compares
an agent's attribution span against the span of agents trained where the
observation channel carries no information. The test rejects a planted signal at
z = 12.27 and fails to reject the market agent at z = +0.23. We further show that
the established sanity check for this purpose, randomising network parameters,
does not transfer to reinforcement learning: on CartPole its reference
distribution is 42× wider than an environment-level null, and across eight
checkpoints on two control tasks it never exceeds z = 2.45, including on a
converged policy. Finally, we show the market agent's attribution can be driven
from 39.9% to 3.2% of total mass at no measurable cost in return, while the same
intervention on a feature that genuinely carries signal destroys 81% of return.
Attribution is therefore not identified by behaviour when the attributed feature
does not matter.

---

## 1 Introduction

A common argument for interpretability is oversight: if we can see which inputs
drive a policy, we can decide whether to trust it. This argument requires that an
attribution mean something. It is not obvious that attributions do, and it is
less obvious how one would tell.

The difficulty is that attribution methods return an answer regardless. Given any
policy and any state, Shapley values, integrated gradients, and permutation
importance all produce a vector of numbers, ranked, signed, and plottable.
Nothing in the output distinguishes an attribution that reflects structure the
agent exploits from one that reflects nothing at all.

In supervised learning this is handled by randomisation tests. Adebayo et al.
[1] compare explanations of a trained model against explanations of a model with
randomised parameters, and against a model trained on permuted labels. If the
explanation does not change, it does not depend on what the model learned. These
tests are standard, and applying them killed several saliency methods.

Reinforcement learning has adopted only half of this. Huber et al. [4] apply the
parameter-randomisation test to saliency maps for Atari agents, and note that the
data-randomisation test has no obvious RL analogue: there is no labelled dataset
to permute, and it is unclear how a sequential decision maker is supposed to
respond to artificially corrupted features.

This paper supplies that analogue and shows that the half already adopted does
not work.

**Contributions.**

1. **An environment-level null for RL.** Rather than corrupting the model or the
   labels, we corrupt the environment's *observation channel*, leaving dynamics
   and reward intact. The agent trains normally, on a real objective, with
   nothing to condition on. This requires no access to task internals and applies
   to any environment. (§3)

2. **The established null fails in RL, measurably.** Parameter randomisation
   produces a reference distribution 42× wider on CartPole and degenerate on
   Acrobot. Across eight checkpoints on two tasks it never exceeds z = 2.45,
   including on a converged CartPole policy scoring 404 of 500, while the
   environment null reaches z = 117. The two disagree on five of eight. (§5.5)

3. **A validation protocol with ground truth.** Explanation evaluation is
   hampered by the absence of ground truth [3]. We construct environments where
   the correct attribution is known: a planted signal, a pure null, and a decoy
   that is predictive in-sample and provably worthless out of sample. (§4.1, §5.2)

4. **A worked case of an uninformative explanation passing every check.** (§5.1)

5. **Attribution is not identified by behaviour.** An auxiliary training penalty
   drives the market agent's dominant feature from 39.9% to 3.2% of attribution
   mass at no measurable cost in return; the same intervention on a genuinely
   load-bearing feature costs 81% of return. (§5.6)

**What we do not claim.** The argument that RL attribution should target outcomes
rather than policy outputs is due to Beechey et al. [2], whose SVERL framework
unifies behaviour, outcomes and prediction, and whose FastSVERL [7] addresses the
scalability, off-policy and temporal problems we only name. Our test is an
adaptation of the data-randomisation test of [1]. Deletion-based fidelity
evaluation is established practice in XRL [3]. Our contribution is the RL
construction, the demonstration that the incumbent alternative fails, and the
empirical consequences.

---

## 2 Related work

**Shapley values in RL.** Beechey et al. [2] give a theoretical account of
Shapley-value explanation for RL, showing that earlier applications explained the
wrong object, and proposing characteristic functions built on agent performance.
Their framework identifies three explanatory targets: behaviour, outcomes, and
prediction. FastSVERL [7] makes the approach tractable and addresses temporal
dependence and off-policy data. We adopt the outcomes framing and implement it
independently; we do not extend the theory.

**Sanity checks for explanations.** Adebayo et al. [1] introduced parameter- and
data-randomisation tests; subsequent work has examined their limitations [8] and
extended them to object detectors. Huber et al. [4] apply the parameter test to
perturbation-based saliency for Atari agents and observe that the data test lacks
an RL analogue. We propose one.

**Manipulating explanations.** Slack et al. [5] construct adversarial classifiers
that detect the off-manifold points LIME and SHAP query, and behave differently
on them, yielding innocuous explanations for biased models. That is an attack on
the explainer's sampling. Our steering result is different in kind: we train a
normal agent with an invariance penalty, the resulting explanation is *correct*
about the resulting model, and the point is that behaviour does not pin the
explanation down. It is a non-identifiability result rather than an attack.

**Uncertainty in Shapley estimation.** A line of work certifies top-k Shapley
rankings against Monte Carlo error [6]. These guarantees concern the estimator.
We show in §5.4 that they are orthogonal to whether the explained model learned
anything: a ranking can be certified stable and describe nothing.

**Evaluation in XRL.** Surveys note that the absence of ground-truth explanations
is a central obstacle, and that perturbation-based fidelity — removing features
identified as important and measuring performance degradation — is the common
recourse [3]. We use deletion curves in that established sense and do not claim
them as a contribution.

---

## 3 Method

### 3.1 The statistic

Let $v(S)$ denote the expected episode return of an agent that observes only the
features in $S \subseteq N$, with features outside $S$ replaced at every timestep
by draws from a reference distribution. Define the **attribution span**

$$
\Delta = v(N) - v(\emptyset).
$$

Both Shapley values and integrated gradients satisfy an additivity identity —
efficiency and completeness respectively — under which per-feature attributions
sum to $\Delta$. The span can therefore be read off any such attribution.

It is worth being precise about what $\Delta$ is, because it is easy to overstate.
$\Delta$ is **not a Shapley quantity**. It is the difference between two masked
rollouts, and depends only on the masking scheme. Efficiency tells us the Shapley
values happen to sum to it. This has a practical consequence we exploit: $\Delta$
requires two evaluations rather than $2^{|N|}$.

### 3.2 The null

The test compares an observed $\Delta$ against the distribution of $\Delta$ for
agents that had nothing to learn. We construct these by corrupting the
observation channel:

> **Blind-observation null.** Replace every observation with an independent draw
> from a fixed reference distribution matched to the environment's observation
> moments. Leave dynamics and reward untouched. Train normally.

An agent trained this way faces the real task and receives real return; it simply
cannot condition on anything. This is the RL counterpart of training on permuted
labels, and unlike parameter randomisation it produces a *normally trained*
policy — the failure mode that occurs in deployment, rather than an artificial
one.

Matching the reference moments matters. A network fed out-of-range inputs fails
for reasons of scale rather than information, which would be a different
experiment.

### 3.3 The test

Given $\Delta_{\text{obs}}$ and null draws $\Delta_1, \dots, \Delta_n$, we report
a rank p-value and a normal-approximation p-value, and require both:

$$
p_{\text{rank}} = \frac{\#\{i : |\Delta_i - \bar\Delta| \geq |\Delta_{\text{obs}} - \bar\Delta|\} + 1}{n+1},
\qquad
z = \frac{\Delta_{\text{obs}} - \bar\Delta}{s}.
$$

Two p-values are reported because neither suffices. The rank statistic assumes no
distributional shape but bottoms out at $1/(n+1)$: with $n = 8$ it cannot reject
at $\alpha = 0.05$ however extreme the observation, and in an early run it
reported $p = 0.111$ for a signal lying 10.5 standard deviations outside the
null. The normal statistic has resolution but assumes a shape the null need not
have. Requiring both keeps the test conservative in the direction that matters.

**Degenerate nulls.** If every null draw is identical, $s = 0$ and $z$ is
undefined. This is not an absence of information but its opposite: any deviation
is maximally surprising. We return $z = \pm\infty$ and surface the degeneracy
explicitly. It occurs in practice — §5.5 and §5.7 — whenever the task lets a
blind agent converge on a single behaviour.

---

## 4 Experimental setup

### 4.1 Environments with known answers

Explanation evaluation lacks ground truth [3]. We construct three synthetic
corpora in which the correct answer is fixed by construction. Episodes are
fixed-length with no early termination, so the horizon is controlled exactly.

- **Planted signal.** One of 18 features is informative about the binary outcome;
  the rest are noise or constants. Any correct method must rank it first.
- **Null.** Identical shapes and frictions, no informative feature. The optimal
  policy is to abstain.
- **Decoy.** One feature correlates 0.989 with the outcome in the first 60% of
  episodes and 0.006 in the remainder. An agent trained on the first part learns
  it; on held-out episodes acting on it earns nothing.

The precise claim about the planted corpus matters. The planted feature is the
only one informative about the *outcome*. Others can still affect the *return*
without predicting anything, because return depends on behaviour: an agent that
cannot tell where it is in an episode acts incoherently and pays transaction
costs. The test is that the planted feature ranks first with the majority of
attribution mass, not that everything else is zero.

### 4.2 A market with terminal ground truth

Our main empirical setting is Kalshi `KXBTC15M`, a binary contract settling to
\$1.00 if BTC is higher after 15 minutes and \$0.00 otherwise. We use 6,428
settled contracts over 68 days (89,992 transitions, outcome rate 0.4995), with a
walk-forward split by time and purge gaps; the test split is evaluated once.

This instrument was chosen for one property: every episode resolves to a known
value, so a critic's $V(s)$ can be checked against realised settlement frequency.
State is 18 causal features (nine from the contract's own tape, five from Binance
BTC spot joined backward as-of, four position features). Actions are target
inventory in $\{-1, 0, +1\} \times q_{\max}$. Reward is the change in
mark-to-market equity net of costs, which telescopes exactly to net episode P&L.

Transaction costs are the exchange's published taker schedule,
$\lceil 0.07 \cdot C \cdot P (1-P) \rceil$, maximised at $P = 0.5$ where this
contract trades by construction. With the measured 1-cent modal spread this is a
9% round-trip hurdle on a 50-cent notional, and it determines most of what
follows.

### 4.3 Control tasks

CartPole-v1 (4 features) and Acrobot-v1 (6 features) provide settings with no
relationship to trading and well-understood optimal policies. Both have
observation spaces small enough to enumerate all $2^{|N|}$ coalitions, so
per-feature Shapley values there are exact.

### 4.4 Reproducibility

All results derive from public endpoints with no API keys. Hyperparameters were
fixed a priori from synthetic sweeps where the correct answer is known
analytically, not tuned on validation performance. The full pipeline is one
command; 226 tests cover the environment, the attribution implementations, and
six no-lookahead properties verified by mutation testing.

One caveat. The exchange serves a moving window, so re-running the ingest returns
a different set of contracts. A cold rebuild five days later produced 6,425
episodes against 6,428, sharing 5,960 tickers. **The overlapping episodes are
byte-identical on every field**: the ingest is deterministic per contract, and
what changes is membership. All conclusions reproduced on the rebuilt corpus
(§5.8).

---

## 5 Results

### 5.1 An explanation that passes every check and says nothing

The market is empirically efficient. Its own implied probability has a weighted
mean absolute calibration error of 0.0172 across ten bins. Spot return correlates
0.476 with the eventual outcome but only 0.016 with the residual after the price
has been accounted for.

No policy beats abstention (Table 1). Losses track turnover almost exactly.

**Table 1.** Test split, 1,284 held-out episodes. p-values are paired bootstrap
against always-flat on the same episodes, necessary because per-episode P&L has a
standard deviation near 50 against differences near 1.

| policy | mean P&L | trades/ep | p vs flat |
|---|---|---|---|
| always-flat | +0.000 | 0.00 | — |
| buy-and-hold | −1.608 | 1.00 | 0.23 |
| PPO (5 seeds) | −0.595 ± 0.176 | 1.37 | 0.10 |
| logistic (refit) | −5.729 | 2.52 | <0.0001 |
| mean-reversion | −11.615 | 4.94 | <0.0001 |
| random | −18.532 | 9.33 | <0.0001 |

PPO's shortfall is its own. On synthetic data containing no signal by
construction, the same agent scores −0.47 ± 0.47; here it scores −0.595 ± 0.176.
Nothing is left to attribute to the market.

Yet its explanations look entirely reasonable. Across five seeds whose pairwise
performance differences are statistically indistinguishable in all 10 pairs:

- behaviour attributions have cross-seed rank correlation **0.849** (min 0.761),
  with **100%** agreement on the top feature and **100%** sign agreement;
- the top-5 ranking is **fully certified**, all four adjacent gaps exceeding
  their combined standard errors (`time_to_expiry_frac` 0.2625 ± 0.0009);
- the ranking is semantically legible: time-to-expiry dominates, which reads as
  an agent that learned when not to trade.

The null test disagrees. Against 24 blind-trained agents (null span
+8.19 ± 5.11):

**Table 2.** Null-model test on the market.

| case | span | z | p | verdict |
|---|---|---|---|---|
| planted signal | +70.89 | **+12.27** | 0.040 | informative |
| null corpus | +11.86 | +0.72 | 0.480 | not distinguishable |
| **real market** | **+9.37** | **+0.23** | 0.840 | **not distinguishable** |

The test has power and specificity. The market agent's explanation sits at the
centre of the distribution of explanations of nothing.

We flag one consequence for our own reporting. An earlier draft of this work
stated that "the whole observation is worth +5.914 per episode against being
blind" and presented it as a finding. Blind agents produce +8.19 ± 5.11. It was
noise.

### 5.2 The test is validated against ground truth

On the planted corpus, outcome attribution ranks the planted feature **first with
55% of total attribution mass**. This check is unavailable on real data and is
what licenses the rest.

On the decoy corpus, an agent trained where the decoy predicts the outcome
(in-sample +47.24, held out −6.83) is explained on held-out episodes where the
decoy is provably worthless:

| method | share of mass | rank |
|---|---|---|
| per-decision, $\pi(a\mid s)$ | 6.5% | **3 of 18** |
| outcome-based | 1.1% | **10 of 18** |

Both statements are true. The policy does key on the decoy, so per-decision
attribution is correct about behaviour and misleading about value. An auditor
asking what an agent relies on that does not work gets the wrong answer from it.

Deletion curves [3] agree: removing features in the order each ranking considers
important, the outcome-based ranking degrades return faster (AUC −187.8 against
−165.3; a random ranking gives +366.6). The gap concentrates at k = 1.

### 5.3 How much edge is required

Sweeping planted signal strength calibrates the test. The x-axis is the agent's
*measured* edge, which is what a practitioner has.

| agent edge / ep | z | detected |
|---|---|---|
| +2.25 | +1.63 | no |
| +3.45 | +2.62 | yes |
| +15.41 | +6.06 | yes |
| +47.11 | +12.27 | yes |

Detection begins near +3.45 per episode. The market agent earns −0.418: an order
of magnitude below threshold and of the wrong sign. These edges are in-sample and
therefore optimistic; the threshold should be read as a lower bound.

### 5.4 Estimation guarantees do not substitute

Work on Shapley estimation error [6] certifies that a top-k ranking is stable
under Monte Carlo noise. That is a guarantee about the estimator. The market
agent's top-5, with standard errors around 0.001 on a leading value of 0.26, has
all four adjacent pairs separated beyond their combined error: **fully certified
by that criterion**. The same explanation fails the null test at z = +0.23.

A ranking can be certified stable while the explanation it ranks is
indistinguishable from an explanation of nothing. Only one of these two
properties is commonly reported.

### 5.5 The established null does not transfer to RL

We construct both nulls for the same task and compare.

**Table 3.** Null construction, CartPole-v1 and Acrobot-v1. Shapley values here
are exact (all coalitions enumerated).

| environment | parameter randomisation [1] | environment randomisation (ours) | ratio |
|---|---|---|---|
| CartPole-v1 | +55.34 ± **138.92** | −1.35 ± **3.29** | 42× |
| Acrobot-v1 | +64.03 ± **124.94** | +0.00 ± **0.00** | degenerate |

A randomly initialised policy behaves arbitrarily, so masking its inputs moves
return unpredictably and the reference distribution is very wide. The resulting
test has almost no power:

| environment | training | return | z (env null) | z (weight null) |
|---|---|---|---|---|
| CartPole | 10% | 291.3 | **+89.4** | +1.71 |
| CartPole | 100% | 404.3 | **+117.6** | +2.37 |
| Acrobot | 50% | −500.0 | +0.00 | −0.51 |
| Acrobot | 100% | −120.6 | **+∞** | +2.45 |

Across eight checkpoints the parameter null never exceeds z = 2.45, including on
a converged CartPole policy scoring 404 of 500. The two nulls disagree on five of
eight. The choice of null is not a detail; it decides the verdict, and the
incumbent construction is the one that fails.

Acrobot additionally shows the test tracking *whether there is anything to
explain* rather than agent quality. Its agent fails completely until it learns
(−500 at 10%, 25%, 50%), the span is exactly 0.00 at those checkpoints, and the
test correctly declines. CartPole behaves differently because its observations
matter at every skill level, and the test fires throughout. We do not claim that
undertrained agents generally produce empty explanations; they do not.

### 5.6 Attribution is not identified by behaviour

We add an auxiliary penalty during training on the divergence between
$\pi(\cdot \mid s)$ and $\pi(\cdot \mid s')$, where $s'$ has one feature
resampled from the batch marginal — the same interventional perturbation the
attribution measures.

**Table 4.** Steering a single feature's attribution.

| corpus | target | attribution | return | p vs baseline |
|---|---|---|---|---|
| market | `time_to_expiry_frac` | 39.9% → **3.2%** | −3.49 → −1.65 | **0.66** |
| planted signal | `spot_ret_since_open` | 46.5% → **1.5%** | +45.04 → **+7.92** | **<0.001** |

The attribution is equally suppressible in both cases (92% and 97%). What differs
is the price. Where the feature is the planted signal, suppressing it destroys
**81% of return**. On the market it costs nothing measurable.

Explanations are therefore steerable at no cost exactly where they are not
tracking anything, which independently corroborates §5.1 by a route that shares
no machinery with the null test. The control matters: had steering succeeded on
both corpora, the penalty would merely be defeating the attribution method.

### 5.7 What the verdict does and does not depend on

**Attribution family.** Integrated gradients, which shares no machinery with
Shapley, ranks features at 0.981 correlation with Shapley across five seeds and
is comparably seed-stable (0.750 against 0.800). Per-feature attributions are not a
Shapley artefact. IG cannot test the outcome-level claim, however: episode return
is not differentiable through the environment, so IG has no outcome analogue. Its
behaviour-level span reports the market agent as informative (z = +4.58) — which
is consistent rather than contradictory, and is precisely the
behaviour-versus-outcome distinction of [2]. The agent's behaviour does respond
to its observations; that responsiveness earns nothing.

**Credit assignment.** Leave-one-out ($v(N) - v(N\setminus i)$) and only-one-in
($v(\{i\}) - v(\emptyset)$) are the extremes of the average Shapley takes, and
neither satisfies efficiency, so each carries its own total.

| scheme | planted signal | real market |
|---|---|---|
| span | +75.74 (z = +10.66) | +8.84 (z = −0.42) |
| leave-one-out | +101.49 (z = +10.96) | −3.89 (z = −1.57) |
| only-one-in | +93.46 (z = +2.29) | −0.60 (z = −0.72) |

Every scheme fires on the planted signal and none on the market: the verdict is
scheme-independent. **The per-feature ranking is not.** Leave-one-out and
only-one-in rank features at only 0.309 correlation with each other on the same
agent. *Is there anything to explain* is robust; *which features matter* is not,
and a ranking from a single scheme should not be reported as the answer.

**Horizon.** We hypothesised that the parameter null's excess variance arises
because randomising weights perturbs the measurement's *domain* — return being a
functional of the policy-induced state distribution — and that this compounds
along the trajectory. The prediction is that its variance grows with horizon
faster than the environment null's. It does not. Over horizons 4 to 28 both grow
near-linearly at indistinguishable rates ($\alpha = 0.94$, $r^2 = 0.995$ against
$\alpha = 0.83$, $r^2 = 0.998$), with the ratio flat at 1.7–1.8×. **The
hypothesis is rejected**, and the 42× gap on CartPole remains unexplained.

The experiment did establish something else. At horizon 56 the environment null
collapses to a standard deviation of 0.04 with two distinct values across sixteen
agents: every blind agent converged on the same abstaining policy. This is the
Acrobot degeneracy again, and it now has a mechanism — the environment null
degenerates when the task affords enough time to learn that inaction is optimal.

### 5.8 Reproduction from scratch

All figures above are from a corpus rebuilt from scratch after deleting every
cached file. An earlier corpus, collected five days before and sharing 93% of its
contracts, gave PPO −0.418 ± 0.260 (p = 0.129) against −0.595 ± 0.176
(p = 0.100), and null-test z of +12.27 / +0.72 / −0.15 against +12.27 / +0.72 /
+0.23. Every conclusion is identical on both. `buy-and-hold` moved from +0.068 to
−1.608, confirming that its earlier positive figure reflected that period's 52.3%
outcome rate rather than an edge.

---

## 6 Limitations

**The construction is adapted.** The test is the data-randomisation test of [1]
with an RL-appropriate null. The outcomes framing is [2]'s. Deletion-based
fidelity is standard [3].

**The span measures information in aggregate.** An explanation could clear the
test and still misattribute among features. §5.7 shows the ranking is
scheme-dependent even when the verdict is not.

**Temporal credit assignment is named, not solved.** Masking a feature for the
whole episode measures whether it mattered, not when. This is FastSVERL's [7]
territory.

**Off-policy attribution is not handled.** Masked coalitions induce a different
policy and therefore a different state distribution; estimates are on-policy for
the masked policy, not counterfactuals about the unmasked one. Read the numbers
comparatively.

**The 42× gap is unexplained.** Our proposed mechanism was tested and rejected
(§5.7). A remaining conjecture, consistent with all three environments but
untested, is that the parameter null's width depends on how often a randomly
initialised network is an accidentally competent policy: sometimes on CartPole
and Acrobot, never on a market with no exploitable signal.

**Scope.** One real domain and two classic-control tasks. One attribution family
for the outcome-level tests. Corpus membership is not reproducible across time,
though content is.

---

## 7 Conclusion

An attribution can be stable, precisely estimated, plausible, and empty. The
checks practitioners apply — consistency across runs, separation beyond
estimation error, semantic legibility — do not detect this, because none of them
ask whether the explained agent learned anything. A null-model test does, using a
statistic the attribution framework already provides. The established test for
this purpose does not survive the move to reinforcement learning, and we
recommend against its use there without the measurement in §5.5.

---

## References

[1] Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.
*Sanity Checks for Saliency Maps.* NeurIPS 2018. arXiv:1810.03292

[2] Beechey, D., Smith, T. M. S., Şimşek, Ö. *Explaining Reinforcement Learning
with Shapley Values.* ICML 2023. arXiv:2306.05810

[3] Surveys of explainable reinforcement learning; see e.g. *A Survey on
Explainable Deep Reinforcement Learning*, arXiv:2502.06869, and Milani et al.,
ACM Computing Surveys, 2024.

[4] Huber, T., Limmer, B., André, E. *Benchmarking Perturbation-Based Saliency
Maps for Explaining Atari Agents.* Frontiers in Artificial Intelligence, 2022.

[5] Slack, D., Hilgard, S., Jia, E., Singh, S., Lakkaraju, H. *Fooling LIME and
SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AIES 2020.
arXiv:1911.02508

[6] Statistical significance of feature importance rankings; see
arXiv:2401.15800 and related work on PAC-style top-k Shapley identification.

[7] Beechey, D., Şimşek, Ö. *Approximating Shapley Explanations in Reinforcement
Learning.* NeurIPS 2025. arXiv:2511.06094

[8] Binder, A. et al. *Shortcomings of Top-Down Randomization-Based Sanity
Checks.* arXiv:2211.12486

[9] Lundberg, S., Lee, S.-I. *A Unified Approach to Interpreting Model
Predictions.* NeurIPS 2017.

[10] Sundararajan, M., Taly, A., Yan, Q. *Axiomatic Attribution for Deep
Networks.* ICML 2017.

[11] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. *Proximal
Policy Optimization Algorithms.* arXiv:1707.06347

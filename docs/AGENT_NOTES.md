# agent notes: what ppo does on a no-signal problem

supporting material for docs/REPORT.md. records phase-3 findings that shape how
phase-4 results must be read.

## why ppo rather than dqn

measured, not assumed. on the null corpus, tabular q learned action values in
the wrong direction:

| action | learned | true | bias |
|---|---|---|---|
| SHORT | -1.258 | -2.750 | **+1.492** |
| LONG | -2.262 | -2.750 | **+0.488** |
| FLAT | -0.701 | +0.000 | **-0.701** |

trading is overestimated and abstention underestimated, which is the max
operator's known bias pointing in exactly the worst direction for a problem
whose correct answer is mostly "do not trade". a policy gradient has no max
operator. ppo also carries an explicit value head, which docs/MDP.md section 1.2
makes falsifiable against realised settlement.

## the feature-scaling bug

ppo initially returned 0.00 on a corpus where the benchmark was 47.25, by
collapsing to always-FLAT. cause: `volume_rate` is log1p(volume) and sits at a
constant 9.210, which saturated **31.3%** of first-layer tanh units to zero
gradient.

| variant | return | benchmark |
|---|---|---|
| raw obs, entropy 0.01 | 0.00 | 47.25 |
| raw obs, entropy 0.05 | 0.00 | 47.25 |
| normalised obs, entropy 0.01 | **47.18** | 47.25 |
| normalised obs, entropy 0.05 | 47.21 | 47.25 |

entropy tuning changed nothing on raw inputs, which rules out the exploration
explanation. normalisation drops saturation to 0.0%.

this failure is worth naming because **a saturated network that never learns is
indistinguishable from an agent correctly concluding there is no signal**, and
this project's headline result is precisely a claim about there being no signal.
the env now warns when constructed without a normalizer if any feature exceeds
magnitude 5, and tests/test_agents.py::TestNormalizerGuard pins that.

## ppo does not reliably learn to abstain on pure noise

the honest limitation. on the null corpus, where FLAT earns exactly 0.00:

| entropy coef | return | trades/ep | flat fraction | final entropy |
|---|---|---|---|---|
| 0.01 | -0.47 +/- 0.47 | 0.50 | 0.62 | 0.540 |
| 0.05 | -0.65 +/- 0.65 | 0.33 | 0.71 | 0.689 |
| 0.20 | -1.07 +/- 1.07 | 0.50 | 0.38 | 1.087 |

three things to note.

**the outcome is bimodal across seeds.** in every row the standard deviation
equals the magnitude of the mean, because one seed abstains at exactly 0.00 and
the other commits to a direction at roughly -1. this is why results must be
reported as mean +/- std over multiple seeds: a single run could show either
outcome and neither would be representative.

**more training makes it worse, not better.** at 80 updates the flat fraction is
0.62; at 250 updates it collapses to 0.03 with the policy 96% LONG. the agent
overfits in-sample settlement noise over time. this is the opposite of the usual
under-training diagnosis and it sets the training budget.

**entropy tuning does not fix it.** raising the coefficient to 0.2 keeps entropy
near uniform (1.087 against ln 3 = 1.099) so the policy simply trades at random,
which is worse. 0.01 remains the best setting and still solves the learnable
corpus at 100.0% and 99.9% across two seeds.

### consequence for phase 4

ppo carries a small negative bias of roughly -0.5 per episode on data with no
signal, against a ±50 per-episode payoff. so a slightly negative ppo result on
the real corpus **cannot be attributed entirely to the market**; part of it is
this artefact. the always-flat baseline is what makes the difference visible,
and the report must not claim ppo "learned to abstain" without qualification.

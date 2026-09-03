# nano-l1

**no free null.** deciding whether an explanation of an RL agent means anything
requires a reference distribution: what it would have looked like had the agent
learned nothing. reinforcement learning has no agreed way to build one. this
builds four, on a deep-RL agent trading Kalshi 15-minute binary contracts, and
each one fails in its own way. built alongside an existing Go matching engine,
which is left intact.

[**the paper**](docs/paper/main.pdf) &middot;
[4-page version](docs/paper/workshop.pdf) &middot;
[source](docs/paper/main.tex)

---

### why this exists

Interpretability is proposed as a basis for oversight: if we can see which
inputs drive a policy, we can decide whether to trust it. That argument needs
attributions to carry information about the agent. This project shows they need
not, and that the reference distribution you compare against decides the answer.

### what it finds

**the trading result is negative and that is not the interesting part.** the
market is efficient (calibration error 0.0172), Kalshi's fee schedule imposes a
9% round-trip hurdle, and nothing beats doing nothing. the agent's own shortfall
is shown to be its measured bias rather than a fact about the market.

**an explanation can pass every check we could apply and still be empty.** the agent
has no measurable edge, yet its attributions are stable across seeds (0.850 rank
correlation, unanimous top feature), separated beyond their own estimation
error, and read plausibly. against agents trained where there is provably
nothing to learn, the test cannot tell them apart (z = +0.72 on a held-out null
corpus) while correctly flagging a planted signal (z = +12.27).

**explanations are cheap to forge.** an auxiliary training penalty drives the
agent's dominant feature from 39.9% to 3.2% of attribution mass at no measurable
cost in return. the same intervention on a feature that genuinely carries signal
destroys 81% of return. no deception and no attack on the explainer is required;
it falls out of ordinary training whenever the feature is not load-bearing.

**the test separates tasks, not datasets.** every case where it fires would
otherwise be synthetic, which invites the objection that it detects planted
signal rather than information. holding the corpus, features, normalizer, split
and null construction fixed and varying only the objective: on the same real
episodes, an agent scored for *calling* the settlement fires at z = +3.39, while
the trading agent does not. the market data is not uninformative; a calibrated
price is very informative. the information is not monetizable against a 9%
round-trip fee, and an attribution of the trading agent cannot tell those two
situations apart.

**every null construction perturbs something besides the information.** the
signal-free corpus varies the corpus. blinding the observation channel removes
the agent's capacity to respond, collapsing the reference to a point mass and
firing on a corpus built to contain no signal. permuting the outcomes leaves the
price forecasting a label it no longer matches and hands the null agents a $49
per episode arbitrage, so the agent under test lands fifteen standard deviations
*below* its own null. which perturbation you accept decides the verdict.

**parameter randomization is an underpowered reference in RL.** across sixteen
checkpoints on four control tasks, using randomized weights to generate
reference draws never exceeds z = 2.45, including on a converged agent. its
reference distribution is, to three significant figures, the spread of
random-initialization return, so it measures the variance of initialization and
not the explanation.

**the canonical version of that check clears the empty explanation.** run as
adebayo et al. actually propose, degrading one explanation by progressively
randomizing layers, it clears the empty market explanation on both explanatory
targets we tried (rank correlation 0.38 on behaviour, -0.02 on outcomes, so the
explanation does depend on the weights) while our null test finds nothing
outcome-relevant to explain. passing it is not evidence there is anything to
explain. on behaviour it also flags the informative planted-signal agent (0.996,
barely moved), but that inversion does not survive the change of target (0.46),
so what the check flags depends on which explanation you degrade.

**the span survives off-manifold masking; the ranking does not.** the test
statistic is built from two coalitions that are both on the data manifold, so it
is immune by construction. the per-feature ranking is not: under conditional
masking it correlates at only 0.767 and disagrees on which feature matters most.

### honesty

we applied the paper's own thesis to the paper. it argues that the choice of
null decides the verdict and that nobody checks; so we checked ours, found the
published null was measuring spans on a different corpus from the observation,
rebuilt it the way the method section actually defines, and reported that the
headline survives but only borderline (64% of bootstrap resamples). an earlier
run at n = 12 had the two constructions on opposite sides of the threshold.

the paper carries a ledger of seven predictions this project made and then
rejected on its own evidence, and separates them from the one claim that is
structural rather than empirical. numbers in the paper are not
transcribed by hand: `verify_paper_numbers.py` asserts every one of them against
the artifact that produced it and fails the build on a mismatch.

### running it

```sh
./reproduce.sh          # every figure and number, from public data, no API keys
./reproduce.sh --quick  # reduced budgets, for a smoke test
```

### what is where

this repository holds two bodies of work. the paper uses only the first.

**the research**

| | |
|---|---|
| **[docs/paper/](docs/paper/)** | the paper, its 4-page version, LaTeX source, and anonymised builds |
| [docs/paper/tmlr/main.pdf](docs/paper/tmlr/main.pdf) | the TMLR submission build, generated by `make_tmlr.py` |
| **[services/agent-rl/](services/agent-rl/)** | everything the paper runs: env, PPO agent, attribution, null constructions, 312 tests |
| [reproduce.sh](reproduce.sh) | one command, every figure and number, CPU only |
| `reports/` | json artifacts and figures, regenerated by reproduce.sh |
| [docs/REPORT.md](docs/REPORT.md) | the trading result the research was built on |
| [docs/MDP.md](docs/MDP.md) | the problem formulation |
| [docs/AGENT_NOTES.md](docs/AGENT_NOTES.md) | supporting measurements behind REPORT.md |
| [services/agent-rl/README.md](services/agent-rl/README.md) | what each module and script does |

**the matching engine**, which this repository started as and which the research
was built alongside. unchanged by it.

| | |
|---|---|
| **[docs/ENGINE.md](docs/ENGINE.md)** | architecture, benchmarks, API, how to run the stack |
| `services/engine-go/` | Go order book and matching engine |
| `services/agent-py/`, `services/backtest-py/` | sklearn agent and backtester |
| `services/dashboard-react/`, `services/feed-sim/` | dashboard and feed simulator |
| `infra/`, `shared/` | docker, kafka, postgres, JSON schemas |

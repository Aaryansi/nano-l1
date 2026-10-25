# Independent reference-bank calibration protocol

This is a separate synthetic study. It does not amend the completed `research_v1` release or its paper results. The configuration below is fixed before inspecting this study's full-run outcomes. Execution pilots use reduced budgets and remain separately identified.

## Question and replication

Estimate how often each comparison procedure issues a flag when both the reference bank and the observed agent are drawn afresh. The three procedures are the legacy reference-centered rank/normal rule, the upper-tail raw-span rank and the pooled-symmetric two-sided rank. A flag concerns the specified return intervention, without establishing feature-level fidelity.

Use 20 independent bank replicates, 24 reference agents per bank and 30 observed agents at each signal strength 0, 0.2 and 1. Every agent trains for 40 PPO updates on 1,200 independently generated training episodes, with 1,200 independently generated evaluation episodes. There are 14 steps per episode. The generator uses price 0.5, spread 0.02 and signal noise 0.3. The master seed is 20261024. The full study contains 2,280 trained policies.

Every reference uses the zero-strength population. Observed and reference agents use the same architecture, trainer, normalization, costs, episode budget and endpoint intervention. The only population difference in signal alternatives is the declared strength. Each agent fits its normalizer on its own training draw. Evaluation is greedy. Full-input replacement uses 256 flat-inventory background rows and fresh whole-row draws at every decision. The recorded masking seed is separate from the policy seed, unlike the older release's coupled seeds. Thus this is a newly identified common procedure rather than another bank silently substituted into the old release.

Create separate `SeedSequence` child streams for banks and agents. Record four seeds per agent for training data, evaluation data, policy training and background/masking. Reject reused seeds before training. A seed mismatch is not repaired by selecting a favorable completed run. Disjoint recorded streams support the simulation design; metadata alone does not prove statistical exchangeability or properties of a pseudorandom generator.

Conditional on a bank, all observed trials are generated independently. Marginally they share that bank and cannot be treated as independent Bernoulli replicates. Banks are the independent units for marginal uncertainty. Results assume that separate simulated bank streams represent independent draws from the same declared procedure.

## Outcomes and uncertainty

Retain every bank, strength and trial. Preserve both endpoint-return arrays, episode identities, seeds, corpus fingerprints, fitted normalizers, checkpoints and update logs. Reconstruct all spans, tail comparisons, decisions and conditional rates. Replay saved models before promoting the scientific results. A degenerate legacy reference remains unresolved. An unresolved comparison does not become a successful null check or a resolved non-rejection.

The marginal estimand is the probability that a procedure issues a flag over a new bank and observed policy. For each bank, its flag fraction includes every declared trial in the denominator. This operational flag probability is reported alongside the unresolved probability and the descriptive flag fraction among resolved trials. Conditional, per-bank Wilson intervals retain the resolved denominator for the legacy procedure. The standard ranks can compute p-values without a reference standard deviation; their rank floors and ability to reject at alpha 0.05 remain explicit.

Across banks, average bank-specific flag fractions. Give each bank equal weight. Report their sample SD and a pointwise bounded-mean confidence interval. Do not pool all trials into a binomial interval. Do not use a collapsed bootstrap interval to conclude that a population error probability is zero.

For independent bank fractions `X_b` in [0,1], the classical bounded-variable exponential argument gives a tail bound `exp(-B * KL(x, mu))`. Invert both tails with threshold `log(2/0.05)/B`. The resulting interval uses the number of banks, including when all observed fractions are zero or one. It can be wide with 20 banks. This is an application of established concentration bounds, without a new validity theorem. See [Hoeffding (1963), Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830).

For clarity, convexity gives `E exp(lambda X) <= 1 - mu + mu exp(lambda)` for any bounded `X` in [0,1]. Independence multiplies those bounds. Optimizing the Chernoff bound yields Bernoulli KL, even though a bank fraction need not itself be Bernoulli. A two-tail union bound and monotone inversion give the stated pointwise interval. Paired method differences lie in [-1,1]; apply the same construction to `(difference + 1)/2` and transform back. These intervals are not simultaneous across methods, strengths or contrasts.

The signal alternatives measure this finite-budget learning-and-comparison procedure together. They do not isolate the comparison rule's sensitivity from learning success. Neither observed rates nor the concentration bound establish exchangeability for market policies or a universal explanation null.

## Execution and review

From the repository root, with the pinned research environment:

```sh
PY=services/agent-rl/.venv/bin/python
"$PY" services/agent-rl/scripts/calibrate_reference_banks.py \
  --out runs/reproductions/reference-banks-smoke --quick --seed 17
"$PY" services/agent-rl/scripts/calibrate_reference_banks.py \
  --out runs/reproductions/reference-banks-smoke --verify-only --allow-quick --replay-models
```

The full command uses committed, clean source and the declared defaults:

```sh
"$PY" services/agent-rl/scripts/calibrate_reference_banks.py \
  --out runs/reproductions/reference-banks-v1 --jobs 4
"$PY" services/agent-rl/scripts/calibrate_reference_banks.py \
  --out runs/reproductions/reference-banks-v1 --verify-only --replay-models
```

Two isolated spawned workers run banks by default, with one numerical thread per worker. The full command above requests four workers. `--jobs 1` runs serially. Verify serial/spawn equality with execution budgets before the full run. Record source, configuration, package versions and runtime before training, require them unchanged at completion, and hash every generated artifact. Refuse nonempty outputs and writes inside released evidence or manuscript directories. A started, incomplete, dirty full or altered run cannot pass verification. The default verifier rejects reduced/custom budgets unless `--allow-quick` explicitly limits the result to execution or pilot evidence.

Review support code and execution evidence in one scoped PR. After merge, freeze the full-run source and complete the study in its own ignored directory. Results and any manuscript promotion require a separate review. Do not append new results to an immutable completed release.

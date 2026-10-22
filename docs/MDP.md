# Implemented market decision process

This document specifies the current simulator. It describes an offline research model, with execution and cost assumptions, rather than verified live trading conditions. The evaluation and inference protocol is in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Episodes and data

An episode represents one binary contract with settlement Y in {0, 1}. The pinned release contains 6,425 episodes with 14 decision rows each. Chronological splitting assigns 3,854 training, 1,283 validation and 1,284 test episodes, with four episodes purged around split boundaries. Exact identifiers, corpus hashes and split membership are recorded under [market_snapshot_v1](../data/releases/market_snapshot_v1/). The reproduction runner verifies them before training.

The simulator reads aligned bid, ask, last-price, volume, staleness, flow, time and spot arrays. Market features use backward differences and trailing windows. Candle-derived quote inputs are coarse proxies; they do not establish an executable order book or available depth. The agent does not observe settlement before termination.

## Observation and actions

The observation has 18 coordinates. The first 14 contain nine contract features and five spot features, listed in `nano_rl/env/features.py`. Their means and standard deviations are fitted only on training market rows and frozen for evaluation. Synthetic agents retain their own training normalizer. The last four coordinates are recomputed from the active policy's inventory at each step:

| Coordinate | Definition |
| --- | --- |
| position | signed contracts q divided by cap Q |
| avg_entry_price | average entry price e, zero while flat |
| unrealized_pnl | q(P - e) / Q at the current mid P |
| time_in_position | elapsed holding steps divided by episode step count, capped at one |

The position block is not standardized with market statistics. Its clock starts at zero on opening or direction change, advances on holding, adding or reducing, and resets when flat. Scalar and vector implementations share this update rule and are tested for trajectory parity.

Actions choose target inventory {-Q, 0, +Q}; the default Q is 100 contracts. An action trades the difference from current inventory, so reversing direction can trade 200 contracts. This is a simplified signed yes-equivalent accounting model, with no funding, margin, queue priority or partial-fill simulation.

## Execution, costs and terminal convention

With costs enabled, a buy uses the supplied ask and a sell the bid. For each nonzero order of size C at execution price p, the assumed fee is `ceil_to_cents(0.07 * abs(C) * p * (1-p))`. Rounding occurs once per order. Optional impact is linear in order size divided by bar volume; its default coefficient is zero. No claim is made here that this coefficient or these fill assumptions reproduce the exact historical fee schedule or liquidity for every contract.

Cash changes by `-delta_q * execution_price - fee`. Before termination, equity is cash plus position times the next row's mid. At termination, the default mark is the recorded settlement Y, with no additional settlement fee in the model. The `force_close_at_last_quote` alternative liquidates at the final quote before terminal marking. It changes the mark and incurs execution costs, so that contrast does not isolate a settlement fee.

The zero-cost ablation executes at mid and charges no fees or spread crossing. Independently retraining under this objective changes the learning problem. A return difference cannot uniquely separate predictability, optimization limits and transaction-cost effects.

## Reward and critic

The default reward is the change in marked equity after each action. At gamma=1, its sum telescopes to final equity minus initial equity. The market PPO configuration uses gamma=1. This connects the dense training reward to terminal profit and loss; it does not establish that PPO finds an optimal policy.

The critic estimates expected policy return in reward units, dollars for the default objective. It does not predict settlement probability. Its predictions are checked against stochastic policy rollouts, with Monte Carlo uncertainty. The separate objective-control experiment compares this financial objective with a direction-prediction reward using the same observation and transition setup, and retains all fixed budgets and seeds.

## Explanation interventions

The return endpoint span is the difference between return with original policy inputs and return under a specified replacement intervention. Full replacement redraws a background row each decision step. The empirical background is sampled with flat inventory, an explicit baseline choice. Market-row and donor-trajectory controls retain the recipient's true inventory and evaluation clock while replacing selected market inputs.

A donor row can lie on empirical row support while being inconsistent with the recipient's inventory, history or reward. A donor trajectory preserves its own market sequence, not the joint process of the recipient. Nearest-neighbor replacement approximates a conditional distribution without establishing its correctness.

Retaining all inventory inputs also retains some market information. For nonzero scaled position z = q/Q and scaled unrealized profit u = q(P-e)/Q, the current mid satisfies P = e + u/z. Thus these controls cannot simultaneously preserve the full financial state and hide all price information.

Shapley efficiency makes the sum of feature contributions equal the endpoint span for the chosen characteristic function. It does not make that sum a measure of feature-level fidelity. A faithful explanation of an unprofitable policy can remain meaningful.

## Reference populations and inference limits

The synthetic zero-strength fixture has balanced independent settlements and default constant price 0.5. At those defaults, predictable trading has no positive population edge before costs, while flat earns zero. A nondefault price can produce a directional base-rate edge even with zero feature signal. The planted fixture supplies repeated noisy signals. The first-observation sign-and-hold policy has a directly computed realized payoff, but is not a proven optimum when aggregation, delayed entry and abstention are possible.

Synthetic training and evaluation draws are independent. The historical market evaluation is conditional on a fixed time window. Low residual correlations, non-rejection and finite training failures do not establish market efficiency or absence of exploitable information.

Global settlement shuffling preserves the feature arrays and marginal outcome rate but changes price-settlement and reward relationships. It is not equivalent to removing incremental trading information. Conditional randomization may legitimately retain information in the conditioning price; its validity requires a justified joint conditional law, especially for sequential data.

The legacy rank/normal rule is a diagnostic without a general exchangeability guarantee. Under its own convention, insufficient or zero-scale references remain unresolved. Standard rank comparators are reported separately with explicit assumptions; their guarantee is marginal over exchangeable observed and reference draws. Trial rates against one reused bank remain conditional on that bank. Market exchangeability with synthetic references is not established.

Primary market uncertainty uses training-seed and common UTC-day resampling. Pairwise historical comparisons use common UTC-day blocks. These estimates are limited by the number of independent time blocks; non-rejection does not establish equivalence. Control-task interpretation additionally requires measured competence, observation dependence and a resolved reference. All failures remain part of the results.

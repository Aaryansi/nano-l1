# mdp design: rl trading on kalshi 15-minute btc binaries

phase 1 deliverable. design only, no code. every quantitative claim below was
verified against the live api before being written down; the verification
commands live in `services/agent-rl/scripts/` once phase 2 lands.

---

## 1. instrument

**series**: `KXBTC15M` on Kalshi, titled "BTC price up in next 15 mins?"

a binary contract that settles to $1.00 if BTC spot is higher at the close of a
15-minute window than at its open, and $0.00 otherwise. price is quoted in
dollars on [0, 1] and is directly interpretable as the market's implied
probability of the event.

### why this instrument rather than spot crypto or equities

1. **natural episode boundaries with terminal ground truth.** each contract
   opens and closes exactly 900 seconds apart and then resolves to a known 0/1
   value. this is the textbook finite-horizon episodic mdp, rather than a
   continuing task chopped into arbitrary slices, which is how most rl-for-
   trading work is forced to handle spot markets.

2. **the value function becomes falsifiable.** because every episode has a
   realized binary outcome, a critic's `V(s)` can be checked against the
   empirical frequency of resolution for states with similar predicted value.
   that is a reliability diagram for the value head. this is the single
   strongest reason to prefer this instrument: it turns "explain the value
   predictions" (phase 5c) from a qualitative exercise into a calibration
   measurement with ground truth.

3. **semantically auditable features.** time-to-expiry, implied probability,
   and spot-distance-from-strike produce Shapley attributions a human can
   audit. "the agent sold because time to expiry collapsed while spot sat below
   the strike" is checkable. "orderbook imbalance lag-3 contributed +0.02" is
   not.

4. **bounded price and bounded loss.** price on [0, 1] needs no return
   normalization and the maximum loss per contract is known in advance, which
   removes a class of accounting bugs.

---

## 2. data sources (verified)

| source | endpoint / file | auth | what it gives | status |
|---|---|---|---|---|
| Kalshi markets | `/trade-api/v2/markets?series_ticker=KXBTC15M&status=settled` | none | episode boundaries, settlement result | verified: 2,400 markets in 12 pages / 5.0s, spanning 25.3 days, pagination not exhausted |
| Kalshi trades | `/trade-api/v2/markets/trades?ticker=...` | none | per-trade tape: ts, yes price, size, taker side | verified: 20,000+ trades on a single market |
| Kalshi candlesticks | `/trade-api/v2/series/KXBTC15M/markets/{t}/candlesticks` | none | 1-min ohlc of price, `yes_bid`, `yes_ask`, volume, open interest | verified: 15 candles per episode, bid/ask present |
| Binance spot | `data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/*.zip` | none | underlying tick tape | verified: http 200, 353 MB/month |

all four are free and require no api key or account.

measured properties of the Kalshi tape:

- exactly **one market per 15-minute window**, 96 per day, every episode 900s
- outcomes near balanced: **105 yes / 95 no** over a 200-market sample, so no
  degenerate majority class
- trade density: **68% of 10-second buckets contain at least one trade**, median
  **327 trades per active bucket**
- measured spread over one full episode: **0.0100 for 9 of 15 minutes**,
  tightening to 0.0010–0.0020 only in the last third as the contract pins. the
  modal spread is **1 cent**, not the 0.001 best case. friction estimates below
  use the modal value.
- sampled episode opened at **exactly 0.5000**, which is the coin-flip prior the
  contract construction implies
- terminal integrity: sampled market resolved `yes`, `settlement_value = 1.0000`,
  last traded price 0.999, so tape and resolution agree

---

## 3. episode

one episode = one market's full life.

- `t = 0` at market open, `t = T` at market close, 900 seconds later
- terminal state is reached at market close, always, with probability 1
- terminal payoff is the true settlement value in {0, 1}, not the last traded
  price

episodes are the atomic unit of the train/val/test split (see §8). no episode
straddles a split boundary, which would leak.

---

## 4. time discretization

**decision step = 10 seconds, so 90 steps per episode.**

justification: the measured density of 327 trades per active 10s bucket means a
10s bar is well populated, while 68% bucket coverage means roughly a third of
bars have no trade at all and must be forward-filled. that gap is not hidden, it
is exposed to the agent as an explicit `staleness` feature (§5).

alternatives rejected:
- **1-minute steps** (15 per episode) align with the candlestick endpoint but
  give only ~36k decision steps across the corpus and too few decisions per
  episode for credit assignment to be interesting.
- **event-based steps** (one step per trade) match the data-generating process
  but produce variable-length episodes, complicate the discount and the
  attribution comparison, and are not worth the added machinery at this stage.

spread is sourced from the 1-minute candlestick containing the step and held
constant within that minute. this is an approximation and is recorded as such
in the limitations section of the report.

---

## 5. state space

all features are **strictly causal**: computed only from data with timestamp
`<= t`. the Binance spot series is joined to the Kalshi clock with a backward
(as-of) join, never forward.

### market features (Kalshi)

| # | feature | definition |
|---|---|---|
| 1 | `implied_prob` | last traded yes price at or before `t`, in [0, 1] |
| 2 | `spread` | `yes_ask - yes_bid` from the containing 1-min candle |
| 3 | `p_change_30s` | `implied_prob(t) - implied_prob(t-30s)` |
| 4 | `p_change_60s` | `implied_prob(t) - implied_prob(t-60s)` |
| 5 | `p_realized_vol` | stdev of `implied_prob` over trailing 60s |
| 6 | `flow_imbalance` | (yes-taker volume − no-taker volume) / total, trailing 60s |
| 7 | `volume_rate` | contracts traded in trailing 30s, log1p scaled |
| 8 | `staleness` | seconds since last observed trade, capped and scaled |

### underlying features (Binance BTC spot)

| # | feature | definition |
|---|---|---|
| 9 | `spot_ret_since_open` | `S_t / S_open − 1`; the contract resolves on the **sign** of this at `T` |
| 10 | `spot_ret_30s` | short-horizon spot momentum |
| 11 | `spot_ret_60s` | short-horizon spot momentum |
| 12 | `spot_realized_vol` | trailing 60s realized vol of spot |
| 13 | `spot_implied_gap` | z-scored `spot_ret_since_open` minus `(implied_prob − 0.5)`, scaled; a model-free lead-lag signal |

feature 13 encodes the project's one genuinely testable alpha hypothesis: that
Kalshi's implied probability **lags** the Binance spot move. it is causal and
cheap to compute. if the hypothesis is false, the evaluation will show it, and
the report will say so.

### time

| # | feature | definition |
|---|---|---|
| 14 | `time_to_expiry_frac` | `(T − t) / T`, in [0, 1] |

a binary's sensitivity to the underlying is extremely time-dependent; near
expiry the price pins to 0 or 1. this feature is not optional.

### position / inventory (position-aware state, as required)

| # | feature | definition |
|---|---|---|
| 15 | `position` | signed yes-equivalent contracts held, normalized by `q_max` |
| 16 | `avg_entry_price` | volume-weighted entry price of the open position, 0 if flat |
| 17 | `unrealized_pnl` | mark-to-market pnl of the open position, normalized |
| 18 | `time_in_position` | steps since position last changed sign or opened |

**18 features.** small enough that exact permutation Shapley over feature
subsets is tractable with sampling, which matters for phase 5.

normalization statistics are fit on the **training split only** and applied
unchanged to val and test.

---

## 6. action space

**discrete target position**: `a in {-1, 0, +1}`, interpreted as a target
inventory of `a * q_max` yes-equivalent contracts.

the environment computes the delta from current to target position and executes
that delta, charging cost on `|delta|` only.

### why target-position rather than buy/sell/hold

- **position limits become structural, not a constraint to enforce.** an agent
  cannot accumulate unbounded inventory, because the action names the level, not
  an increment.
- **no-op is well defined.** choosing the current position costs nothing, so
  "hold" is a genuine free action rather than an increment of zero that still
  has to be special-cased.
- **cleaner credit assignment.** the action determines the next state's position
  deterministically, so the effect of an action on inventory is not confounded
  by history. this materially simplifies the trajectory-aware attribution in
  phase 5b.
- **churn is penalized automatically**, since cost is charged on the delta.

on Kalshi a short-yes position is implemented as a long-no position; the two are
economically identical for a binary. the environment tracks signed
yes-equivalent inventory and maps to the concrete no-side order at execution
time. this mapping will be unit-tested in phase 2.

---

## 7. reward

$$ r_t = \big(V_t - V_{t-1}\big) - c_t $$

where `V_t` is mark-to-market equity at step `t` and `c_t` is the transaction
cost incurred at step `t` (§7.2).

at the terminal step, the position is marked at the **true settlement value**
in {0, 1}, not at the last traded price.

### 7.1 why change-in-mark-to-market rather than terminal-only

the rewards **telescope exactly**:

$$ \sum_{t=1}^{T} r_t = V_T - V_0 - \sum_t c_t $$

which is precisely net episode pnl. so the dense per-step reward is not a
heuristic shaping term that changes the optimal policy; it is return-equivalent
to the sparse terminal reward at `gamma = 1`, while giving the learner a signal
at every one of the 90 steps instead of one signal per episode.

**discount factor `gamma = 1`.** episodes are finite, short (90 steps), and
always terminate. discounting would bias the agent toward early profit with no
economic justification, and would break the exact telescoping property above.

### 7.2 frictions, and why they dominate this problem

**fees.** Kalshi's taker fee, confirmed against two independent sources:

$$ \text{fee} = \lceil\, 0.07 \times C \times P \times (1-P) \,\rceil_{\text{cent}} $$

with `C` contracts and `P` the price in dollars, rounded up to the next cent per
order. a maker formula with a smaller constant exists but is series-dependent
and is **not** assumed here; modelling the agent as a pure taker is the
conservative choice.

this formula is maximized at `P = 0.5`, where `P(1-P) = 0.25`:

| quantity | value at P = 0.50 |
|---|---|
| contract notional | 50 cents |
| fee, one way | 1.75 cents = **3.5% of notional** |
| fee, round trip | 3.50 cents = **7.0% of notional** |
| spread cost, round trip (modal spread 0.01) | 1.00 cent = **2.0% of notional** |
| **total round-trip friction** | **4.5 cents = 9.0% of notional** |

**this is the single most important fact about the problem.** a
"BTC up in 15 minutes?" contract trades near 0.50 almost by construction, and
that is exactly where the fee formula is maximal. fees are roughly **3.5x** the
spread cost, and the two together impose a **9% round-trip hurdle**. a
high-turnover taker strategy cannot be profitable here unless it forecasts a
near-coin-flip with an edge exceeding 9% per round trip, which is not plausible.

this reframes what success means for the project (§9).

**slippage.** modelled as crossing the measured spread: buys execute at
`yes_ask`, sells at `yes_bid`. a size-proportional impact term is applied when
order size is large relative to observed volume in the bar. both assumptions are
stated in the report rather than buried.

### 7.3 risk-adjusted variant

a variance-penalized reward `r_t - lambda * (r_t - \bar{r})^2` will be run as an
**ablation**, not as the primary objective. reporting a single primary objective
avoids the temptation to select whichever variant looked best on test.

---

## 8. walk-forward split

splits are **contiguous in time and never shuffled**. the unit is the episode.

```
|<----------- train ----------->|gap|<-- val -->|gap|<-- test -->|
         ~60%                          ~20%             ~20%
```

- a **purge gap** of at least one episode sits between adjacent splits, so that
  trailing-window features computed near a boundary cannot see across it.
- **test is touched exactly once**, after all tuning is frozen on val.
- target corpus ~60 days (~5,760 episodes). 25.3 days is verified available and
  pagination was not exhausted; confirming the full depth is the first task of
  phase 2, and if the corpus is smaller than targeted the split ratios hold and
  the reduced sample size is reported.

a rolling-origin multi-fold variant is a stretch goal, reported only if the
single-split result is already sound.

---

## 9. what counts as success

given §7.2, the honest expected outcome is that **no policy, learned or
baseline, is profitable net of fees on this contract**. the project is designed
so that this is a publishable finding rather than a failure:

1. **the primary scientific claim is about explanation, not profit.** the
   deliverable is a Shapley attribution layer that engages with temporal credit
   assignment, evaluated on an mdp with real terminal ground truth.

2. **a correct agent should learn to abstain.** facing a 9% round-trip cost, the
   optimal policy is close to "hold flat unless conviction is extreme". if the
   agent learns that, it is evidence the learning loop works, and the
   interesting explainability question becomes *why it abstains* and *which
   features drive the rare decisions to trade*.

3. **the value head is separately checkable.** even a policy that never trades
   profitably can have a well-calibrated `V(s)`, and calibration is measurable
   against realized settlement (§1.2). a well-calibrated critic paired with an
   unprofitable policy is itself an honest, interesting result.

4. **a zero-cost ablation isolates the source of failure.** running the same
   agent with fees set to zero separates "the agent cannot predict" from "the
   agent can predict but not enough to cover costs". these are very different
   findings and the report will distinguish them.

---

## 10. no-lookahead guarantees

to be enforced in code and asserted in tests during phase 2:

1. every feature at step `t` reads only rows with timestamp `<= t`
2. Binance spot is joined as-of backward; a forward join fails the test suite
3. normalization statistics are fit on train only
4. splits are contiguous in time with purge gaps; episodes never straddle
5. settlement value is visible **only** at the terminal transition, never as a
   feature
6. a dedicated test shuffles future data and asserts that in-sample features are
   bit-identical, which catches accidental forward leakage

---

## 11. open questions and risks

| risk | mitigation |
|---|---|
| corpus depth beyond 25.3 days unverified | first task of phase 2; split ratios are depth-independent |
| spread held constant within each 1-minute candle | approximation, stated in report; per-trade quote reconstruction is a stretch goal |
| maker fee schedule is series-dependent and unconfirmed | modelled as pure taker, which is strictly conservative |
| lead-lag hypothesis (feature 13) may simply be false | this is a result, not a blocker; reported either way |
| Kalshi api rate limits not formally documented | observed 0.4s/market with politeness sleeps; ingest is cached to disk so it runs once |
| 18 features is enough to make exact Shapley exponential | permutation sampling with reported standard errors, not exact enumeration |

---

## 12. what phase 2 will build against this

a Gymnasium-style environment implementing exactly the above, plus a test suite
asserting: reward telescoping to net pnl, fee formula correctness at several
prices including the `P = 0.5` maximum, position limits, the yes/no short
mapping, terminal settlement marking, and the six no-lookahead properties in
§10. tests pass before any agent code is written.

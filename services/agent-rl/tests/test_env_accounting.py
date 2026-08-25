"""environment accounting tests.

the reward telescoping property is the load-bearing claim of the mdp design
(docs/MDP.md section 7.1): it is what makes the dense per-step reward
return-equivalent to the sparse terminal reward rather than a shaping hack.
if it fails, every reported return is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.env.binary_market import ACTION_TO_TARGET, BinaryMarketEnv
from nano_rl.env.costs import CostModel, fee_dollars
from nano_rl.env.features import PositionState

from .conftest import make_batch

LONG, FLAT, SHORT = 2, 1, 0


def run_policy(env: BinaryMarketEnv, actions: list[int], episode: int = 0):
    """drive an env through a fixed action sequence, returning rewards + info."""
    env.reset(seed=0, options={"episode": episode})
    rewards, infos = [], []
    for a in actions:
        _, r, term, _, info = env.step(a)
        rewards.append(r)
        infos.append(info)
        if term:
            break
    return rewards, infos


class TestTelescoping:
    """sum(rewards) must equal final equity, exactly, in every scenario."""

    @pytest.mark.parametrize(
        "actions",
        [
            [FLAT] * 10,
            [LONG] * 10,
            [SHORT] * 10,
            [LONG, FLAT, SHORT, FLAT, LONG, LONG, SHORT, FLAT, LONG, FLAT],
            [LONG, SHORT] * 5,  # maximal churn
        ],
    )
    def test_rewards_sum_to_final_equity(self, flat_batch, actions) -> None:
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        rewards, infos = run_policy(env, actions)
        assert sum(rewards) == pytest.approx(infos[-1]["equity"], abs=1e-9)

    def test_holds_on_a_moving_market(self, rising_batch) -> None:
        env = BinaryMarketEnv(rising_batch, random_episode_order=False)
        rewards, infos = run_policy(env, [LONG] * 10)
        assert sum(rewards) == pytest.approx(infos[-1]["equity"], abs=1e-9)

    def test_holds_with_costs_disabled(self, flat_batch) -> None:
        env = BinaryMarketEnv(
            flat_batch, cost_model=CostModel(enabled=False), random_episode_order=False
        )
        rewards, infos = run_policy(env, [LONG, SHORT, FLAT, LONG] + [FLAT] * 6)
        assert sum(rewards) == pytest.approx(infos[-1]["equity"], abs=1e-9)

    def test_initial_equity_is_zero(self, flat_batch) -> None:
        """sum(rewards) == final equity only holds if we start at zero."""
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        env.reset(seed=0, options={"episode": 0})
        assert env._prev_equity == 0.0


class TestTerminalSettlement:
    """the terminal mark must be the true settlement, not the last price."""

    def test_long_into_yes_settlement_marks_at_one(self) -> None:
        # mid pinned at 0.60 but the contract resolves YES, so a long position
        # must be marked at 1.00, not 0.60.
        batch = make_batch(
            n_episodes=1, n_steps=5, mid_path=np.full((1, 5), 0.60), settlement=[1.0]
        )
        env = BinaryMarketEnv(batch, random_episode_order=False)
        _, infos = run_policy(env, [LONG] * 5)
        assert infos[-1]["mark"] == pytest.approx(1.0)

    def test_long_into_no_settlement_marks_at_zero(self) -> None:
        batch = make_batch(
            n_episodes=1, n_steps=5, mid_path=np.full((1, 5), 0.60), settlement=[0.0]
        )
        env = BinaryMarketEnv(batch, random_episode_order=False)
        _, infos = run_policy(env, [LONG] * 5)
        assert infos[-1]["mark"] == pytest.approx(0.0)
        # bought at the ask and it expired worthless: strictly negative.
        assert infos[-1]["equity"] < 0

    def test_holding_to_expiry_beats_liquidating(self) -> None:
        """holding to expiry avoids the closing leg of the fee AND the spread.

        this must compare against liquidation at the last MARKET quote. closing
        at the settlement price would be free regardless, because the fee's
        P*(1-P) term vanishes at P in {0, 1}, which would make the comparison
        vacuously equal.
        """
        batch = make_batch(
            n_episodes=1, n_steps=5, mid_path=np.full((1, 5), 0.50), settlement=[1.0]
        )
        held = BinaryMarketEnv(
            batch, force_close_at_last_quote=False, random_episode_order=False
        )
        closed = BinaryMarketEnv(
            batch, force_close_at_last_quote=True, random_episode_order=False
        )

        _, held_info = run_policy(held, [LONG] * 5)
        _, closed_info = run_policy(closed, [LONG] * 5)

        assert held_info[-1]["fees"] < closed_info[-1]["fees"]
        assert held_info[-1]["equity"] > closed_info[-1]["equity"]
        # the forced close must actually flatten the book
        assert closed_info[-1]["position"] == pytest.approx(0.0)

    def test_closing_at_settlement_price_would_be_free(self) -> None:
        """documents why the flag closes at the market quote, not settlement.

        pinned as a regression test: an earlier implementation closed at the
        settlement mark and was a silent no-op.
        """
        assert fee_dollars(100, 1.0) == 0.0
        assert fee_dollars(100, 0.0) == 0.0

    def test_flat_at_expiry_is_unaffected_by_settlement(self) -> None:
        """a flat agent's pnl cannot depend on the resolution."""
        for settle in (0.0, 1.0):
            batch = make_batch(
                n_episodes=1, n_steps=5, mid_path=np.full((1, 5), 0.5), settlement=[settle]
            )
            env = BinaryMarketEnv(batch, random_episode_order=False)
            rewards, infos = run_policy(env, [FLAT] * 5)
            assert sum(rewards) == pytest.approx(0.0)
            assert infos[-1]["equity"] == pytest.approx(0.0)


class TestCostsAreApplied:
    def test_flat_policy_pays_nothing(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        rewards, infos = run_policy(env, [FLAT] * 10)
        assert infos[-1]["fees"] == 0.0
        assert infos[-1]["trades"] == 0
        assert sum(rewards) == pytest.approx(0.0)

    def test_churn_loses_money_on_a_flat_market(self, flat_batch) -> None:
        """the central economic fact that makes abstention correct."""
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        rewards, infos = run_policy(env, [LONG, SHORT] * 5)
        assert sum(rewards) < 0
        assert infos[-1]["fees"] > 0
        assert infos[-1]["trades"] == 10

    def test_more_churn_costs_strictly_more(self, flat_batch) -> None:
        env_a = BinaryMarketEnv(flat_batch, random_episode_order=False)
        env_b = BinaryMarketEnv(flat_batch, random_episode_order=False)
        _, low = run_policy(env_a, [LONG] + [FLAT] * 9)
        _, high = run_policy(env_b, [LONG, SHORT] * 5)
        assert high[-1]["fees"] > low[-1]["fees"]

    def test_zero_cost_ablation_makes_churn_free(self, flat_batch) -> None:
        """with frictions off, churning a flat market and ENDING FLAT nets zero.

        the sequence must finish flat. ending with an open position leaves a
        settlement payoff in the result, which is real pnl and not a cost
        artefact: on this fixture (mid 0.50, settles 0.0) an open short would
        correctly earn +50.
        """
        env = BinaryMarketEnv(
            flat_batch, cost_model=CostModel(enabled=False), random_episode_order=False
        )
        rewards, infos = run_policy(env, [LONG, SHORT] * 4 + [FLAT, FLAT])
        assert infos[-1]["position"] == pytest.approx(0.0)
        assert sum(rewards) == pytest.approx(0.0, abs=1e-9)
        assert infos[-1]["fees"] == 0.0

    def test_open_short_earns_settlement_when_contract_expires_worthless(
        self, flat_batch
    ) -> None:
        """the flip side of the above, asserted rather than left implicit.

        short 100 at mid 0.50 into a contract that settles at 0.00 is a
        genuine +50, and the env must report it.
        """
        env = BinaryMarketEnv(
            flat_batch, cost_model=CostModel(enabled=False), random_episode_order=False
        )
        rewards, infos = run_policy(env, [SHORT] * 10)
        assert infos[-1]["position"] == pytest.approx(-100.0)
        assert sum(rewards) == pytest.approx(50.0, abs=1e-9)

    def test_fee_matches_the_cost_model(self, flat_batch) -> None:
        """one round trip must charge exactly two hand-computed fees."""
        env = BinaryMarketEnv(flat_batch, max_position=100.0, random_episode_order=False)
        _, infos = run_policy(env, [LONG, FLAT] + [FLAT] * 8)
        # buy 100 at the ask (0.51), then sell 100 at the bid (0.49)
        expected = fee_dollars(100, 0.51) + fee_dollars(100, 0.49)
        assert infos[-1]["fees"] == pytest.approx(expected)


class TestPositionLimits:
    def test_position_never_exceeds_max(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, max_position=100.0, random_episode_order=False)
        rng = np.random.default_rng(7)
        env.reset(seed=0, options={"episode": 0})
        for _ in range(10):
            _, _, term, _, info = env.step(int(rng.integers(0, 3)))
            assert abs(info["position"]) <= 100.0 + 1e-9
            if term:
                break

    def test_repeated_long_does_not_accumulate(self, flat_batch) -> None:
        """target-position semantics: holding LONG must not keep buying."""
        env = BinaryMarketEnv(flat_batch, max_position=100.0, random_episode_order=False)
        _, infos = run_policy(env, [LONG] * 10)
        assert infos[-1]["position"] == pytest.approx(100.0)
        # one trade to open, none thereafter
        assert infos[-1]["trades"] == 1

    def test_all_actions_reachable(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, max_position=100.0, random_episode_order=False)
        env.reset(seed=0, options={"episode": 0})
        seen = set()
        for a in (LONG, SHORT, FLAT):
            _, _, _, _, info = env.step(a)
            seen.add(round(info["position"], 6))
        assert seen == {100.0, -100.0, 0.0}


class TestPositionStateMath:
    """average-entry bookkeeping, especially across a flip through zero."""

    def test_open_sets_entry(self) -> None:
        p = PositionState()
        p.apply_fill(10, 0.40)
        assert p.position == 10
        assert p.avg_entry_price == pytest.approx(0.40)

    def test_adding_blends_entry(self) -> None:
        p = PositionState()
        p.apply_fill(10, 0.40)
        p.apply_fill(10, 0.60)
        assert p.avg_entry_price == pytest.approx(0.50)

    def test_full_close_resets(self) -> None:
        p = PositionState()
        p.apply_fill(10, 0.40)
        realized = p.apply_fill(-10, 0.60)
        assert p.position == 0
        assert p.avg_entry_price == 0.0
        assert realized == pytest.approx(2.0)  # 10 * (0.60-0.40)

    def test_flip_through_zero_resets_entry_to_fill_price(self) -> None:
        """a classic pnl bug: keeping the old entry after flipping sides."""
        p = PositionState()
        p.apply_fill(10, 0.40)
        p.apply_fill(-30, 0.60)
        assert p.position == pytest.approx(-20)
        assert p.avg_entry_price == pytest.approx(0.60)

    def test_unrealized_is_signed_correctly(self) -> None:
        p = PositionState()
        p.apply_fill(10, 0.40)
        assert p.unrealized_pnl(0.50) == pytest.approx(1.0)
        p2 = PositionState()
        p2.apply_fill(-10, 0.40)
        # short gains when price falls
        assert p2.unrealized_pnl(0.30) == pytest.approx(1.0)

    def test_flat_has_no_unrealized(self) -> None:
        assert PositionState().unrealized_pnl(0.7) == 0.0


class TestApiShape:
    def test_reset_returns_obs_and_info(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "episode" in info

    def test_episode_terminates_exactly_at_n_steps(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        env.reset(seed=0, options={"episode": 0})
        for i in range(flat_batch.n_steps):
            _, _, term, trunc, _ = env.step(FLAT)
            assert term == (i == flat_batch.n_steps - 1)
            assert not trunc

    def test_obs_is_finite_everywhere(self, rising_batch) -> None:
        env = BinaryMarketEnv(rising_batch, random_episode_order=False)
        obs, _ = env.reset(seed=0, options={"episode": 0})
        assert np.all(np.isfinite(obs))
        for a in [LONG, SHORT, FLAT] * 4:
            obs, _, term, _, _ = env.step(a)
            assert np.all(np.isfinite(obs))
            if term:
                break

    def test_action_space_size_matches_targets(self, flat_batch) -> None:
        env = BinaryMarketEnv(flat_batch, random_episode_order=False)
        assert env.action_space.n == len(ACTION_TO_TARGET)

    def test_seeded_reset_is_reproducible(self, flat_batch) -> None:
        a = BinaryMarketEnv(flat_batch)
        b = BinaryMarketEnv(flat_batch)
        oa, ia = a.reset(seed=123)
        ob, ib = b.reset(seed=123)
        assert ia["episode"] == ib["episode"]
        np.testing.assert_array_equal(oa, ob)

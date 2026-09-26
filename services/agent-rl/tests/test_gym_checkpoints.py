"""snapshotting a training run must not change the run.

this pins a bug that was live in the reported gym experiment. checkpoints were
taken by constructing a fresh ActorCritic and calling load_state_dict on it.
constructing one draws from the global torch RNG to initialise weights that are
then immediately overwritten, and those draws shift the stream that net.act()
samples actions from. so asking for four snapshots changed the trajectory being
snapshotted: at the same seed, a checkpointed run and an unsnapshotted one
diverged after the first snapshot, by up to 0.04 in weight terms.

the consequence was subtle and bad. "the agent at 100% of training, seed 0" was
not the agent you get by training with seed 0, so the checkpoint-based table and
any independently trained agent at the same seed were not comparable, and
nothing in the pipeline could have revealed it.
"""

from __future__ import annotations

import numpy as np
import torch

from nano_rl.envs.gym_null import GymPPOConfig, make_env, train_gym_ppo

STEPS = 6000
FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def flat(net) -> np.ndarray:
    return torch.cat([p.flatten() for p in net.parameters()]).detach().numpy()


def train(fractions=(), seed=0):
    env = make_env("CartPole-v1")
    try:
        return train_gym_ppo(
            env, GymPPOConfig(seed=seed), total_steps=STEPS,
            checkpoint_fractions=fractions,
        )
    finally:
        env.close()


class TestTrainingIsDeterministic:
    def test_the_same_seed_gives_the_same_agent(self):
        a, _ = train()
        b, _ = train()
        assert np.allclose(flat(a), flat(b))

    def test_different_seeds_give_different_agents(self):
        a, _ = train(seed=0)
        b, _ = train(seed=1)
        assert not np.allclose(flat(a), flat(b))


class TestSnapshottingDoesNotPerturbTheRun:
    def test_asking_for_checkpoints_does_not_change_the_final_agent(self):
        plain, _ = train()
        snapped, _ = train(FRACTIONS)
        assert np.allclose(flat(plain), flat(snapped)), (
            "taking snapshots changed the trajectory. the snapshot must not "
            "consume randomness the training loop goes on to use."
        )

    def test_the_number_of_snapshots_does_not_change_it_either(self):
        few, _ = train((1.0,))
        many, _ = train((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        assert np.allclose(flat(few), flat(many))

    def test_every_requested_fraction_is_returned(self):
        _, cps = train(FRACTIONS)
        assert [f for f, _ in cps] == list(FRACTIONS)

    def test_the_last_snapshot_is_the_final_agent(self):
        final, cps = train(FRACTIONS)
        assert np.allclose(flat(cps[-1][1]), flat(final))

    def test_snapshots_are_independent_copies(self):
        # a snapshot that aliased the live net would make every checkpoint
        # identical to the final agent, which would silently turn the
        # progress sweep into four copies of one row.
        _, cps = train(FRACTIONS)
        early, late = flat(cps[0][1]), flat(cps[-1][1])
        assert not np.allclose(early, late)

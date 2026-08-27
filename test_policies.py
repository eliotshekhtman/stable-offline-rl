import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import policies
from offlinerlkit.policy import MOBILEPolicy


class DummyEnv:
    def __init__(self, env_id: str):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self.spec = SimpleNamespace(id=env_id)


class ScalarActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.zeros(1))


class ConstantCritic(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, observations, actions):
        return self.value.expand(len(observations), 1)


class ZeroPenaltyMOBILE(MOBILEPolicy):
    def compute_lcb(self, observations, actions):
        return torch.zeros((len(observations), 1))

    def actforward(self, observations, deterministic=False):
        actions = self.actor.action.expand(len(observations), 1)
        return actions, actions * 0.0


def make_constant_mobile(value: float, return_shift: float, clamp_target_q: bool):
    actor = ScalarActor()
    critics = nn.ModuleList([ConstantCritic(value), ConstantCritic(value)])
    return ZeroPenaltyMOBILE(
        dynamics=object(),
        actor=actor,
        critics=critics,
        actor_optim=torch.optim.SGD(actor.parameters(), lr=0.0),
        critics_optim=torch.optim.SGD(critics.parameters(), lr=0.0),
        gamma=0.99,
        alpha=0.0,
        deterministic_backup=True,
        clamp_target_q=clamp_target_q,
        return_shift=return_shift,
    )


class MobileShiftTests(unittest.TestCase):
    def test_lift_dynamics_uses_continuing_termination(self):
        dynamics = policies.build_dynamics(
            obs_dim=2,
            action_dim=1,
            task="Lift",
            args=SimpleNamespace(device="cpu"),
            hidden_dims=[4, 4, 4, 4],
            penalty_coef=0.5,
        )

        self.assertIs(dynamics.terminal_fn, policies.termination_fn_never)
        np.testing.assert_array_equal(
            dynamics.terminal_fn(np.zeros((3, 2)), None, np.zeros((3, 2))),
            [[False], [False], [False]],
        )

    @patch("policies.build_dynamics", return_value=object())
    def test_reacher_initializes_online_and_target_critics_with_return_shift(self, _):
        args = SimpleNamespace(
            device="cpu", epoch=10, mobile_return_shift=30.0,
            model_manipulation_settings=False,
            model_actor_learning_rate=None, model_critic_learning_rate=3e-4,
            mopo_penalty_coef=0.5, mobile_penalty_coef=1.5,
        )
        torch.manual_seed(7)
        shifted, _, _ = policies.build_model_based_policy(
            "mobile", DummyEnv("Reacher-v5"), args, discount=0.99
        )
        args.mobile_return_shift = 0.0
        torch.manual_seed(7)
        unshifted, _, _ = policies.build_model_based_policy(
            "mobile", DummyEnv("Reacher-v5"), args, discount=0.99
        )

        self.assertEqual(shifted._return_shift, 30.0)
        self.assertTrue(shifted._clamp_target_q)
        for shifted_critic, unshifted_critic in zip(shifted.critics, unshifted.critics):
            torch.testing.assert_close(
                shifted_critic.last.bias, unshifted_critic.last.bias + 30.0
            )
        for shifted_critic, unshifted_critic in zip(shifted.critics_old, unshifted.critics_old):
            torch.testing.assert_close(
                shifted_critic.last.bias, unshifted_critic.last.bias + 30.0
            )

    @patch("policies.build_dynamics", return_value=object())
    def test_non_reacher_mobile_ignores_return_shift(self, _):
        args = SimpleNamespace(
            device="cpu", epoch=10, mobile_return_shift=30.0,
            model_manipulation_settings=False,
            model_actor_learning_rate=None, model_critic_learning_rate=3e-4,
            mopo_penalty_coef=0.5, mobile_penalty_coef=1.5,
        )
        torch.manual_seed(7)
        policy, _, _ = policies.build_model_based_policy(
            "mobile", DummyEnv("HalfCheetah-v5"), args, discount=0.99
        )

        self.assertEqual(policy._return_shift, 0.0)
        self.assertTrue(policy._clamp_target_q)

    @patch("policies.build_dynamics", return_value=object())
    def test_mobile_manipulation_settings_match_published_architecture(self, build_dynamics):
        args = SimpleNamespace(
            device="cpu", epoch=200, mobile_return_shift=30.0,
            model_manipulation_settings=True,
            model_actor_learning_rate=None, model_critic_learning_rate=3e-4,
            mopo_penalty_coef=0.5, mobile_penalty_coef=1.0,
        )
        policy, _, _ = policies.build_model_based_policy(
            "mobile", DummyEnv("Lift"), args, discount=0.99
        )

        hidden_widths = [
            layer.out_features
            for layer in policy.actor.backbone.model
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(hidden_widths, [256, 256, 256])
        self.assertEqual(len(policy.critics), 10)
        self.assertTrue(policy._max_q_backup)
        self.assertEqual(policy._penalty_coef, 1.0)
        self.assertEqual(policy.actor_optim.param_groups[0]["lr"], 3e-5)
        self.assertEqual(
            build_dynamics.call_args.kwargs["hidden_dims"],
            [400, 400, 400, 400],
        )

    @patch("policies.build_dynamics", return_value=object())
    def test_model_actor_learning_rate_override(self, _):
        args = SimpleNamespace(
            device="cpu", epoch=200, mobile_return_shift=30.0,
            model_manipulation_settings=False,
            model_actor_learning_rate=3e-5,
            model_critic_learning_rate=1e-4,
            mopo_penalty_coef=0.5, mobile_penalty_coef=1.5,
        )
        policy, _, _ = policies.build_model_based_policy(
            "mopo", DummyEnv("Lift"), args, discount=0.99
        )

        self.assertEqual(policy.actor_optim.param_groups[0]["lr"], 3e-5)
        self.assertEqual(policy.critic1_optim.param_groups[0]["lr"], 1e-4)

    def test_shifted_clamped_backup_matches_unshifted_backup_above_floor(self):
        batch_part = {
            "observations": torch.zeros((1, 1)),
            "actions": torch.zeros((1, 1)),
            "next_observations": torch.zeros((1, 1)),
            "rewards": torch.full((1, 1), -0.1),
            "terminals": torch.zeros((1, 1)),
        }
        batch = {"real": batch_part, "fake": batch_part}
        unshifted = make_constant_mobile(0.0, return_shift=0.0, clamp_target_q=False)
        shifted = make_constant_mobile(30.0, return_shift=30.0, clamp_target_q=True)

        unshifted_loss = unshifted.learn(batch)["loss/critic"]
        shifted_loss = shifted.learn(batch)["loss/critic"]

        self.assertAlmostEqual(unshifted_loss, 0.01, places=6)
        self.assertAlmostEqual(shifted_loss, unshifted_loss, places=5)


if __name__ == "__main__":
    unittest.main()

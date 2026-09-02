import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import policies
from chunked_dynamics import RecursiveChunkDynamics
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.policy import MOBILEPolicy


class DummyEnv:
    def __init__(self, env_id: str, action_dim: int = 1):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,))
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
    @staticmethod
    def model_args(**overrides):
        values = {
            "device": "cpu",
            "epoch": 10,
            "mobile_return_shift": 30.0,
            "model_manipulation_settings": False,
            "model_actor_learning_rate": None,
            "model_critic_learning_rate": 3e-4,
            "mopo_penalty_coef": 0.5,
            "mobile_penalty_coef": 1.5,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_algorithm_registries_match_active_builders(self):
        self.assertEqual(
            policies.MODEL_FREE_ALGOS,
            ("bc", "cql", "iql", "td3bc", "edac"),
        )
        self.assertEqual(
            policies.MODEL_BASED_ALGOS,
            ("mopo", "combo", "mobile"),
        )
        self.assertEqual(policies.ROBOMIMIC_TASKS, {"can", "lift"})

    def test_td3bc_requires_observation_normalization_buffer(self):
        with self.assertRaisesRegex(ValueError, "TD3\\+BC requires a replay buffer"):
            policies.build_model_free_policy(
                "td3bc", DummyEnv("Reacher-v5"), None,
                SimpleNamespace(device="cpu"), discount=0.99,
            )

    def test_inference_only_model_based_policies_strictly_load_training_state(self):
        env = DummyEnv("Reacher-v5")
        args = self.model_args()
        for algo in policies.MODEL_BASED_ALGOS:
            with self.subTest(algo=algo), patch.object(
                policies, "build_dynamics", return_value=MagicMock()
            ):
                torch.manual_seed(7)
                trained_policy, trained_dynamics, _ = policies.build_model_based_policy(
                    algo, env, args, discount=0.99, build_dynamics_model=True
                )
                torch.manual_seed(8)
                inference_policy, inference_dynamics, _ = policies.build_model_based_policy(
                    algo, env, args, discount=0.99, build_dynamics_model=False
                )

                self.assertIsNotNone(trained_dynamics)
                self.assertIsNone(inference_dynamics)
                inference_policy.load_state_dict(
                    trained_policy.state_dict(), strict=True
                )
                self.assertEqual(
                    set(inference_policy.state_dict()),
                    set(trained_policy.state_dict()),
                )

    def test_model_based_builder_rejects_unsupported_algorithm_before_construction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
            policies.build_model_based_policy(
                "unsupported",
                DummyEnv("Lift"),
                self.model_args(),
                discount=0.99,
            )

    def test_chunk_one_recursive_request_uses_exact_plain_dynamics_path(self):
        torch.manual_seed(13)
        direct_policy, direct_dynamics, _ = policies.build_model_based_policy(
            "mopo",
            DummyEnv("HalfCheetah-v5"),
            self.model_args(),
            discount=0.99,
            chunk_length=1,
            base_discount=0.99,
            dynamics_chunk_mode="direct",
            primitive_action_dim=1,
        )
        torch.manual_seed(13)
        recursive_policy, recursive_dynamics, _ = policies.build_model_based_policy(
            "mopo",
            DummyEnv("HalfCheetah-v5"),
            self.model_args(),
            discount=0.99,
            chunk_length=1,
            base_discount=0.99,
            dynamics_chunk_mode="recursive",
            primitive_action_dim=1,
        )

        self.assertIs(type(direct_dynamics), EnsembleDynamics)
        self.assertIs(type(recursive_dynamics), EnsembleDynamics)
        self.assertNotIsInstance(recursive_dynamics, RecursiveChunkDynamics)
        for key, value in direct_policy.state_dict().items():
            torch.testing.assert_close(recursive_policy.state_dict()[key], value)
        for key, value in direct_dynamics.model.state_dict().items():
            torch.testing.assert_close(recursive_dynamics.model.state_dict()[key], value)

    def test_recursive_chunk_dynamics_uses_primitive_action_width_only(self):
        policy, dynamics, _ = policies.build_model_based_policy(
            "mobile",
            DummyEnv("Lift", action_dim=4),
            self.model_args(),
            discount=0.99**4,
            chunk_length=4,
            base_discount=0.99,
            dynamics_chunk_mode="recursive",
            primitive_action_dim=1,
        )

        self.assertIsInstance(dynamics, RecursiveChunkDynamics)
        self.assertEqual(dynamics.model.backbones[0].weight.shape[1], 3)
        action, _ = policy.actforward(torch.zeros((2, 2)), deterministic=True)
        self.assertEqual(tuple(action.shape), (2, 4))

    def test_recursive_builder_rejects_inconsistent_action_dimensions(self):
        with self.assertRaisesRegex(ValueError, "primitive action dimension"):
            policies.build_model_based_policy(
                "mopo",
                DummyEnv("Lift", action_dim=4),
                self.model_args(),
                discount=0.99**4,
                chunk_length=4,
                base_discount=0.99,
                dynamics_chunk_mode="recursive",
                primitive_action_dim=2,
            )

    def test_recursive_builder_rejects_inconsistent_macro_discount(self):
        with self.assertRaisesRegex(ValueError, "macro discount"):
            policies.build_model_based_policy(
                "mopo",
                DummyEnv("Lift", action_dim=4),
                self.model_args(),
                discount=0.99,
                chunk_length=4,
                base_discount=0.99,
                dynamics_chunk_mode="recursive",
                primitive_action_dim=1,
            )

    def test_recursive_mobile_real_components_complete_macro_step_and_lcb(self):
        policy, dynamics, _ = policies.build_model_based_policy(
            "mobile",
            DummyEnv("Lift", action_dim=4),
            self.model_args(),
            discount=0.99**4,
            chunk_length=4,
            base_discount=0.99,
            dynamics_chunk_mode="recursive",
            primitive_action_dim=1,
        )
        dynamics.scaler.fit(
            np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
        )
        observations = torch.zeros((3, 2))
        action_chunks = torch.zeros((3, 4))

        sampled_next_observations = dynamics.sample_next_obss(
            observations, action_chunks, num_samples=2
        )
        lcb = policy.compute_lcb(observations, action_chunks)
        next_observations, rewards, terminals, _ = dynamics.step(
            observations.numpy(), action_chunks.numpy()
        )

        self.assertEqual(tuple(sampled_next_observations.shape), (2, 5, 3, 2))
        self.assertEqual(tuple(lcb.shape), (3, 1))
        self.assertEqual(next_observations.shape, (3, 2))
        self.assertEqual(rewards.shape, (3, 1))
        self.assertEqual(terminals.shape, (3, 1))
        self.assertTrue(torch.isfinite(sampled_next_observations).all())
        self.assertTrue(torch.isfinite(lcb).all())
        self.assertTrue(np.isfinite(next_observations).all())
        self.assertTrue(np.isfinite(rewards).all())

    def test_recursive_policy_and_dynamics_checkpoint_round_trip_is_strict(self):
        torch.manual_seed(17)
        policy, dynamics, _ = policies.build_model_based_policy(
            "mopo",
            DummyEnv("Lift", action_dim=4),
            self.model_args(),
            discount=0.99**4,
            chunk_length=4,
            base_discount=0.99,
            dynamics_chunk_mode="recursive",
            primitive_action_dim=1,
        )
        dynamics.scaler.fit(
            np.random.default_rng(1).normal(size=(32, 3)).astype(np.float32)
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            torch.save(policy.state_dict(), checkpoint_dir / "policy.pth")
            dynamics.save(checkpoint_dir)

            torch.manual_seed(23)
            loaded_policy, loaded_dynamics, _ = policies.build_model_based_policy(
                "mopo",
                DummyEnv("Lift", action_dim=4),
                self.model_args(),
                discount=0.99**4,
                chunk_length=4,
                base_discount=0.99,
                dynamics_chunk_mode="recursive",
                primitive_action_dim=1,
            )
            loaded_policy.load_state_dict(
                torch.load(
                    checkpoint_dir / "policy.pth",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            loaded_dynamics.load(checkpoint_dir)

        for key, value in policy.state_dict().items():
            torch.testing.assert_close(loaded_policy.state_dict()[key], value)
        for key, value in dynamics.model.state_dict().items():
            torch.testing.assert_close(loaded_dynamics.model.state_dict()[key], value)
        np.testing.assert_array_equal(loaded_dynamics.scaler.mu, dynamics.scaler.mu)
        np.testing.assert_array_equal(loaded_dynamics.scaler.std, dynamics.scaler.std)

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

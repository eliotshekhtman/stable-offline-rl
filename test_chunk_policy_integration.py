"""CPU learning smoke tests for the optional actors in the actual policies."""

import copy
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.nn.utils import parametrize

import policies
from chunk_policy import fit_observation_stats
from offlinerlkit.modules.dist_module import NormalWrapper, TanhNormalWrapper


def build_policy(algo, length, rho=0.8, manipulation=False, dynamics_mode=None, version=None):
    args = SimpleNamespace(
        device="cpu", epoch=10, iql_hidden_dims=[16, 16],
        iql_learning_rate=3e-4, iql_temperature=3.0, iql_expectile=0.7,
        iql_lr_schedule="cosine", td3bc_hidden_dims=[16, 16],
        td3bc_learning_rate=3e-4, td3bc_alpha=0.05,
        model_manipulation_settings=manipulation, model_actor_learning_rate=None,
        model_critic_learning_rate=3e-4, mopo_penalty_coef=0.5,
        mobile_penalty_coef=1.5, mobile_return_shift=30.0,
    )
    if rho is not None:
        args.chunk_policy = "decay-sn"
        args.chunk_policy_rho = rho
        if version is not None:
            args.chunk_policy_config = {"version": version, "type": "decay-sn", "rho": rho}
    normalization = {}
    if rho is not None and algo != "td3bc" and version != 1:
        observations = np.asarray([
            [-3, 1, 3, 0.2, 2], [1, 3, 5, 1.2, 3], [5, 2, 7, 2.2, 4],
        ], dtype=np.float32)
        normalization["observation_stats"] = fit_observation_stats(observations)
    env = SimpleNamespace(
        observation_space=gym.spaces.Box(-10.0, 10.0, shape=(5,)),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2 * length,)),
        spec=SimpleNamespace(id="Lift" if manipulation else "Reacher-v5"),
    )
    if algo in ("td3bc", "iql"):
        buffer = SimpleNamespace(normalize_obs=lambda: (np.zeros((1, 5)), np.ones((1, 5))))
        return policies.build_model_free_policy(
            algo, env, buffer, args, discount=0.99**length, chunk_length=length,
            **normalization,
        )[0]
    return policies.build_model_based_policy(
        algo, env, args, discount=0.99**length, chunk_length=length,
        dynamics_chunk_mode=dynamics_mode or "direct",
        build_dynamics_model=dynamics_mode is not None,
        **normalization,
    )[0]


class SyntheticDynamics:
    """Small ensemble interface; policy updates and penalties are not mocked."""

    def sample_next_obss(self, observations, actions, num_samples):
        offsets = torch.tensor([-0.1, 0.1]).view(1, 2, 1, 1)
        return (observations[None, None] + offsets).expand(num_samples, -1, -1, -1)

    def step(self, observations, actions):
        count = len(observations)
        return (observations + 0.1, np.ones((count, 1), dtype=np.float32),
                np.zeros((count, 1), dtype=bool), {})


def make_batch(length):
    return {
        "observations": torch.randn(8, 5),
        "next_observations": torch.randn(8, 5),
        "actions": torch.rand(8, 2 * length) * 1.8 - 0.9,
        "rewards": torch.rand(8, 1), "terminals": torch.zeros(8, 1),
    }


class ChunkPolicyLearningTests(unittest.TestCase):
    def test_actual_learning_steps_all_algorithms_and_horizons(self):
        for algo in ("td3bc", "iql", "mopo", "mobile"):
            for length in (1, 4, 16):
                with self.subTest(algo=algo, length=length):
                    torch.manual_seed(11)
                    policy = build_policy(algo, length)
                    if algo in ("mopo", "mobile"):
                        policy.dynamics = SyntheticDynamics()
                    policy.train()
                    before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
                    batch = make_batch(length)
                    for _ in range(4):
                        losses = policy.learn(
                            {"real": batch, "fake": batch} if algo in ("mopo", "mobile") else batch
                        )
                        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, policy.actor.parameters())))
                    optimizer_ids = {id(p) for group in policy.actor_optim.param_groups for p in group["params"]}
                    self.assertEqual(optimizer_ids, {id(p) for p in policy.actor.parameters()})
                    for parameter in policy.actor.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter).all())
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    policy.eval()
                    actions = policy.select_action(batch["observations"].numpy(), deterministic=True)
                    self.assertEqual(actions.shape, (8, 2 * length))
                    self.assertTrue(np.isfinite(actions).all())
                    self.assertTrue((np.abs(actions) <= 1).all())
                    restored = build_policy(algo, length)
                    restored.load_state_dict(policy.state_dict(), strict=True)
                    restored.eval()
                    np.testing.assert_array_equal(
                        restored.select_action(batch["observations"].numpy(), deterministic=True), actions,
                    )
                    if algo == "td3bc":
                        self.assertFalse(policy.actor_old.training)

    def test_model_based_manipulation_learning_and_rollouts(self):
        for algo in ("mopo", "mobile"):
            with self.subTest(algo=algo):
                policy = build_policy(algo, 4, manipulation=True)
                policy.dynamics = SyntheticDynamics()
                batch = make_batch(4)
                policy.train()
                losses = policy.learn({"real": batch, "fake": batch})
                self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                rollout, info = policy.rollout(batch["observations"].numpy(), 2)
                self.assertEqual(info["num_transitions"], 16)
                self.assertEqual(rollout["actions"].shape, (16, 8))
                self.assertEqual(policy._target_entropy, -8)

    def test_probabilistic_distributions_and_sigma_formula_are_preserved(self):
        observations = torch.randn(3, 5)
        for algo in ("iql", "mopo", "mobile"):
            with self.subTest(algo=algo):
                policy = build_policy(algo, 4)
                policy.eval()
                actor = policy.actor
                dist = actor(observations)
                features = actor.backbone(observations)
                mean = actor.dist_net.mu(features)
                if algo == "iql":
                    self.assertIs(type(dist), NormalWrapper)
                    torch.testing.assert_close(dist.mode(), mean.tanh())
                    torch.testing.assert_close(dist.scale, actor.dist_net.sigma_param.view(1, -1).exp().expand(3, -1))
                else:
                    self.assertIs(type(dist), TanhNormalWrapper)
                    self.assertFalse(parametrize.is_parametrized(actor.dist_net.sigma))
                    torch.testing.assert_close(dist.mode()[0], mean.tanh())
                    torch.testing.assert_close(dist.scale, actor.dist_net.sigma(features).clamp(-5, 2).exp())
                    self.assertEqual(policy._target_entropy, -8)
                    actions, raw = dist.rsample()
                    expected = torch.distributions.Normal(dist.loc, dist.scale).log_prob(raw).sum(-1, keepdim=True)
                    expected -= torch.log(1 - actions.square() + 1e-6).sum(-1, keepdim=True)
                    torch.testing.assert_close(dist.log_prob(actions, raw), expected)

    def test_rho_changes_only_decay_at_initialization(self):
        for algo in ("td3bc", "iql", "mopo", "mobile"):
            with self.subTest(algo=algo):
                torch.manual_seed(29)
                original = copy.deepcopy(build_policy(algo, 4, rho=1).state_dict())
                torch.manual_seed(29)
                decaying = build_policy(algo, 4, rho=0.8).state_dict()
                self.assertEqual(original.keys(), decaying.keys())
                for key in original:
                    if key.endswith(".decay"):
                        self.assertFalse(torch.equal(original[key], decaying[key]))
                    else:
                        torch.testing.assert_close(original[key], decaying[key], rtol=0, atol=0)

    def test_actual_ensemble_dynamics_direct_and_recursive_interfaces(self):
        for algo in ("mopo", "mobile"):
            for mode in ("direct", "recursive"):
                with self.subTest(algo=algo, mode=mode):
                    policy = build_policy(algo, 4, dynamics_mode=mode)
                    dynamics = policy.dynamics
                    input_dim = 5 + (8 if mode == "direct" else 2)
                    dynamics.scaler.fit(np.random.default_rng(3).normal(size=(32, input_dim)))
                    dynamics.model.eval()
                    batch = make_batch(4)
                    policy.train()
                    rollout, info = policy.rollout(batch["observations"].numpy(), 1)
                    self.assertEqual(info["num_transitions"], 8)
                    self.assertTrue(np.isfinite(rollout["next_obss"]).all())
                    losses = policy.learn({"real": batch, "fake": batch})
                    self.assertTrue(all(np.isfinite(value) for value in losses.values()))


if __name__ == "__main__":
    unittest.main()

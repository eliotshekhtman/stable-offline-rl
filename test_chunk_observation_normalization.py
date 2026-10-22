"""Fixed actor/value observation preprocessing for version-2 decay-SN policies."""

import copy
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import policies
from chunk_policy import NormalizedObservationBackbone, fit_observation_stats
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.nets import MLP


MEAN = np.array([1.0, -2.0, 7.0, 10.0, -4.0], dtype=np.float32)
STD = np.array([0.03, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)


def make_policy(algo, length=4, *, stats=(MEAN, STD), version=None,
                manipulation=False, dynamics_mode=None, standard=False):
    args = SimpleNamespace(
        device="cpu", epoch=10, iql_hidden_dims=[16, 16],
        iql_learning_rate=3e-4, iql_temperature=3.0, iql_expectile=0.7,
        iql_lr_schedule="cosine", model_manipulation_settings=manipulation,
        model_actor_learning_rate=None, model_critic_learning_rate=3e-4,
        mopo_penalty_coef=0.5, mobile_penalty_coef=1.5,
        mobile_return_shift=30.0,
    )
    if not standard:
        args.chunk_policy = "decay-sn"
        args.chunk_policy_rho = 0.8
    if version is not None:
        args.chunk_policy_config = {"version": version, "type": "decay-sn", "rho": 0.8}
    env = SimpleNamespace(
        observation_space=gym.spaces.Box(-100.0, 100.0, shape=(5,)),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2 * length,)),
        spec=SimpleNamespace(id="Lift" if manipulation else "Reacher-v5"),
    )
    if algo == "iql":
        return policies.build_model_free_policy(
            algo, env, None, args, discount=0.99**length,
            chunk_length=length, observation_stats=stats,
        )[0]
    return policies.build_model_based_policy(
        algo, env, args, discount=0.99**length, chunk_length=length,
        observation_stats=stats, dynamics_chunk_mode=dynamics_mode or "direct",
        build_dynamics_model=dynamics_mode is not None,
    )[0]


def make_batch(length, count=8):
    return {
        "observations": torch.as_tensor(MEAN) + torch.as_tensor(STD) * torch.randn(count, 5),
        "next_observations": torch.as_tensor(MEAN) + torch.as_tensor(STD) * torch.randn(count, 5),
        "actions": torch.rand(count, 2 * length) * 1.8 - 0.9,
        "rewards": torch.rand(count, 1),
        "terminals": torch.zeros(count, 1),
    }


def networks(policy, algo):
    if algo == "iql":
        names = ("actor", "critic_q1", "critic_q2", "critic_q1_old", "critic_q2_old", "critic_v")
        return {name: getattr(policy, name) for name in names}
    if algo == "mopo":
        names = ("actor", "critic1", "critic2", "critic1_old", "critic2_old")
        return {name: getattr(policy, name) for name in names}
    return {
        "actor": policy.actor,
        **{f"critic_{i}": network for i, network in enumerate(policy.critics)},
        **{f"target_{i}": network for i, network in enumerate(policy.critics_old)},
    }


class RawDynamicsSpy:
    """Observe dynamics inputs without replacing the algorithm's learning paths."""

    def __init__(self):
        self.steps = []
        self.samples = []

    def step(self, observations, actions):
        self.steps.append((observations.copy(), actions.copy()))
        count = len(observations)
        return (observations + 0.1, np.ones((count, 1), dtype=np.float32),
                np.zeros((count, 1), dtype=bool), {})

    def sample_next_obss(self, observations, actions, num_samples):
        self.samples.append((observations.detach().clone(), actions.detach().clone()))
        offsets = observations.new_tensor([-0.1, 0.1]).view(1, 2, 1, 1)
        return (observations[None, None] + offsets).expand(num_samples, -1, -1, -1)


class ObservationStatsTests(unittest.TestCase):
    def test_exact_td3_replay_buffer_oracle_and_no_source_mutation(self):
        observations = np.random.default_rng(41).normal(size=(31, 5)).astype(np.float32)
        observations[:, 2] = 7.0
        before = observations.copy()
        buffer = ReplayBuffer(31, (5,), np.float32, 2, np.float32)
        buffer.load_dataset({
            "observations": observations,
            "next_observations": observations * 50.0 + 100.0,
            "actions": np.zeros((31, 2), dtype=np.float32),
            "rewards": np.zeros(31), "terminals": np.zeros(31),
        })
        expected_mean, expected_std = buffer.normalize_obs()
        mean, std = fit_observation_stats(observations)
        self.assertEqual(mean.shape, (5,))
        self.assertEqual(std.shape, (5,))
        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(std.dtype, np.float32)
        np.testing.assert_array_equal(mean, expected_mean[0])
        np.testing.assert_array_equal(std, expected_std[0])
        self.assertEqual(std[2], np.float32(0.001))
        np.testing.assert_array_equal(observations, before)
        mean64, std64 = fit_observation_stats(observations.astype(np.float64))
        np.testing.assert_array_equal(mean64, mean)
        np.testing.assert_array_equal(std64, std)

    def test_invalid_training_observations_rejected(self):
        for observations in (np.empty((0, 5)), np.zeros(5), np.full((3, 5), np.nan)):
            with self.subTest(shape=observations.shape):
                with self.assertRaises(ValueError):
                    fit_observation_stats(observations)


class NormalizedBackboneTests(unittest.TestCase):
    def test_prefix_transform_shapes_and_input_gradients(self):
        for shape in ((7,), (4, 7), (0, 7), (2, 3, 7)):
            with self.subTest(shape=shape):
                identity = nn.Identity()
                identity.output_dim = 7
                wrapper = NormalizedObservationBackbone(identity, MEAN, STD)
                values = torch.randn(shape, requires_grad=True)
                before = values.detach().clone()
                actual = wrapper(values)
                expected = torch.cat([
                    (values[..., :5] - torch.as_tensor(MEAN)) / torch.as_tensor(STD),
                    values[..., 5:],
                ], dim=-1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_gradient = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)[0]
                expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                torch.testing.assert_close(values, before, rtol=0, atol=0)
                self.assertEqual(wrapper.output_dim, 7)

    def test_parameter_identity_and_frozen_copied_buffers(self):
        backbone = MLP(7, [8, 8])
        original_parameters = list(backbone.parameters())
        mean, std = MEAN.copy(), STD.copy()
        wrapper = NormalizedObservationBackbone(backbone, mean, std)
        self.assertEqual([id(p) for p in wrapper.parameters()], [id(p) for p in original_parameters])
        self.assertEqual(set(dict(wrapper.named_buffers())), {"mean", "std"})
        self.assertFalse(wrapper.mean.requires_grad)
        self.assertFalse(wrapper.std.requires_grad)
        mean[:] = 100
        std[:] = 100
        np.testing.assert_array_equal(wrapper.mean.numpy(), MEAN)
        np.testing.assert_array_equal(wrapper.std.numpy(), STD)
        values = torch.randn(4, 7)
        expected = backbone(torch.cat([
            (values[:, :5] - torch.as_tensor(MEAN)) / torch.as_tensor(STD), values[:, 5:],
        ], dim=-1))
        torch.testing.assert_close(wrapper(values), expected, rtol=0, atol=0)

    def test_state_restores_exact_stats_and_rejects_missing_or_wrong_shapes(self):
        original = NormalizedObservationBackbone(MLP(7, [8]), MEAN, STD)
        restored = NormalizedObservationBackbone(MLP(7, [8]), np.zeros(5), np.ones(5))
        restored.load_state_dict(original.state_dict(), strict=True)
        values = torch.randn(3, 7)
        torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)
        missing = dict(original.state_dict())
        missing.pop("std")
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing, strict=True)
        malformed = dict(original.state_dict())
        malformed["mean"] = torch.zeros(6)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(malformed, strict=True)
        for name, value in (("mean", torch.full((5,), float("nan"))),
                            ("std", torch.full((5,), float("inf"))),
                            ("std", torch.zeros(5)), ("std", -torch.ones(5))):
            with self.subTest(name=name, value=value):
                malformed = dict(original.state_dict())
                malformed[name] = value
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(malformed, strict=True)

    def test_invalid_stats_rejected(self):
        for mean, std in ((MEAN, np.zeros(5)), (MEAN, -STD), (MEAN, np.ones(4)),
                          (np.full(5, np.nan), STD), (MEAN, np.full(5, np.inf))):
            with self.subTest(mean=mean, std=std):
                with self.assertRaises(ValueError):
                    NormalizedObservationBackbone(MLP(7, [8]), mean, std)


class NormalizedPolicyTests(unittest.TestCase):
    def watch_normalization(self, policy, algo):
        """Check each actual network call, including targets and auxiliary paths."""
        records, handles = {}, []
        for name, network in networks(policy, algo).items():
            wrapper = network.backbone
            self.assertIsInstance(wrapper, NormalizedObservationBackbone)
            torch.testing.assert_close(wrapper.mean, torch.as_tensor(MEAN), rtol=0, atol=0)
            torch.testing.assert_close(wrapper.std, torch.as_tensor(STD), rtol=0, atol=0)
            records[name] = {"raw": [], "inner": []}

            def before_outer(module, inputs, name=name):
                records[name]["raw"].append(inputs[0].detach().clone())

            def before_inner(module, inputs, name=name, wrapper=wrapper):
                raw = records[name]["raw"][-1]
                expected = torch.cat([(raw[..., :5] - wrapper.mean) / wrapper.std, raw[..., 5:]], dim=-1)
                torch.testing.assert_close(inputs[0], expected, rtol=0, atol=0)
                records[name]["inner"].append(inputs[0].detach().clone())

            handles.extend((wrapper.register_forward_pre_hook(before_outer),
                            wrapper.backbone.register_forward_pre_hook(before_inner)))
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        return records

    def test_identity_normalization_preserves_legacy_initialization_and_update(self):
        for algo in ("iql", "mopo", "mobile"):
            with self.subTest(algo=algo):
                torch.manual_seed(44)
                legacy = make_policy(algo, version=1, stats=None)
                torch.manual_seed(44)
                normalized = make_policy(algo, stats=(np.zeros(5), np.ones(5)))
                self.assertEqual(len(list(legacy.parameters())), len(list(normalized.parameters())))
                for old, new in zip(legacy.parameters(), normalized.parameters()):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                batch = make_batch(4)
                legacy.eval()
                normalized.eval()
                np.testing.assert_array_equal(
                    legacy.select_action(batch["observations"].numpy(), deterministic=True),
                    normalized.select_action(batch["observations"].numpy(), deterministic=True),
                )
                losses = []
                for policy in (legacy, normalized):
                    if algo != "iql":
                        policy.dynamics = RawDynamicsSpy()
                    policy.train()
                    torch.manual_seed(45)
                    losses.append(policy.learn(batch if algo == "iql" else {"real": batch, "fake": batch}))
                self.assertEqual(losses[0], losses[1])
                for old, new in zip(legacy.parameters(), normalized.parameters()):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_learning_all_network_paths_frozen_stats_and_strict_roundtrip(self):
        for algo in ("iql", "mopo", "mobile"):
            for length in (1, 4, 16):
                with self.subTest(algo=algo, length=length):
                    torch.manual_seed(31)
                    policy = make_policy(algo, length)
                    records = self.watch_normalization(policy, algo)
                    if algo != "iql":
                        policy.dynamics = RawDynamicsSpy()
                    before = {name: value.clone() for name, value in policy.named_buffers()
                              if name.endswith((".mean", ".std"))}
                    batch = make_batch(length)
                    original_batch = {name: value.clone() for name, value in batch.items()}
                    actor_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
                    policy.train()
                    for _ in range(3):
                        losses = policy.learn(batch if algo == "iql" else {"real": batch, "fake": batch})
                        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                    self.assertTrue(any(not torch.equal(old, new) for old, new in
                                        zip(actor_before, policy.actor.parameters())))
                    for name, value in before.items():
                        torch.testing.assert_close(dict(policy.named_buffers())[name], value, rtol=0, atol=0)
                    for name, value in original_batch.items():
                        torch.testing.assert_close(batch[name], value, rtol=0, atol=0)
                    for name, calls in records.items():
                        self.assertGreater(len(calls["inner"]), 0, name)
                        self.assertEqual(len(calls["raw"]), len(calls["inner"]), name)
                    if algo == "mobile":
                        expected = torch.cat([batch["observations"], batch["observations"]])
                        for observations, _ in policy.dynamics.samples:
                            torch.testing.assert_close(observations, expected, rtol=0, atol=0)
                    policy.eval()
                    expected_actions = policy.select_action(batch["observations"].numpy(), deterministic=True)
                    restored = make_policy(algo, length, stats=(np.zeros(5), np.ones(5)))
                    restored.load_state_dict(copy.deepcopy(policy.state_dict()), strict=True)
                    restored.eval()
                    np.testing.assert_array_equal(
                        restored.select_action(batch["observations"].numpy(), deterministic=True), expected_actions,
                    )
                    for name, network in networks(policy, algo).items():
                        if name == "actor":
                            continue
                        args = (batch["observations"],) if name == "critic_v" else (
                            batch["observations"], batch["actions"],
                        )
                        torch.testing.assert_close(network(*args), networks(restored, algo)[name](*args), rtol=0, atol=0)

    def test_model_based_rollouts_stay_raw_and_empty_real_batch_learns(self):
        for algo in ("mopo", "mobile"):
            with self.subTest(algo=algo):
                policy = make_policy(algo)
                self.watch_normalization(policy, algo)
                policy.dynamics = RawDynamicsSpy()
                batch = make_batch(4)
                observations = batch["observations"].numpy().copy()
                rollout, info = policy.rollout(observations, 2)
                self.assertEqual(info["num_transitions"], 16)
                np.testing.assert_array_equal(policy.dynamics.steps[0][0], observations)
                np.testing.assert_array_equal(policy.dynamics.steps[1][0], observations + 0.1)
                np.testing.assert_array_equal(rollout["obss"][:8], observations)
                np.testing.assert_array_equal(rollout["next_obss"][:8], observations + 0.1)
                empty = {name: value[:0] for name, value in batch.items()}
                losses = policy.learn({"real": empty, "fake": batch})
                self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                if algo == "mobile":
                    torch.testing.assert_close(policy.dynamics.samples[-1][0], batch["observations"], rtol=0, atol=0)

    def test_mobile_max_q_backup_and_lcb_use_normalized_successors(self):
        policy = make_policy("mobile", manipulation=True)
        records = self.watch_normalization(policy, "mobile")
        policy.dynamics = RawDynamicsSpy()
        batch = make_batch(4)
        losses = policy.learn({"real": batch, "fake": batch})
        self.assertTrue(policy._max_q_backup)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        mix = torch.cat([batch["observations"], batch["observations"]])
        predicted = policy.dynamics.sample_next_obss(mix, torch.cat([batch["actions"]] * 2), policy._num_samples)
        expected_raw = predicted.reshape(-1, 5)
        self.assertTrue(any(torch.equal(values[:, :5], expected_raw)
                            for values in records["actor"]["raw"] if len(values) == len(expected_raw)))
        max_q_count = 2 * len(batch["observations"]) * 10
        self.assertTrue(any(len(values) == max_q_count for values in records["actor"]["raw"]))
        for name, calls in records.items():
            self.assertGreater(len(calls["inner"]), 0, name)

    def test_direct_and_recursive_dynamics_accept_raw_observations(self):
        for algo in ("mopo", "mobile"):
            for mode in ("direct", "recursive"):
                with self.subTest(algo=algo, mode=mode):
                    policy = make_policy(algo, dynamics_mode=mode)
                    self.watch_normalization(policy, algo)
                    dynamics = policy.dynamics
                    input_dim = 5 + (8 if mode == "direct" else 2)
                    dynamics.scaler.fit(np.random.default_rng(3).normal(size=(32, input_dim)))
                    dynamics.model.eval()
                    batch = make_batch(4)
                    rollout, info = policy.rollout(batch["observations"].numpy(), 1)
                    self.assertEqual(info["num_transitions"], 8)
                    np.testing.assert_array_equal(rollout["obss"], batch["observations"].numpy())
                    self.assertTrue(np.isfinite(rollout["next_obss"]).all())
                    losses = policy.learn({"real": batch, "fake": batch})
                    self.assertTrue(all(np.isfinite(value) for value in losses.values()))

    def test_only_new_decay_configs_require_and_install_normalization(self):
        for algo in ("iql", "mopo", "mobile"):
            with self.subTest(algo=algo):
                with self.assertRaises(ValueError):
                    make_policy(algo, stats=None)
                for kwargs in ({"standard": True}, {"version": 1}):
                    policy = make_policy(algo, stats=None, **kwargs)
                    self.assertFalse(any(isinstance(module, NormalizedObservationBackbone)
                                         for module in policy.modules()))


if __name__ == "__main__":
    unittest.main()

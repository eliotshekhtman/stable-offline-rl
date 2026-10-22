import io
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from chunk_policy import (
    CHUNK_POLICY_ALGOS,
    CHUNK_POLICY_MODES,
    ChunkTD3BCPolicy,
    DecayingActor,
    DecayingChunkMean,
    apply_chunk_spectral_norm,
    get_chunk_policy_config,
    refresh_chunk_spectral_norm,
    validate_chunk_policy_config,
)
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian, TanhDiagGaussian
from offlinerlkit.nets import MLP


def make_actor(chunk_length=4, hidden_dims=(32, 32), rho=0.8):
    return DecayingActor(
        MLP(5, hidden_dims), 3 * chunk_length, 1.5, "cpu", chunk_length, rho
    )


def normalized_linears(actor):
    return [
        module for module in actor.modules()
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module, "weight")
    ]


def snapshot(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


class ChunkPolicyConfigTests(unittest.TestCase):
    def test_standard_has_no_metadata_and_needs_no_new_arguments(self):
        self.assertEqual(CHUNK_POLICY_MODES, ("standard", "decay-sn"))
        for algo in ("bc", "cql", "edac", "combo", *CHUNK_POLICY_ALGOS):
            self.assertIsNone(get_chunk_policy_config(SimpleNamespace(), algo))
            self.assertIsNone(get_chunk_policy_config(
                SimpleNamespace(chunk_policy="standard", chunk_policy_rho=None), algo
            ))

    def test_enabled_metadata_is_canonical_for_supported_algorithms(self):
        for algo in CHUNK_POLICY_ALGOS:
            for rho in (0.01, 0.8, 1):
                config = get_chunk_policy_config(
                    SimpleNamespace(chunk_policy="decay-sn", chunk_policy_rho=rho), algo
                )
                version = 1 if algo == "td3bc" else 2
                self.assertEqual(config, {"version": version, "type": "decay-sn", "rho": float(rho)})
                self.assertIs(type(config["rho"]), float)
                self.assertEqual(validate_chunk_policy_config(config, algo), config)

    def test_rho_requires_enabled_mode_and_supported_algorithm(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            get_chunk_policy_config(SimpleNamespace(chunk_policy_rho=0.8), "td3bc")
        for algo in ("bc", "cql", "edac", "combo", "unknown"):
            with self.subTest(algo=algo), self.assertRaisesRegex(ValueError, "unsupported"):
                get_chunk_policy_config(
                    SimpleNamespace(chunk_policy="decay-sn", chunk_policy_rho=0.8), algo
                )
        for mode in (None, "other", True, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                get_chunk_policy_config(SimpleNamespace(chunk_policy=mode), "td3bc")

    def test_rho_is_required_finite_numeric_and_in_range(self):
        for rho in (None, False, True, "0.8", 0, -1, 1.01, math.inf, -math.inf, math.nan):
            with self.subTest(rho=rho), self.assertRaisesRegex(ValueError, "rho"):
                get_chunk_policy_config(
                    SimpleNamespace(chunk_policy="decay-sn", chunk_policy_rho=rho), "iql"
                )

    def test_checkpoint_metadata_rejects_unknown_and_malformed_fields(self):
        valid = {"version": 1, "type": "decay-sn", "rho": 0.8}
        invalid = [None, [], "decay-sn", {}, {**valid, "G": 1},
                   {"type": "decay-sn", "rho": 0.8}]
        invalid.extend({**valid, "version": version} for version in (0, 2, True, 1.0, "1", None))
        invalid.extend({**valid, "type": mode} for mode in ("standard", "other", None))
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_chunk_policy_config(config, "td3bc")

    def test_recorded_version_overrides_training_default(self):
        for algo in CHUNK_POLICY_ALGOS:
            config = {"version": 1, "type": "decay-sn", "rho": 0.8}
            self.assertEqual(get_chunk_policy_config(
                SimpleNamespace(chunk_policy_config=config), algo
            ), config)
        for algo in ("iql", "mopo", "mobile"):
            config = {"version": 2, "type": "decay-sn", "rho": 0.8}
            self.assertEqual(validate_chunk_policy_config(config, algo), config)
            with self.assertRaises(ValueError):
                validate_chunk_policy_config({**config, "version": 3}, algo)


class DecayingMeanTests(unittest.TestCase):
    def test_formula_and_chunk_order_with_unscaled_offsets(self):
        mean = DecayingChunkMean(2, 6, 3, 0.5).double()
        with torch.no_grad():
            for k, head in enumerate(mean.heads):
                head.weight.copy_(torch.tensor([[k + 1, 0.0], [0.0, k + 2]]))
                head.bias.copy_(torch.tensor([10.0 + k, -20.0 - k]))
        features = torch.tensor([2.0, 3.0], dtype=torch.float64)
        expected = torch.tensor([12.0, -14.0, 13.0, -16.5, 13.5, -19.0], dtype=torch.float64)
        torch.testing.assert_close(mean(features), expected, rtol=0, atol=0)
        offsets = torch.cat([head.bias for head in mean.heads])
        torch.testing.assert_close(mean(torch.zeros_like(features)), offsets, rtol=0, atol=0)
        self.assertEqual(mean.primitive_dim, 2)
        self.assertIn("decay", mean.state_dict())

    def test_any_leading_dimensions_and_bias_gradients(self):
        for chunk_length in (1, 4, 16):
            with self.subTest(chunk_length=chunk_length):
                mean = DecayingChunkMean(5, 3 * chunk_length, chunk_length, 0.7)
                for shape in ((5,), (2, 5), (2, 3, 5)):
                    inputs = torch.randn(shape)
                    actual = mean(inputs)
                    expected = torch.cat([
                        head.bias + 0.7 ** k * (inputs @ head.weight.T)
                        for k, head in enumerate(mean.heads)
                    ], dim=-1)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(actual.shape, (*shape[:-1], 3 * chunk_length))
                mean(torch.randn(2, 3, 5)).sum().backward()
                for head in mean.heads:
                    torch.testing.assert_close(head.bias.grad, torch.full((3,), 6.0))

    def test_invalid_dimensions_and_rho(self):
        for kwargs in (
            {"chunk_length": 0}, {"chunk_length": -1}, {"chunk_length": True},
            {"chunk_length": 2.0}, {"chunk_length": 4}, {"action_dim": 0},
            {"action_dim": True}, {"latent_dim": 0}, {"rho": 0}, {"rho": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecayingChunkMean(**{
                    **dict(latent_dim=5, action_dim=6, chunk_length=2, rho=0.8), **kwargs
                })

    def test_deterministic_actor_preserves_tensor_interface_and_action_bounds(self):
        for chunk_length in (1, 4, 16):
            actor = make_actor(chunk_length)
            observations = np.random.default_rng(1).normal(size=(7, 5)).astype(np.float64)
            actions = actor(observations)
            self.assertEqual(actions.shape, (7, 3 * chunk_length))
            self.assertEqual(actions.dtype, torch.float32)
            self.assertEqual(actions.device.type, "cpu")
            self.assertTrue(torch.all(actions.abs() <= 1.5))
            expected = 1.5 * actor.last(actor.backbone(torch.tensor(observations).float())).tanh()
            torch.testing.assert_close(actions, expected)


class SpectralNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def assertStateEqual(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)

    def test_attach_after_initialization_preserves_raw_weights_and_biases(self):
        for initialization in ("orthogonal", "small-uniform"):
            actor = make_actor()
            before = []
            for layer in actor.modules():
                if isinstance(layer, nn.Linear):
                    if initialization == "orthogonal":
                        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    else:
                        nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    nn.init.constant_(layer.bias, 0.25)
                    before.append((layer.weight.clone(), layer.bias.clone()))
            apply_chunk_spectral_norm(actor)
            actor.eval()
            for layer, (weight, bias) in zip(normalized_linears(actor), before):
                torch.testing.assert_close(layer.parametrizations.weight.original, weight, rtol=0, atol=0)
                torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
                self.assertLess(abs(torch.linalg.matrix_norm(layer.weight, 2).item() - 1), 0.05)
                self.assertEqual(layer.parametrizations.weight[0].n_power_iterations, 1)
            optimizer = torch.optim.Adam(actor.parameters())
            self.assertEqual({id(p) for p in actor.parameters()},
                             {id(p) for p in optimizer.param_groups[0]["params"]})

    def test_probabilistic_mean_only_and_standard_deviation_unchanged(self):
        for distribution, conditioned in ((DiagGaussian, False), (TanhDiagGaussian, True)):
            backbone = MLP(5, (32, 32))
            dist = distribution(32, 12, unbounded=True, conditioned_sigma=conditioned)
            dist.mu = DecayingChunkMean(32, 12, 4, 0.8)
            actor = ActorProb(backbone, dist)
            sigma_state = snapshot(dist.sigma) if conditioned else dist.sigma_param.detach().clone()
            apply_chunk_spectral_norm(actor)
            self.assertEqual(len(normalized_linears(actor)), 6)
            self.assertTrue(all(parametrize.is_parametrized(h, "weight") for h in dist.mu.heads))
            if conditioned:
                self.assertFalse(parametrize.is_parametrized(dist.sigma))
                self.assertStateEqual(snapshot(dist.sigma), sigma_state)
            else:
                torch.testing.assert_close(dist.sigma_param, sigma_state, rtol=0, atol=0)
            observations = torch.randn(7, 5)
            # Compare distribution wiring at the same effective weights.
            actor.eval()
            logits = actor.backbone(observations)
            actual = actor(observations)
            expected_scale = (dist.sigma(logits).clamp(-5, 2).exp() if conditioned
                              else dist.sigma_param.T.exp().expand(7, 12))
            torch.testing.assert_close(actual.scale, expected_scale)
            torch.testing.assert_close(actual.mean, dist.mu(logits))

    def test_training_updates_estimates_and_eval_freezes_them(self):
        actor = make_actor()
        apply_chunk_spectral_norm(actor)
        with torch.no_grad():
            for layer in normalized_linears(actor):
                layer.parametrizations.weight.original.add_(torch.randn_like(layer.weight) * 0.1)
        before = snapshot(actor)
        actor(torch.randn(4, 5))
        after_train = snapshot(actor)
        self.assertTrue(any(not torch.equal(before[k], after_train[k]) for k in before if k.endswith("._u")))
        actor.eval()
        for _ in range(3):
            actor(torch.randn(4, 5))
        self.assertStateEqual(snapshot(actor), after_train)

    def test_multiple_forwards_backward_and_optimizer_steps(self):
        for chunk_length in (1, 4, 16):
            actor = make_actor(chunk_length)
            apply_chunk_spectral_norm(actor)
            optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
            for _ in range(3):
                optimizer.zero_grad()
                first = actor(torch.randn(7, 5))
                second = actor(torch.randn(7, 5))
                loss = first.square().mean() + second.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in actor.parameters()))
                optimizer.step()

    def test_measured_unit_target_error_before_and_after_optimizer_updates(self):
        initial_errors, updated_errors = [], []
        for chunk_length in (1, 4, 16):
            actor = make_actor(chunk_length)
            apply_chunk_spectral_norm(actor)
            actor.eval()
            initial_errors.extend(abs(torch.linalg.matrix_norm(layer.weight, 2).item() - 1)
                                  for layer in normalized_linears(actor))
            optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
            actor.train()
            for _ in range(8):
                optimizer.zero_grad()
                actor(torch.randn(16, 5)).square().mean().backward()
                optimizer.step()
            actor.eval()
            updated_errors.extend(abs(torch.linalg.matrix_norm(layer.weight, 2).item() - 1)
                                  for layer in normalized_linears(actor))
            # Exact SVD is diagnostic only. The production normalizer uses
            # PyTorch's one-step estimate and may slightly exceed unit norm.
            for k, head in enumerate(actor.last.heads):
                slope_norm = torch.linalg.matrix_norm(actor.last.decay[k] * head.weight, 2)
                self.assertLessEqual(slope_norm.item(), 1.05 * actor.last.rho ** k)
        self.assertLess(max(initial_errors), 0.05)
        self.assertLess(max(updated_errors), 0.05)
        print(f"\nChunk SN max unit-target error: initial={max(initial_errors):.6f}, "
              f"after 8 Adam steps={max(updated_errors):.6f}")

    def test_refresh_preserves_raw_parameters_and_all_module_modes(self):
        actor = make_actor()
        apply_chunk_spectral_norm(actor)
        actor.eval()
        actor.last.heads[0].parametrizations.weight[0].train()
        modes = {name: module.training for name, module in actor.named_modules()}
        before_parameters = {name: value.clone() for name, value in actor.named_parameters()}
        refresh_chunk_spectral_norm(actor)
        self.assertEqual({name: module.training for name, module in actor.named_modules()}, modes)
        self.assertStateEqual(dict(actor.named_parameters()), before_parameters)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_state_round_trip_is_exact_without_load_or_eval_recalibration(self):
        actor = make_actor()
        apply_chunk_spectral_norm(actor)
        actor(torch.randn(4, 5))
        actor.eval()
        observations = torch.randn(7, 5)
        expected = actor(observations)
        saved_state = snapshot(actor)
        stream = io.BytesIO()
        torch.save(actor.state_dict(), stream)
        stream.seek(0)
        restored = make_actor()
        apply_chunk_spectral_norm(restored)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertStateEqual(snapshot(restored), saved_state)
        restored.eval()
        for _ in range(3):
            torch.testing.assert_close(restored(observations), expected, rtol=0, atol=0)
        self.assertStateEqual(snapshot(restored), saved_state)

    def test_td3_target_refresh_uses_ema_weights_and_preserves_eval_mode(self):
        actor = make_actor(hidden_dims=(8,))
        apply_chunk_spectral_norm(actor)
        critic1, critic2 = [Critic(MLP(17, (8,))) for _ in range(2)]
        policy = ChunkTD3BCPolicy(
            actor, critic1, critic2,
            torch.optim.Adam(actor.parameters()),
            torch.optim.Adam(critic1.parameters()), torch.optim.Adam(critic2.parameters()),
            tau=0.25,
        )
        old_parameters = {name: value.clone() for name, value in policy.actor_old.named_parameters()}
        target_linears = normalized_linears(policy.actor_old)
        previous_v = [layer.parametrizations.weight[0]._v.clone() for layer in target_linears]
        with torch.no_grad():
            for layer in normalized_linears(policy.actor):
                layer.parametrizations.weight.original.add_(torch.randn_like(layer.weight))
        refresh_chunk_spectral_norm(policy.actor)
        online_state = snapshot(policy.actor)
        expected_parameters = {
            name: old_parameters[name] * 0.75 + value * 0.25
            for name, value in policy.actor.named_parameters()
        }
        policy._sync_weight()
        self.assertStateEqual(dict(policy.actor_old.named_parameters()), expected_parameters)
        self.assertStateEqual(snapshot(policy.actor), online_state)
        self.assertTrue(all(not module.training for module in policy.actor_old.modules()))
        for layer, old_v in zip(target_linears, previous_v):
            parametrization = layer.parametrizations.weight[0]
            target_weight = layer.parametrizations.weight.original.detach()
            expected_u = F.normalize(target_weight @ old_v, dim=0, eps=1e-12)
            expected_v = F.normalize(target_weight.T @ expected_u, dim=0, eps=1e-12)
            torch.testing.assert_close(parametrization._u, expected_u)
            torch.testing.assert_close(parametrization._v, expected_v)
        self.assertTrue(any(
            not torch.equal(old.parametrizations.weight[0]._v, new.parametrizations.weight[0]._v)
            for old, new in zip(target_linears, normalized_linears(policy.actor))
        ))
        target_state = snapshot(policy.actor_old)
        policy.actor_old(torch.randn(4, 5))
        self.assertStateEqual(snapshot(policy.actor_old), target_state)


if __name__ == "__main__":
    unittest.main()

"""Global sensitivity-gain, compatibility, and real-policy tests for chunk C."""

import copy
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import eval as evaluation
import plot
import policies
import sweep
import test_chunk_policy_pipeline as pipeline
from chunk_policy import (
    DecayingActor,
    DecayingChunkMean,
    NormalizedObservationBackbone,
    apply_chunk_spectral_norm,
    fit_observation_stats,
    get_chunk_policy_config,
    validate_chunk_policy_config,
)
from offlinerlkit.nets import MLP
from test_chunk_policy_integration import SyntheticDynamics
from test_chunk_policy_pipeline import SUPPORTED_ALGOS, TinyEnv, parse_args, policy_args


OBSERVATIONS = np.asarray([
    [1, 4, -2], [3, 2, 0], [-1, 3, 1], [2, -1, 4],
], dtype=np.float32)


def gain_args(algo, c, rho=0.8):
    args = policy_args(algo, rho)
    args.chunk_policy_c = c
    return args


def build_gain_policy(algo, args, length=4, dynamics=False):
    env = TinyEnv(2 * length)
    dataset = {
        "observations": OBSERVATIONS.copy(),
        "next_observations": OBSERVATIONS.copy() + 0.1,
        "actions": np.zeros((len(OBSERVATIONS), 2 * length), dtype=np.float32),
        "rewards": np.zeros(len(OBSERVATIONS), dtype=np.float32),
        "terminals": np.zeros(len(OBSERVATIONS), dtype=np.float32),
    }
    config = get_chunk_policy_config(args, algo)
    normalization = (
        {"observation_stats": fit_observation_stats(OBSERVATIONS)}
        if config is not None and config["version"] == 2 else {}
    )
    if algo in policies.MODEL_FREE_ALGOS:
        buffer = sweep.build_buffer(dataset, env, "cpu")
        policy = policies.build_model_free_policy(
            algo, env, buffer, args, discount=0.99**length,
            chunk_length=length, **normalization,
        )[0]
    else:
        policy = policies.build_model_based_policy(
            algo, env, args, discount=0.99**length, chunk_length=length,
            build_dynamics_model=dynamics, **normalization,
        )[0]
    return policy, dataset, env


class ChunkGainConfigTests(unittest.TestCase):
    def test_cli_default_explicit_gain_and_no_effect_on_standard_schema(self):
        self.assertIsNone(parse_args().chunk_policy_c)
        for algo in SUPPORTED_ALGOS:
            default = policy_args(algo)
            legacy = copy.deepcopy(default)
            del legacy.chunk_policy_c
            self.assertEqual(
                sweep.make_training_schema(algo, "Reacher-v5", {}, 4, default),
                sweep.make_training_schema(algo, "Reacher-v5", {}, 4, legacy),
            )
            for c in (0.5, 1, 2, 5):
                with self.subTest(algo=algo, c=c):
                    args = parse_args(
                        "--algos", algo, "--chunk-policy", "decay-sn",
                        "--chunk-policy-rho", "1", "--chunk-policy-c", str(c),
                    )
                    self.assertEqual(args.chunk_policy_c, c)
                    config = get_chunk_policy_config(args, algo)
                    self.assertEqual(config.get("c", 1), c)
                    self.assertEqual(config["version"], 1 if algo == "td3bc" else 2)
                    self.assertEqual("c" in config, c != 1)

    def test_invalid_cli_gain_or_gain_without_decay_mode_fails(self):
        cases = [["--chunk-policy-c", str(c)] for c in (1, 2)]
        cases.append(["--chunk-policy", "standard", "--chunk-policy-c", "1"])
        cases.extend([
            "--chunk-policy", "decay-sn", "--chunk-policy-rho", "1",
            f"--chunk-policy-c={c}",
        ] for c in ("0", "-1", "nan", "inf", "-inf", "not-a-number"))
        for flags in cases:
            with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args("--algos", "iql", *flags)

    def test_metadata_validates_gain_and_recorded_config_overrides_cli(self):
        for algo in SUPPORTED_ALGOS:
            for version in ((1,) if algo == "td3bc" else (1, 2)):
                legacy = {"version": version, "type": "decay-sn", "rho": 0.8}
                self.assertEqual(validate_chunk_policy_config(legacy, algo), legacy)
                self.assertEqual(validate_chunk_policy_config({**legacy, "c": 1}, algo), legacy)
                for c in (None, False, True, "2", 0, -1, math.inf, -math.inf, math.nan):
                    with self.subTest(algo=algo, version=version, c=c), self.assertRaisesRegex(ValueError, "[cC]"):
                        validate_chunk_policy_config({**legacy, "c": c}, algo)
                args = gain_args(algo, 5, rho=1)
                args.chunk_policy_config = {**legacy, "c": 2}
                self.assertEqual(get_chunk_policy_config(args, algo), {**legacy, "c": 2.0})

    def test_unit_gain_reuses_exact_schema_and_other_gains_have_distinct_cache_entries(self):
        for algo in SUPPORTED_ALGOS:
            with self.subTest(algo=algo), tempfile.TemporaryDirectory() as directory:
                schemas = [
                    sweep.make_training_schema(algo, "Reacher-v5", {}, 4, gain_args(algo, c))
                    for c in (None, 1, 2, 5)
                ]
                self.assertEqual(schemas[0], schemas[1])
                self.assertNotIn("c", schemas[0]["policy_architecture"])
                root = Path(directory)
                for name, schema in zip(("legacy", "c2", "c5"), (schemas[0], schemas[2], schemas[3])):
                    sweep.write_json_atomic(root / name / "run_manifest.json", {"training_schema": schema})
                with patch.object(sweep, "run_is_complete", return_value=True):
                    for name, schema in zip(("legacy", "legacy", "c2", "c5"), schemas):
                        self.assertEqual(sweep.find_trained_run(root, schema), root / name)
                    unmatched = sweep.make_training_schema(algo, "Reacher-v5", {}, 4, gain_args(algo, 3))
                    self.assertIsNone(sweep.find_trained_run(root, unmatched))

    def test_plot_cohorts_separate_gains_and_null_selects_only_default_decay(self):
        rows = []
        for c in (1, 2, 5):
            for length in (1, 4):
                row = pipeline.ChunkPolicyPlotTests.row(length, rho=0.8)
                row["training_schema"] = sweep.make_training_schema(
                    "iql", "Reacher-v5", {}, length, gain_args("iql", c),
                )
                rows.append(row)
        with self.assertRaisesRegex(ValueError, "multiple seed-averaged configurations"):
            plot.algorithm_groups(rows, "chunk_length")
        cohort = pipeline.ChunkPolicyPlotTests.cohort(*[
            {"algo": "iql", "match": {
                "policy_architecture.type": "decay-sn",
                "policy_architecture.c": c,
            }} for c in (None, 2, 5)
        ])
        standard_rows = [pipeline.ChunkPolicyPlotTests.row(length) for length in (1, 4)]
        selected = plot.select_plot_cohort(rows + standard_rows, cohort)
        self.assertEqual(len(selected), 6)
        groups = plot.algorithm_groups(selected, "chunk_length")
        self.assertEqual(len(groups), 3)
        for (_, grouped_rows), expected_c in zip(groups, (1, 2, 5)):
            self.assertEqual(sorted(row["chunk_length"] for row in grouped_rows), [1, 4])
            self.assertEqual({row["training_schema"]["policy_architecture"].get("c", 1)
                              for row in grouped_rows}, {expected_c})


class ChunkGainFormulaTests(unittest.TestCase):
    def test_gain_scales_state_dependent_term_once_and_never_bias(self):
        for c in (0.5, 1, 2, 5):
            with self.subTest(c=c):
                mean = DecayingChunkMean(2, 6, 3, 0.5, c=c).double()
                with torch.no_grad():
                    for k, head in enumerate(mean.heads):
                        head.weight.copy_(torch.tensor([[k + 1, 0.0], [0.0, k + 2]]))
                        head.bias.copy_(torch.tensor([10.0 + k, -20.0 - k]))
                x = torch.tensor([2.0, 3.0], dtype=torch.float64)
                offsets = torch.cat([head.bias for head in mean.heads])
                slopes = torch.tensor([2, 6, 2, 4.5, 1.5, 3], dtype=torch.float64)
                torch.testing.assert_close(mean(x), offsets + c * slopes, rtol=0, atol=0)
                torch.testing.assert_close(mean(torch.zeros_like(x)), offsets, rtol=0, atol=0)
                torch.testing.assert_close(mean.decay, torch.tensor([c, c / 2, c / 4]).double(), rtol=0, atol=0)
                mean(x.expand(4, -1)).sum().backward()
                for head in mean.heads:
                    torch.testing.assert_close(head.bias.grad, torch.full_like(head.bias, 4), rtol=0, atol=0)

    def test_global_jacobian_bound_is_c_not_c_per_layer(self):
        for c in (1, 2, 5):
            for rho in (0.5, 1):
                for max_action in (1.0, 1.5):
                    with self.subTest(c=c, rho=rho, max_action=max_action):
                        actor = DecayingActor(
                            MLP(2, [2, 2], activation=nn.Identity), 6,
                            max_action, "cpu", 3, rho, c=c,
                        )
                        with torch.no_grad():
                            for layer in actor.modules():
                                if isinstance(layer, nn.Linear):
                                    layer.weight.copy_(torch.eye(2))
                                    layer.bias.zero_()
                        apply_chunk_spectral_norm(actor)
                        actor.eval()
                        origin = torch.zeros(2, requires_grad=True)
                        jacobian = torch.autograd.functional.jacobian(actor, origin).reshape(3, 2, 2)
                        for k, block in enumerate(jacobian):
                            expected = max_action * c * rho**k
                            torch.testing.assert_close(block, torch.eye(2) * expected, rtol=2e-6, atol=2e-6)
                        self.assertTrue((actor(torch.tensor([100.0, -100.0])).abs() <= max_action).all())

    def test_module_rejects_nonpositive_nonfinite_or_nonnumeric_gain(self):
        for c in (None, False, True, "2", 0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(c=c), self.assertRaises(ValueError):
                DecayingChunkMean(2, 6, 3, 0.8, c=c)


class ChunkGainPolicyTests(unittest.TestCase):
    def assert_states_equal(self, before, after):
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)

    def test_omitted_and_explicit_unit_gain_are_bitwise_identical(self):
        for algo in SUPPORTED_ALGOS:
            for standard in (False, True):
                with self.subTest(algo=algo, standard=standard):
                    variants = []
                    for setting in ("missing", None, 1):
                        if standard and setting == 1:
                            continue
                        args = gain_args(algo, setting, rho=None if standard else 0.8)
                        if setting == "missing":
                            del args.chunk_policy_c
                        torch.manual_seed(53)
                        policy, _, _ = build_gain_policy(algo, args)
                        policy.eval()
                        variants.append((copy.deepcopy(policy.state_dict()), policy.select_action(OBSERVATIONS, deterministic=True)))
                    for state, actions in variants[1:]:
                        self.assert_states_equal(variants[0][0], state)
                        np.testing.assert_array_equal(variants[0][1], actions)

    def test_gain_changes_only_existing_decay_buffers_not_sigma_critics_or_dynamics(self):
        for algo in SUPPORTED_ALGOS:
            with self.subTest(algo=algo):
                variants = []
                for c in (1, 5):
                    torch.manual_seed(59)
                    policy, _, _ = build_gain_policy(algo, gain_args(algo, c), dynamics=True)
                    policy.eval()
                    dynamics = getattr(policy, "dynamics", None)
                    variants.append((copy.deepcopy(policy.state_dict()),
                                     copy.deepcopy(dynamics.model.state_dict()) if dynamics else None))
                before, after = variants[0][0], variants[1][0]
                self.assertEqual(before.keys(), after.keys())
                self.assertTrue(any(key.endswith(".decay") for key in before))
                for key in before:
                    torch.testing.assert_close(after[key], before[key] * 5 if key.endswith(".decay") else before[key],
                                               rtol=0, atol=0, msg=key)
                if variants[0][1] is not None:
                    self.assert_states_equal(variants[0][1], variants[1][1])

    def test_real_learning_steps_and_manifest_loader_roundtrip_all_algorithms(self):
        for algo in SUPPORTED_ALGOS:
            for length, c in ((1, 2), (4, 5)):
                with self.subTest(algo=algo, length=length, c=c), tempfile.TemporaryDirectory() as directory:
                    args = gain_args(algo, c, rho=1)
                    torch.manual_seed(61)
                    policy, dataset, env = build_gain_policy(algo, args, length)
                    if algo in ("mopo", "mobile"):
                        policy.dynamics = SyntheticDynamics()
                    before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
                    batch = {
                        "observations": torch.tensor(OBSERVATIONS),
                        "next_observations": torch.tensor(OBSERVATIONS + 0.1),
                        "actions": torch.rand(len(OBSERVATIONS), 2 * length) * 1.8 - 0.9,
                        "rewards": torch.ones(len(OBSERVATIONS), 1),
                        "terminals": torch.zeros(len(OBSERVATIONS), 1),
                    }
                    policy.train()
                    for _ in range(4):
                        losses = policy.learn({"real": batch, "fake": batch} if algo in ("mopo", "mobile") else batch)
                        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, policy.actor.parameters())))
                    self.assertTrue(all(torch.isfinite(value).all() for value in policy.state_dict().values()))
                    policy.eval()
                    actions = policy.select_action(OBSERVATIONS, deterministic=True)
                    self.assertEqual(actions.shape, (len(OBSERVATIONS), 2 * length))
                    self.assertTrue((np.abs(actions) <= 1).all())
                    schema = sweep.make_training_schema(algo, "Reacher-v5", {}, length, args)
                    manifest = {
                        "algo": algo, "epoch": schema["epoch"], "step_per_epoch": schema["step_per_epoch"],
                        "chunk_length": length, "base_discount": 0.99, "macro_discount": 0.99**length,
                        "training_schema": schema,
                    }
                    path = Path(directory) / "policy.pth"
                    torch.save(policy.state_dict(), path)
                    loader_dataset = dict(dataset)
                    if algo != "td3bc":
                        loader_dataset["observations"] = OBSERVATIONS * 100 - 50
                    restored, dynamics = evaluation.load_policy_and_dynamics(manifest, "cpu", path, None, loader_dataset, env)
                    self.assertIsNone(dynamics)
                    self.assert_states_equal(policy.state_dict(), restored.state_dict())
                    np.testing.assert_array_equal(restored.select_action(OBSERVATIONS, deterministic=True), actions)
                    if algo != "td3bc":
                        normalizers = [module for module in restored.modules() if isinstance(module, NormalizedObservationBackbone)]
                        self.assertGreater(len(normalizers), 1)
                        mean, std = fit_observation_stats(OBSERVATIONS)
                        for normalizer in normalizers:
                            np.testing.assert_array_equal(normalizer.mean.numpy(), mean)
                            np.testing.assert_array_equal(normalizer.std.numpy(), std)


if __name__ == "__main__":
    unittest.main()

"""CPU-only parity and isolation checks for the additive Walker MF launcher."""

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from offlinerlkit.buffer import ReplayBuffer
from policies import build_model_free_policy
import walker_composition_mf as sweep


ALGORITHM_FIELDS = {
    "iql": ("iql_temperature", "iql_expectile", "iql_learning_rate",
            "iql_lr_schedule", "iql_hidden_dims"),
    "td3bc": ("td3bc_learning_rate", "td3bc_alpha", "td3bc_hidden_dims"),
}


def training_data():
    rng = np.random.RandomState(29)
    observations = (rng.normal(size=(48, 17)) * np.linspace(0.2, 2, 17) + 5).astype(np.float32)
    # A constant coordinate must use the existing builder's +1e-3 convention.
    observations[:, 0] = 2
    return {
        "observations": observations,
        "next_observations": observations + rng.normal(0, 0.1, observations.shape).astype(np.float32),
        "actions": rng.uniform(-0.8, 0.8, (48, 6)).astype(np.float32),
        "rewards": rng.normal(1, 0.1, 48).astype(np.float32),
        "terminals": np.isin(np.arange(48), [11, 23, 35, 47]).astype(np.float32),
        "episode_ids": np.repeat(np.arange(4), 12),
    }


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def small_args(*extra):
    return sweep.parse_args([
        "--device", "cpu", "--epoch", "3", "--step-per-epoch", "3",
        "--batch-size", "8", "--iql-hidden-dims", "16", "16",
        "--td3bc-hidden-dims", "16", "16", "--quiet", *extra,
    ])


def direct_build(args, algorithm, data):
    env = SimpleNamespace(
        spec=SimpleNamespace(id="walker2d-medium-v2"),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
    )
    buffer = ReplayBuffer(len(data["rewards"]), (17,), np.float32, 6, np.float32, "cpu")
    buffer.load_dataset(data)
    policy, scheduler = build_model_free_policy(algorithm, env, buffer, args, 0.99)
    return policy, buffer, scheduler


class ModelFreeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_policy_equal(self, actual, expected):
        actual_state, expected_state = actual.state_dict(), expected.state_dict()
        self.assertEqual(set(actual_state) - {"observation_mean", "observation_std"}, set(expected_state))
        for key, value in expected_state.items():
            self.assertTrue(torch.equal(actual_state[key], value), key)

    def test_algorithm_defaults_match_ordinary_sweep(self):
        import sweep as ordinary_sweep

        with patch("sys.argv", ["sweep.py", "--env", "Walker2d-v5", "--device", "cpu"]):
            ordinary_args = ordinary_sweep.parse_args()
        args = sweep.parse_args(["--device", "cpu"])
        for algorithm, fields in ALGORITHM_FIELDS.items():
            config = sweep.training_config(args, algorithm)
            for field in fields:
                with self.subTest(algorithm=algorithm, field=field):
                    self.assertEqual(getattr(args, field), getattr(ordinary_args, field))
                    self.assertEqual(config[field], getattr(ordinary_args, field))

    def test_configuration_contains_only_own_algorithm_parameters(self):
        baseline = small_args()
        for algorithm, other in (("iql", "td3bc"), ("td3bc", "iql")):
            changed = small_args()
            for field in ALGORITHM_FIELDS[other]:
                old = getattr(changed, field)
                value = [32, 16] if isinstance(old, list) else "constant" if isinstance(old, str) else old / 2
                setattr(changed, field, value)
            self.assertEqual(sweep.training_config(baseline, algorithm), sweep.training_config(changed, algorithm))
            self.assertFalse(set(ALGORITHM_FIELDS[other]) & set(sweep.training_config(changed, algorithm)))
            self.assertFalse({"real_ratio", "rollout_length", "dynamics_max_epochs", "penalty_coef"}
                             & set(sweep.training_config(changed, algorithm)))

    def test_initialization_and_random_stream_match_existing_builder(self):
        args, data = small_args(), training_data()
        for algorithm in ALGORITHM_FIELDS:
            with self.subTest(algorithm=algorithm):
                seed_rng(57)
                expected, expected_buffer, expected_scheduler = direct_build(args, algorithm, data)
                expected_rng = (random.random(), np.random.random(3), torch.rand(3))
                seed_rng(57)
                actual, actual_buffer, actual_scheduler = sweep.build_policy(args, algorithm, data)
                actual_rng = (random.random(), np.random.random(3), torch.rand(3))
                self.assert_policy_equal(actual, expected)
                self.assertEqual(actual_rng[0], expected_rng[0])
                np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
                torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
                for key, value in expected_buffer.sample_all().items():
                    np.testing.assert_array_equal(actual_buffer.sample_all()[key], value)
                if expected_scheduler is None:
                    self.assertIsNone(actual_scheduler)
                else:
                    self.assertEqual(actual_scheduler.state_dict(), expected_scheduler.state_dict())

    def test_fixed_batch_updates_and_scheduler_match_existing_builder(self):
        data = training_data()
        for algorithm, schedule in (("iql", "cosine"), ("iql", "constant"), ("td3bc", "cosine")):
            args = small_args("--iql-lr-schedule", schedule)
            with self.subTest(algorithm=algorithm, schedule=schedule):
                seed_rng(61)
                expected, expected_buffer, expected_scheduler = direct_build(args, algorithm, data)
                seed_rng(61)
                actual, actual_buffer, actual_scheduler = sweep.build_policy(args, algorithm, data)
                initial = {key: value.clone() for key, value in actual.actor.state_dict().items()}
                for epoch in range(3):
                    for update in range(3):
                        rng_seed = 100 + epoch * 3 + update
                        seed_rng(rng_seed)
                        expected_loss = expected.learn(expected_buffer.sample(8))
                        seed_rng(rng_seed)
                        actual_loss = actual.learn(actual_buffer.sample(8))
                        self.assertEqual(actual_loss, expected_loss)
                        self.assertTrue(all(np.isfinite(value) for value in actual_loss.values()))
                        self.assert_policy_equal(actual, expected)
                    if expected_scheduler is not None:
                        expected_scheduler.step()
                        actual_scheduler.step()
                        self.assertEqual(actual_scheduler.state_dict(), expected_scheduler.state_dict())
                    self.assertEqual(actual.actor_optim.param_groups[0]["lr"],
                                     expected.actor_optim.param_groups[0]["lr"])
                self.assertTrue(any(not torch.equal(initial[key], value)
                                    for key, value in actual.actor.state_dict().items()))

    def test_normalization_uses_only_input_training_observations_once(self):
        data, args = training_data(), small_args()
        before = {key: value.copy() for key, value in data.items()}
        # A disjoint, extreme holdout must not enter training statistics.
        held_out = {key: value.copy() for key, value in data.items()}
        held_out["observations"] += 10000
        policy, buffer, _ = sweep.build_policy(args, "td3bc", data)
        mean = data["observations"].mean(0, keepdims=True)
        std = data["observations"].std(0, keepdims=True) + 1e-3
        np.testing.assert_array_equal(policy.scaler.mu, mean)
        np.testing.assert_array_equal(policy.scaler.std, std)
        self.assertEqual(tuple(policy.observation_mean.shape), (1, 17))
        self.assertEqual(tuple(policy.observation_std.shape), (1, 17))
        np.testing.assert_array_equal(policy.observation_mean.numpy(), mean)
        np.testing.assert_array_equal(policy.observation_std.numpy(), std)
        np.testing.assert_array_equal(buffer.observations, (data["observations"] - mean) / std)
        np.testing.assert_array_equal(buffer.next_observations, (data["next_observations"] - mean) / std)
        self.assertFalse(np.allclose(mean, np.concatenate([data["observations"], held_out["observations"]]).mean(0)))
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        for field in ("observation_mean", "observation_std"):
            self.assertIn(field, dict(policy.named_buffers()))
            self.assertNotIn(field, dict(policy.named_parameters()))
            self.assertFalse(getattr(policy, field).requires_grad)

    def test_algorithm_order_cannot_mutate_or_normalize_cached_dataset(self):
        args, data = small_args(), training_data()
        pristine = {key: value.copy() for key, value in data.items()}
        for order in (("td3bc", "iql"), ("iql", "td3bc")):
            buffers = []
            for algorithm in order:
                _, buffer, _ = sweep.build_policy(args, algorithm, data)
                buffers.append(buffer)
                for key in data:
                    np.testing.assert_array_equal(data[key], pristine[key])
                self.assertFalse(np.shares_memory(data["observations"], buffer.observations))
                self.assertFalse(np.shares_memory(data["next_observations"], buffer.next_observations))
                if algorithm == "iql":
                    np.testing.assert_array_equal(buffer.observations, pristine["observations"])
            self.assertFalse(np.shares_memory(buffers[0].observations, buffers[1].observations))

    def test_checkpoint_contains_normalization_and_preserves_actions(self):
        from walker_composition_mf_eval import load_action_fn

        args, data = small_args(), training_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pth"
            for algorithm in ALGORITHM_FIELDS:
                policy, buffer, _ = sweep.build_policy(args, algorithm, data)
                for _ in range(3):
                    policy.learn(buffer.sample(8))
                torch.save(policy.state_dict(), path)
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual({"observation_mean", "observation_std"} <= set(state), algorithm == "td3bc")
                action_fn = load_action_fn(path, algorithm, sweep.training_config(args, algorithm))
                np.testing.assert_allclose(action_fn(data["observations"]),
                                           policy.select_action(data["observations"], deterministic=True),
                                           rtol=0, atol=0)

    def test_training_constructs_no_dynamics_or_real_environment(self):
        args, data = small_args(), training_data()
        with tempfile.TemporaryDirectory() as tmp:
            for algorithm in ALGORITHM_FIELDS:
                with patch("policies.build_dynamics", side_effect=AssertionError("dynamics constructed")), \
                     patch("policies.build_model_based_policy", side_effect=AssertionError("model-based policy constructed")), \
                     patch("gymnasium.make", side_effect=AssertionError("environment constructed")), \
                     patch("subprocess.run", side_effect=AssertionError("worker spawned")), \
                     redirect_stdout(io.StringIO()):
                    target = Path(tmp) / algorithm
                    target.mkdir()
                    sweep.train(args, algorithm, 93, data, target)
                    self.assertTrue((target / "model" / "policy.pth").is_file())
                    self.assertFalse(list(target.rglob("*dynamics*")))

    def test_trainer_receives_quiet_flag_and_requested_milestones(self):
        args = small_args("--epoch", "12", "--step-per-epoch", "2", "--checkpoint-eval-episodes", "10")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("offlinerlkit.policy_trainer.MFPolicyTrainer") as trainer, \
             patch("offlinerlkit.utils.logger.Logger"):
            sweep.train(args, "iql", 7, training_data(), Path(tmp))
        kwargs = trainer.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_epochs"], [2, 3, 4, 5, 6, 8, 9, 10, 11])
        self.assertEqual((kwargs["epoch"], kwargs["step_per_epoch"], kwargs["batch_size"]), (12, 2, 8))
        self.assertFalse(kwargs["show_progress"])
        self.assertEqual(kwargs["lr_scheduler"].T_max, 12)
        trainer.return_value.train.assert_called_once_with()

    def test_zero_checkpoint_budget_has_no_milestones_or_evaluation(self):
        args = small_args("--checkpoint-eval-episodes", "0")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("offlinerlkit.policy_trainer.MFPolicyTrainer") as trainer, \
             patch("offlinerlkit.utils.logger.Logger"), \
             patch("subprocess.run", side_effect=AssertionError("unexpected evaluation")):
            sweep.train(args, "td3bc", 7, training_data(), Path(tmp))
        self.assertEqual(trainer.call_args.kwargs["checkpoint_epochs"], [])
        self.assertIsNone(trainer.call_args.kwargs["lr_scheduler"])

    def test_training_logs_finite_losses_and_saves_all_requested_checkpoints(self):
        args = small_args("--checkpoint-eval-episodes", "1")
        with tempfile.TemporaryDirectory() as tmp:
            for algorithm in ALGORITHM_FIELDS:
                with redirect_stdout(io.StringIO()):
                    target = Path(tmp) / algorithm
                    target.mkdir()
                    sweep.train(args, algorithm, 19, training_data(), target)
                self.assertTrue((target / "checkpoint" / "step_3" / "policy.pth").is_file())
                self.assertTrue((target / "checkpoint" / "step_6" / "policy.pth").is_file())
                self.assertFalse((target / "checkpoint" / "step_9").exists())
                with (target / "record" / "policy_training_progress.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                losses = [float(value) for row in rows for key, value in row.items() if key.startswith("loss/")]
                self.assertTrue(losses)
                self.assertTrue(all(np.isfinite(losses)))
                final = torch.load(target / "model" / "policy.pth", map_location="cpu", weights_only=True)
                last = torch.load(target / "checkpoint" / "policy.pth", map_location="cpu", weights_only=True)
                self.assertEqual(set(final), set(last))
                for key in final:
                    self.assertTrue(torch.equal(final[key], last[key]), key)

    def test_train_seeding_is_reproducible_despite_prior_random_activity(self):
        args, data = small_args(), training_data()
        with tempfile.TemporaryDirectory() as tmp:
            for algorithm in ALGORITHM_FIELDS:
                snapshots = []
                for repetition in range(2):
                    seed_rng(1000 + repetition)
                    np.random.random(100)
                    torch.rand(100)
                    target = Path(tmp) / f"{algorithm}_{repetition}"
                    target.mkdir()
                    with redirect_stdout(io.StringIO()):
                        sweep.train(args, algorithm, 37, data, target)
                    snapshots.append(torch.load(target / "model" / "policy.pth",
                                                map_location="cpu", weights_only=True))
                for key, value in snapshots[0].items():
                    self.assertTrue(torch.equal(value, snapshots[1][key]), (algorithm, key))


if __name__ == "__main__":
    unittest.main()

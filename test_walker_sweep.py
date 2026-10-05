from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from offlinerlkit.utils.load_dataset import qlearning_dataset

import walker_reference
import walker_sweep


def sample_raw():
    n = 12
    return {"observations": np.arange(n * 17, dtype=np.float32).reshape(n, 17),
            "next_observations": np.arange(n * 17, dtype=np.float32).reshape(n, 17) + 0.5,
            "actions": np.zeros((n, 6), dtype=np.float32),
            "rewards": np.arange(n, dtype=np.float32),
            "terminals": np.isin(np.arange(n), [3, 9]),
            "timeouts": np.isin(np.arange(n), [7])}


class ReferenceDatasetTests(unittest.TestCase):
    def test_exact_reference_arrays_and_raw_episode_boundaries(self):
        raw = sample_raw()
        env = SimpleNamespace(_max_episode_steps=1000)
        expected = qlearning_dataset(env, dataset=raw)
        converted = walker_reference.convert_dataset(env, raw, qlearning_dataset)
        for key in expected:
            np.testing.assert_array_equal(converted[key], expected[key])
            self.assertEqual(converted[key].dtype, expected[key].dtype)
        np.testing.assert_array_equal(converted["episode_ids"], [0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
        self.assertFalse(converted["timeouts"].any())
        self.assertEqual(converted["terminals"].sum(), 2)
        # The unfinished final episode is retained exactly as the reference does.
        self.assertEqual(converted["rewards"][-1], 10)

    def test_converter_drift_is_rejected(self):
        def wrong_converter(env, dataset):
            converted = qlearning_dataset(env, dataset=dataset)
            converted["next_observations"] = converted["observations"].copy()
            return converted

        with self.assertRaisesRegex(RuntimeError, "Reference conversion changed"):
            walker_reference.convert_dataset(None, sample_raw(), wrong_converter)


class WalkerSweepTests(unittest.TestCase):
    def setUp(self):
        self.data = walker_reference.convert_dataset(None, sample_raw(), qlearning_dataset)

    def test_zero_split_preserves_every_array_and_never_calls_shared_splitter(self):
        with patch("rollout.split_dataset", side_effect=AssertionError("should not split")):
            train, test = walker_sweep.split_data(self.data, 0, 1)
        self.assertIs(train, self.data)
        self.assertTrue(all(len(value) == 0 for value in test.values()))

    def test_optional_split_exactly_matches_existing_episode_splitter(self):
        from rollout import split_dataset

        for seed in (0, 1, 99):
            for fraction in (0.2, 0.5):
                expected = split_dataset(self.data, fraction, seed)
                actual = walker_sweep.split_data(self.data, fraction, seed)
                for wanted, got in zip(expected, actual):
                    for key in self.data:
                        np.testing.assert_array_equal(got[key], wanted[key])
                self.assertFalse(set(actual[0]["episode_ids"]) & set(actual[1]["episode_ids"]))
                self.assertEqual(sum(len(part["rewards"]) for part in actual), len(self.data["rewards"]))

    def test_split_does_not_advance_global_rng(self):
        np.random.seed(4)
        expected = np.random.random(5)
        np.random.seed(4)
        walker_sweep.split_data(self.data, 0.2, 5)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_reference_defaults_and_algorithm_specific_dynamics_caps(self):
        args = walker_sweep.parse_args([])
        self.assertEqual((args.epoch, args.step_per_epoch, args.batch_size), (3000, 1000, 256))
        self.assertEqual((args.real_ratio, args.rollout_length, args.penalty_coef), (0.05, 5, 0.5))
        self.assertEqual((args.rollout_batch_size, args.actor_lr, args.critic_lr), (50000, 1e-4, 3e-4))
        self.assertEqual(args.test_fraction, 0)
        self.assertEqual(walker_sweep.dynamics_cap(args, "mobile"), 30)
        self.assertIsNone(walker_sweep.dynamics_cap(args, "mopo"))
        args.dynamics_max_epochs = 0
        self.assertIsNone(walker_sweep.dynamics_cap(args, "mobile"))
        args.dynamics_max_epochs = 1000
        self.assertEqual(walker_sweep.dynamics_cap(args, "mopo"), 1000)

    def test_invalid_arguments_and_other_environments_rejected(self):
        for argv in (["--env", "HalfCheetah-v5"], ["--settings", "minari"],
                     ["--test-fraction", "nan"], ["--test-fraction", "1"],
                     ["--real-ratio", "nan"], ["--actor-lr", "0"],
                     ["--critic-lr", "inf"], ["--penalty-coef", "-1"],
                     ["--epoch", "0"], ["--checkpoint-eval-episodes", "-1"],
                     ["--dynamics-max-epochs", "-1"], ["--seed", "-1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                walker_sweep.parse_args(argv)

    def test_no_evaluation_requests_when_both_counts_are_zero(self):
        args = walker_sweep.parse_args(["--checkpoint-eval-episodes", "0", "--final-eval-episodes", "0"])
        self.assertEqual(walker_sweep.checkpoint_epochs(args), [])
        self.assertEqual(walker_sweep.evaluation_requests(args, Path("/tmp/example"), 1), [])

    def test_checkpoint_and_final_episode_budgets_are_separate(self):
        args = walker_sweep.parse_args(["--epoch", "300", "--checkpoint-eval-episodes", "10",
                                       "--final-eval-episodes", "20"])
        requests = walker_sweep.evaluation_requests(args, Path("/tmp/example"), 7)
        self.assertEqual([entry["epoch"] for entry in requests], list(range(30, 301, 30)))
        self.assertEqual([entry["episodes"] for entry in requests], [10] * 9 + [20])
        self.assertTrue(all(entry["seed"] == 7 for entry in requests))

    def test_dry_run_does_not_make_files_or_spawn_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "not-created"
            with patch("sys.argv", ["walker_sweep.py", "--dry-run", "--storage-root", str(target)]), \
                 patch("walker_sweep.reference_worker", side_effect=AssertionError("worker called")), \
                 redirect_stdout(io.StringIO()):
                walker_sweep.main()
            self.assertFalse(target.exists())

    def test_policy_configuration_uses_reference_termination_not_v5(self):
        from offlinerlkit.utils.termination_fns import termination_fn_walker2d

        args = walker_sweep.parse_args(["--device", "cpu"])
        for algo in ("mobile", "mopo"):
            policy, dynamics, scheduler = walker_sweep.build_policy(args, algo)
            self.assertIs(policy.dynamics, dynamics)
            self.assertIs(dynamics.terminal_fn, termination_fn_walker2d)
            self.assertEqual(policy._gamma, 0.99)
            self.assertEqual(policy._target_entropy, -6)
            self.assertEqual(policy.actor_optim.param_groups[0]["lr"], 1e-4)
            self.assertEqual(scheduler.T_max, 3000)
            if algo == "mobile":
                self.assertTrue(policy._clamp_target_q)
                self.assertEqual(policy._return_shift, 0)
                self.assertEqual(policy._penalty_coef, 0.5)
                self.assertTrue(policy._deteterministic_backup)
                self.assertFalse(policy._max_q_backup)

    def test_saved_actor_reproduces_current_policy_actions(self):
        args = walker_sweep.parse_args(["--device", "cpu"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pth"
            for algo in ("mobile", "mopo"):
                policy, _, _ = walker_sweep.build_policy(args, algo)
                torch.save(policy.state_dict(), path)
                actor = walker_reference.load_actor(path)
                obs = self.data["observations"] / 100
                with torch.inference_mode():
                    actual = actor(obs).mode()[0].numpy()
                np.testing.assert_array_equal(actual, policy.select_action(obs, deterministic=True))


@unittest.skipUnless(os.environ.get("WALKER_REFERENCE_TESTS") == "1", "opt-in legacy runtime integration")
class ReferenceParityTests(unittest.TestCase):
    def test_actor_actions_match_across_runtimes(self):
        args = walker_sweep.parse_args(["--device", "cpu"])
        policy, _, _ = walker_sweep.build_policy(args, "mobile")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(policy.state_dict(), root / "policy.pth")
            observations = sample_raw()["observations"] / 100
            np.save(root / "observations.npy", observations)
            code = textwrap.dedent('''\
                import sys
                from pathlib import Path
                import numpy as np
                import torch
                sys.path.insert(0, sys.argv[1])
                from walker_reference import load_actor
                root = Path(sys.argv[2])
                with torch.inference_mode():
                    actions = load_actor(root / 'policy.pth')(np.load(root / 'observations.npy')).mode()[0]
                np.save(root / 'actions.npy', actions.numpy())
            ''')
            result = subprocess.run([str(args.reference_root / "benchmark-python"), "-c", code,
                                     str(walker_sweep.ROOT), str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            np.testing.assert_array_equal(np.load(root / "actions.npy"),
                                          policy.select_action(observations, deterministic=True))

    def test_mobile_initialization_and_one_update_match_frozen_reference(self):
        # Exercise the AUTHOR'S entrypoint, capturing its policy before training.
        # All patches are process-local test doubles; no upstream files are edited.
        code = textwrap.dedent('''\
            import importlib.util, sys
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch, MagicMock
            import numpy as np
            import torch
            root, out = Path(sys.argv[1]), Path(sys.argv[2])
            sys.path.insert(0, str(root))
            import run_benchmark
            cmd = run_benchmark.command('mobile', 7, 'cpu')
            sys.argv = cmd[1:]
            spec = importlib.util.spec_from_file_location('example', cmd[1])
            example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example)
            env = SimpleNamespace(observation_space=SimpleNamespace(shape=(17,)),
                                  action_space=SimpleNamespace(shape=(6,), high=np.ones(6)),
                                  seed=lambda seed: None)
            with np.load(out / 'batch.npz') as saved:
                data = dict(saved)
            def capture(trainer):
                policy = trainer.policy
                torch.save(policy.state_dict(), out / 'initial.pth')
                torch.save(policy.dynamics.model.state_dict(), out / 'dynamics.pth')
                policy.dynamics.scaler.fit(np.concatenate([data['observations'], data['actions']], -1))
                policy.dynamics.model.eval()
                batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
                for key in ('rewards', 'terminals'):
                    batch[key] = batch[key].reshape(-1, 1)
                torch.manual_seed(77)
                np.random.seed(77)
                losses = policy.learn({'real': batch, 'fake': batch})
                torch.save({'losses': losses, 'state': policy.state_dict()}, out / 'updated.pth')
            with patch.object(example.gym, 'make', return_value=env), \\
                 patch.object(example, 'qlearning_dataset', return_value=data), \\
                 patch.object(example, 'make_log_dirs', return_value=str(out)), \\
                 patch.object(example, 'Logger', return_value=MagicMock()), \\
                 patch.object(example.EnsembleDynamics, 'train'), \\
                 patch.object(example.MBPolicyTrainer, 'train', capture):
                example.train(example.get_args())
        ''')
        args = walker_sweep.parse_args(["--device", "cpu"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = {key: value for key, value in sample_raw().items() if key != "timeouts"}
            data["observations"] = data["observations"] / 100
            data["next_observations"] = data["next_observations"] / 100
            data["actions"] = np.linspace(-0.8, 0.8, 72, dtype=np.float32).reshape(12, 6)
            np.savez(path / "batch.npz", **data)
            result = subprocess.run([str(args.reference_root / "benchmark-python"), "-c", code,
                                     str(args.reference_root), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            torch.manual_seed(7)
            np.random.seed(7)
            policy, dynamics, _ = walker_sweep.build_policy(args, "mobile")
            for module, filename in ((policy, "initial.pth"), (dynamics.model, "dynamics.pth")):
                expected = torch.load(path / filename, weights_only=True)
                self.assertEqual(module.state_dict().keys(), expected.keys())
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            dynamics.scaler.fit(np.concatenate([data["observations"], data["actions"]], -1))
            dynamics.model.eval()
            batch = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}
            for key in ("rewards", "terminals"):
                batch[key] = batch[key].reshape(-1, 1)
            torch.manual_seed(77)
            np.random.seed(77)
            losses = policy.learn({"real": batch, "fake": batch})
            expected = torch.load(path / "updated.pth", weights_only=True)
            for key, value in losses.items():
                self.assertAlmostEqual(value, expected["losses"][key], places=5)
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(value, expected["state"][key], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

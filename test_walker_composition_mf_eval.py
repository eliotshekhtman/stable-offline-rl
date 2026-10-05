"""Temporary-file CPU tests for isolated model-free reference evaluation."""

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import walker_composition_eval as existing
import walker_composition_mf_eval as evaluation
from test_walker_composition_eval import FakeEnv
from walker_composition_data import EXPERT_SHA256
from walker_reference import REFERENCE_MODEL_SHA256


def config(algorithm):
    return {"implementation": evaluation.IMPLEMENTATION, "chunk_length": 1,
            "reward_normalization": False, algorithm + "_hidden_dims": [16, 12]}


def make_policy(algorithm):
    import gymnasium as gym
    import torch
    from offlinerlkit.buffer import ReplayBuffer
    from policies import build_model_free_policy

    torch.set_num_threads(1)
    torch.manual_seed(8)
    rng = np.random.default_rng(3)
    buffer = ReplayBuffer(32, (17,), np.float32, 6, np.float32, "cpu")
    buffer.load_dataset({"observations": rng.normal(3, 2, (32, 17)).astype(np.float32),
                         "next_observations": rng.normal(3, 2, (32, 17)).astype(np.float32),
                         "actions": rng.uniform(-1, 1, (32, 6)).astype(np.float32),
                         "rewards": np.zeros(32), "terminals": np.zeros(32)})
    args = SimpleNamespace(device="cpu", epoch=2, iql_hidden_dims=[16, 12],
                           iql_learning_rate=3e-4, iql_expectile=0.7,
                           iql_temperature=3.0, iql_lr_schedule="cosine",
                           td3bc_hidden_dims=[16, 12], td3bc_learning_rate=3e-4,
                           td3bc_alpha=2.5)
    spaces = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
                             action_space=gym.spaces.Box(-1, 1, (6,)))
    policy, _ = build_model_free_policy(algorithm, spaces, buffer, args, 0.99)
    if algorithm == "td3bc":
        policy.register_buffer("observation_mean", torch.as_tensor(policy.scaler.mu))
        policy.register_buffer("observation_std", torch.as_tensor(policy.scaler.std))
    return policy


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_inference_matches_existing_policy_for_both_observation_dtypes(self):
        import torch
        observations = np.random.default_rng(123).normal(1, 4, (6, 17))
        for algorithm in evaluation.ALGORITHMS:
            policy = make_policy(algorithm)
            path = self.root / (algorithm + ".pth")
            torch.save(policy.state_dict(), path)
            action = evaluation.load_action_fn(path, algorithm, config(algorithm))
            for dtype in (np.float32, np.float64):
                obs = observations.astype(dtype)
                expected = policy.select_action(obs, deterministic=True)
                np.testing.assert_array_equal(action(obs), expected)
                expected_single = policy.select_action(obs[0:1], deterministic=True)[0]
                np.testing.assert_array_equal(action(obs[0]), expected_single)
                self.assertEqual(action(obs[0]).shape, (6,))

    def test_nonfinite_actor_and_missing_actor_keys_are_rejected(self):
        import torch
        policy = make_policy("iql")
        original = policy.state_dict()
        path = self.root / "iql.pth"
        for kind in ("nan", "missing"):
            state = copy.deepcopy(original)
            key = next(key for key in state if key.startswith("actor."))
            if kind == "nan":
                state[key].fill_(float("nan"))
            else:
                del state[key]
            torch.save(state, path)
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                evaluation.load_action_fn(path, "iql", config("iql"))

    def test_td3bc_never_falls_back_for_invalid_normalization(self):
        import torch
        original = make_policy("td3bc").state_dict()
        path = self.root / "td3bc.pth"
        for key, value in (("observation_mean", None), ("observation_std", None),
                           ("observation_mean", torch.zeros(17)),
                           ("observation_std", torch.zeros(1, 17)),
                           ("observation_std", -torch.ones(1, 17)),
                           ("observation_mean", torch.full((1, 17), float("nan"))),
                           ("observation_std", torch.full((1, 17), float("inf")))):
            state = copy.deepcopy(original)
            if value is None:
                del state[key]
            else:
                state[key] = value
            torch.save(state, path)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "normalization|positive"):
                evaluation.load_action_fn(path, "td3bc", config("td3bc"))

    def test_wrong_architecture_metadata_is_rejected(self):
        for algorithm, changes in (("mobile", {}), ("iql", {"implementation": "unknown"}),
                                   ("iql", {"chunk_length": 2}),
                                   ("iql", {"iql_hidden_dims": [0]}),
                                   ("td3bc", {"reward_normalization": True})):
            with self.subTest(algorithm=algorithm, changes=changes), self.assertRaises(ValueError):
                evaluation.load_action_fn(self.root / "absent", algorithm, {**config(algorithm), **changes})

    def test_hidden_dimension_mismatch_cannot_silently_load(self):
        import torch
        path = self.root / "iql.pth"
        torch.save(make_policy("iql").state_dict(), path)
        with self.assertRaises(RuntimeError):
            evaluation.load_action_fn(path, "iql", {**config("iql"), "iql_hidden_dims": [8, 8]})

    def test_wrong_observation_shape_and_nonfinite_input_are_rejected(self):
        import torch
        path = self.root / "iql.pth"
        torch.save(make_policy("iql").state_dict(), path)
        action = evaluation.load_action_fn(path, "iql", config("iql"))
        for obs in (np.zeros(16), np.zeros((1, 1, 17)), np.full(17, np.nan)):
            with self.assertRaises(ValueError):
                action(obs)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "model_free_runs" / "fixture"
        self.run.mkdir(parents=True)
        self.checkpoint = self.run / "checkpoint.pth"
        self.final = self.run / "final.pth"
        self.checkpoint.write_bytes(b"diagnostic fixture")
        self.final.write_bytes(b"final fixture")
        self.manifest = {"protocol": evaluation.RUN_PROTOCOL, "version": 1,
                         "status": "trained", "algorithm": "iql", "training_id": "fixture",
                         "seed": 10000, "training_config": config("iql"),
                         "environment": {"model_xml_sha256": REFERENCE_MODEL_SHA256, "horizon": 1000},
                         "checkpoints": [{"epoch": 30, "step": 30000, "policy_path": str(self.checkpoint)},
                                         {"epoch": 300, "step": 300000, "policy_path": str(self.final), "final": True}]}
        self.manifest_path = self.run / "run_manifest.json"
        evaluation.atomic_json(self.manifest_path, self.manifest)
        self.before = {path.name: path.read_bytes() for path in self.run.iterdir()}
        original_hash = evaluation.file_hash

        def fixture_hash(path):
            return EXPERT_SHA256 if Path(path).name == "walker2d_expert-v2.hdf5" else original_hash(path)

        self.hash_patch = patch.object(evaluation, "file_hash", side_effect=fixture_hash)
        self.hash_patch.start()
        self.worker_patch = patch.object(evaluation.subprocess, "run", side_effect=self.fake_worker)
        self.mock_worker = self.worker_patch.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.hash_patch.stop)
        self.addCleanup(self.worker_patch.stop)

    def fake_worker(self, command, check, env):
        self.assertTrue(check)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(Path(command[1]).name, "walker_composition_mf_eval.py")
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        self.assertEqual(request["algorithm"], self.manifest["algorithm"])
        self.assertEqual(request["training_config"], self.manifest["training_config"])
        records = []
        for item in request["requests"]:
            identity = {key: value for key, value in item.items() if key != "policy_path"}
            records.append({**identity, **existing.episode_statistics(
                FakeEnv(), lambda obs: np.zeros(6), item["reset_seeds"])})
        expert = {**request["expert_request"], **existing.episode_statistics(
            FakeEnv(), lambda obs: np.zeros(6), request["expert_request"]["reset_seeds"])}
        existing.atomic_json(Path(command[command.index("--output") + 1]),
                             {"environment": request["environment"], "records": records, "expert": expert})

    def test_disabled_evaluation_does_not_read_or_spawn(self):
        with patch.object(evaluation, "file_hash", side_effect=AssertionError("No hash")):
            self.assertIsNone(evaluation.evaluate_run(self.root / "missing", 0, 0))
        self.mock_worker.assert_not_called()

    def test_counts_seeds_old_validation_and_read_only_training(self):
        path = evaluation.evaluate_run(self.run, 2, 3)
        report = json.loads(path.read_text())
        self.assertEqual(path.parents[2], self.root / "evals")
        self.assertEqual([item["episodes"] for item in report["records"]], [2, 3])
        self.assertEqual(report["records"][0]["reset_seeds"], [10000, 10001])
        self.assertEqual(report["records"][1]["reset_seeds"], [1010000, 1010001, 1010002])
        self.assertEqual(report["expert"]["reset_seeds"], [1010000, 1010001, 1010002])
        existing.validate_evaluation(report, self.manifest_path, report["evaluation_config"], report["config_id"])
        self.assertEqual(self.before, {path.name: path.read_bytes() for path in self.run.iterdir()})

    def test_final_only_does_not_read_diagnostic_checkpoint(self):
        self.checkpoint.unlink()
        report = json.loads(evaluation.evaluate_run(self.run, 0, 2).read_text())
        self.assertEqual(len(report["records"]), 1)
        self.assertTrue(report["records"][0]["final"])

    def test_reuse_skips_worker_and_validates_records(self):
        path = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        self.assertEqual(evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True), path)
        self.assertEqual(self.mock_worker.call_count, 1)
        report = json.loads(path.read_text())
        report["records"][0]["performance"] = []
        evaluation.atomic_json(path, report)
        with self.assertRaisesRegex(ValueError, "cached evaluation array"):
            evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        self.assertEqual(self.mock_worker.call_count, 1)

    def test_replaced_checkpoint_after_worker_is_rejected(self):
        def replace(command, check, env):
            self.fake_worker(command, check, env)
            self.final.write_bytes(b"replacement")
        self.mock_worker.side_effect = replace
        with self.assertRaisesRegex(ValueError, "changed during"):
            evaluation.evaluate_run(self.run, 0, 1)
        self.assertFalse(list(self.root.rglob("evaluation.json")))

    def test_checkpoint_checksum_and_eval_budget_control_cache_identity(self):
        first = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        second = evaluation.evaluate_run(self.run, 0, 3, reuse_eval=True)
        self.final.write_bytes(b"different saved weights or normalization")
        third = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        self.assertEqual(len({first, second, third}), 3)
        self.manifest["checkpoints"][-1]["policy_sha256"] = evaluation.file_hash(self.final)
        existing.atomic_json(self.manifest_path, self.manifest)
        self.final.write_bytes(b"unexpected modification")
        with self.assertRaisesRegex(ValueError, "training manifest"):
            evaluation.evaluate_run(self.run, 0, 2)

    def test_metadata_and_model_based_runs_rejected_before_worker(self):
        for key, value in (("algorithm", "mobile"), ("status", "training"),
                           ("protocol", "legacy"), ("training_id", "different")):
            evaluation.atomic_json(self.manifest_path, {**self.manifest, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.evaluate_run(self.run, 0, 1)
        self.mock_worker.assert_not_called()

    def test_bulk_discovery_skips_model_based_runs_even_under_broad_root(self):
        old = self.root / "runs" / "old" / "run_manifest.json"
        existing.atomic_json(old, {**self.manifest, "algorithm": "mobile"})
        with redirect_stdout(io.StringIO()), patch.object(evaluation, "evaluate_run", return_value=None) as run:
            evaluation.main(["--root", str(self.root), "--final-eval-episodes", "0"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], self.run)

    def test_config_is_exactly_model_based_protocol_without_architecture_fields(self):
        report = json.loads(evaluation.evaluate_run(self.run, 2, 3).read_text())
        config = report["evaluation_config"]
        expected = {"protocol": existing.PROTOCOL, "metric_version": 1, "metrics": existing.METRICS,
                    "seed_convention": "env.seed(seed + episode); final seed offset 1000000",
                    "seed": 10000, "checkpoint_eval_episodes": 2, "final_eval_episodes": 3,
                    "environment_sha256": existing.identity(self.manifest["environment"]),
                    "checkpoints": [{key: value for key, value in request.items() if key != "policy_path"}
                                    for request in existing.checkpoint_requests(self.manifest, 2, 3, 10000)],
                    "expert": {key: report["expert"][key]
                               for key in ("file_sha256", "actor_sha256", "episodes", "reset_seeds")}}
        self.assertEqual(config, expected)
        self.assertNotIn("training_config", config)
        self.assertNotIn("algorithm", config)


@unittest.skipUnless(os.environ.get("WALKER_COMPOSITION_INTEGRATION") == "1",
                     "Set WALKER_COMPOSITION_INTEGRATION=1 for frozen-runtime CPU tests")
class FrozenRuntimeTests(unittest.TestCase):
    def test_current_policy_actions_match_frozen_inference_and_real_evaluation(self):
        import torch
        with tempfile.TemporaryDirectory(prefix="walker-mf-eval-test-") as temporary:
            root = Path(temporary)
            observations = np.random.default_rng(6).normal(1, 4, (8, 17))
            np.save(root / "observations.npy", observations)
            for algorithm in evaluation.ALGORITHMS:
                policy = make_policy(algorithm)
                run = root / "model_free_runs" / algorithm
                run.mkdir(parents=True)
                torch.save(policy.state_dict(), run / "policy.pth")
                existing.atomic_json(run / "config.json", config(algorithm))
                code = textwrap.dedent("""
                    import json, sys
                    from pathlib import Path
                    import gym, d4rl, numpy as np, torch
                    from walker_composition_mf_eval import load_action_fn
                    from walker_reference import environment_info, TASK
                    torch.set_num_threads(1)
                    root, algorithm = Path(sys.argv[1]), sys.argv[2]
                    run = root / 'model_free_runs' / algorithm
                    fn = load_action_fn(run / 'policy.pth', algorithm,
                                        json.loads((run / 'config.json').read_text()))
                    np.save(run / 'actions.npy', fn(np.load(root / 'observations.npy')))
                    env = gym.make(TASK)
                    try:
                        (run / 'physics.json').write_text(json.dumps(environment_info(env)))
                    finally:
                        env.close()
                """)
                subprocess.run([str(evaluation.DEFAULT_REFERENCE_ROOT / "benchmark-python"),
                                "-c", code, str(root), algorithm], check=True,
                               cwd=Path(__file__).resolve().parent,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
                np.testing.assert_allclose(np.load(run / "actions.npy"),
                                           policy.select_action(observations, deterministic=True),
                                           rtol=0, atol=1e-7)
                manifest = {"protocol": existing.RUN_PROTOCOL, "version": 1, "status": "trained",
                            "algorithm": algorithm, "training_id": algorithm, "seed": 17,
                            "training_config": config(algorithm),
                            "environment": json.loads((run / "physics.json").read_text()),
                            "checkpoints": [{"epoch": 1, "step": 1, "final": True,
                                             "policy_path": str(run / "policy.pth")}]}
                existing.atomic_json(run / "run_manifest.json", manifest)
                path = evaluation.evaluate_run(run, 0, 1, reuse_eval=True)
                report = json.loads(path.read_text())
                self.assertEqual(report["environment"]["foot_friction"], [0.9, 1.9])
                self.assertEqual(report["records"][0]["reset_seeds"], [1000017])
                self.assertEqual(report["records"][0]["episodes"], 1)
                with patch.object(evaluation.subprocess, "run", side_effect=AssertionError("reuse must not spawn")):
                    self.assertEqual(evaluation.evaluate_run(run, 0, 1, reuse_eval=True), path)
            self.assertEqual(len(list((root / "evals" / "experts").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()

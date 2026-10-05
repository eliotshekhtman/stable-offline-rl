"""CPU-only evaluation protocol tests; no existing runs are read or changed."""

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

import walker_composition_eval as evaluation
from walker_reference import REFERENCE_MODEL_SHA256
from walker_composition_data import EXPERT_SHA256


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(1)))
        self._max_episode_steps = 3
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)

    def reset(self):
        self.steps = 0
        self.sim.data.qpos[0] = self.seeds[-1] * 0.1
        return np.zeros(17)

    def step(self, action):
        self.steps += 1
        self.sim.data.qpos[0] += 2.0
        return np.zeros(17), 100.0, self.steps == 3, {}

    def get_normalized_score(self, returns):
        return returns / 1000


class EpisodeMetricTests(unittest.TestCase):
    def test_displacement_is_measured_not_inferred_from_reward(self):
        env = FakeEnv()
        result = evaluation.episode_statistics(env, lambda obs: np.zeros(6), [10000, 10001])
        self.assertEqual(env.seeds, [10000, 10001])
        self.assertEqual(result["performance"], [6.0, 6.0])
        self.assertEqual(result["returns"], [300.0, 300.0])
        self.assertEqual(result["normalized_scores"], [30.0, 30.0])
        self.assertEqual(result["lengths"], [3, 3])
        self.assertEqual(result["performance_mean"], 6.0)
        self.assertEqual(result["std_ddof"], 0)

    def test_seed_independence_of_reset_order(self):
        first = evaluation.episode_statistics(FakeEnv(), lambda obs: np.zeros(6), [7, 8])
        second = evaluation.episode_statistics(FakeEnv(), lambda obs: np.zeros(6), [8, 7])
        self.assertEqual(first["performance"], second["performance"][::-1])

    def test_zero_episodes_and_nonfinite_actions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluation.episode_statistics(FakeEnv(), lambda obs: np.zeros(6), [])
        with self.assertRaisesRegex(ValueError, "Nonfinite evaluation action"):
            evaluation.episode_statistics(FakeEnv(), lambda obs: np.full(6, np.nan), [1])

    def test_horizon_is_enforced(self):
        env = FakeEnv()
        env._max_episode_steps = 2
        with self.assertRaisesRegex(ValueError, "horizon"):
            evaluation.episode_statistics(env, lambda obs: np.zeros(6), [1])


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs" / "training-one"
        self.run.mkdir(parents=True)
        self.checkpoint = self.run / "checkpoint.pth"
        self.final = self.run / "final.pth"
        self.checkpoint.write_bytes(b"checkpoint fixture")
        self.final.write_bytes(b"final fixture")
        self.manifest = {
            "protocol": evaluation.RUN_PROTOCOL, "version": 1, "status": "trained",
            "algorithm": "mobile", "training_id": "training-one", "seed": 10000,
            "environment": {"model_xml_sha256": REFERENCE_MODEL_SHA256, "horizon": 1000},
            "checkpoints": [{"epoch": 30, "step": 30000, "policy_path": str(self.checkpoint)},
                            {"epoch": 300, "step": 300000, "policy_path": str(self.final), "final": True}],
        }
        self.manifest_path = self.run / "run_manifest.json"
        evaluation.atomic_json(self.manifest_path, self.manifest)
        self.original_manifest = self.manifest_path.read_bytes()
        real_hash = evaluation.file_hash

        def fixture_hash(path):
            if Path(path).name == "walker2d_expert-v2.hdf5":
                return EXPERT_SHA256
            return real_hash(path)

        self.hash_patch = patch.object(evaluation, "file_hash", side_effect=fixture_hash)
        self.hash_patch.start()
        self.worker_patch = patch.object(evaluation.subprocess, "run", side_effect=self.fake_worker)
        self.mock_worker = self.worker_patch.start()
        self.addCleanup(self.worker_patch.stop)
        self.addCleanup(self.hash_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def fake_worker(self, command, check, env):
        self.assertTrue(check)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        output_path = Path(command[command.index("--output") + 1])
        records = []
        for item in request["requests"]:
            item = {key: value for key, value in item.items() if key != "policy_path"}
            records.append({**item, **evaluation.episode_statistics(
                FakeEnv(), lambda obs: np.zeros(6), item["reset_seeds"])})
        expert = {**request["expert_request"], **evaluation.episode_statistics(
            FakeEnv(), lambda obs: np.zeros(6), request["expert_request"]["reset_seeds"])}
        evaluation.atomic_json(output_path, {"environment": request["environment"],
                                             "records": records, "expert": expert})

    def test_zero_counts_skip_everything_including_run_discovery(self):
        with patch.object(evaluation, "file_hash", side_effect=AssertionError("hash called")):
            self.assertIsNone(evaluation.evaluate_run(self.root / "not-a-run", 0, 0))
        self.mock_worker.assert_not_called()

    def test_checkpoint_and_final_counts_seeds_and_read_only_run(self):
        path = evaluation.evaluate_run(self.run, 2, 3)
        report = json.loads(path.read_text())
        self.assertEqual([item["episodes"] for item in report["records"]], [2, 3])
        self.assertEqual(report["records"][0]["reset_seeds"], [10000, 10001])
        self.assertEqual(report["records"][1]["reset_seeds"], [1010000, 1010001, 1010002])
        self.assertEqual(report["expert"]["reset_seeds"], [1010000, 1010001, 1010002])
        self.assertEqual(self.manifest_path.read_bytes(), self.original_manifest)
        self.assertEqual(sorted(item.name for item in self.run.iterdir()),
                         ["checkpoint.pth", "final.pth", "run_manifest.json"])

    def test_final_only_never_reads_checkpoint_path(self):
        self.checkpoint.unlink()
        path = evaluation.evaluate_run(self.run, 0, 3)
        report = json.loads(path.read_text())
        self.assertEqual(len(report["records"]), 1)
        self.assertTrue(report["records"][0]["final"])

    def test_checkpoint_only_uses_unoffset_expert_seeds(self):
        self.final.unlink()
        report = json.loads(evaluation.evaluate_run(self.run, 2, 0).read_text())
        self.assertEqual(report["expert"]["reset_seeds"], [10000, 10001])
        self.assertFalse(report["records"][0]["final"])

    def test_reuse_is_checked_and_does_not_spawn_worker(self):
        first = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        content = first.read_bytes()
        second = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        self.assertEqual(first, second)
        self.assertEqual(first.read_bytes(), content)
        self.assertEqual(self.mock_worker.call_count, 1)

    def test_episode_count_seed_checkpoint_and_physics_affect_identity(self):
        paths = [evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True),
                 evaluation.evaluate_run(self.run, 0, 3, reuse_eval=True),
                 evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True, seed=10100)]
        self.final.write_bytes(b"different final fixture")
        paths.append(evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True))
        self.manifest["environment"]["horizon"] = 999
        evaluation.atomic_json(self.manifest_path, self.manifest)
        paths.append(evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True))
        self.assertEqual(len(set(paths)), 5)
        self.assertTrue(all(path.exists() for path in paths))

    def test_corrupt_cached_arrays_and_identity_are_rejected(self):
        path = evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        original = json.loads(path.read_text())
        for modification in ("array", "sha", "config"):
            broken = copy.deepcopy(original)
            if modification == "array":
                broken["records"][0]["performance"] = [1.0]
            elif modification == "sha":
                broken["records"][0]["policy_sha256"] = "different"
            else:
                broken["evaluation_config"]["seed"] = 4
            evaluation.atomic_json(path, broken)
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                evaluation.evaluate_run(self.run, 0, 2, reuse_eval=True)
        self.assertEqual(self.mock_worker.call_count, 1)

    def test_changed_checkpoint_during_worker_is_rejected(self):
        original_worker = self.fake_worker

        def replacing_worker(command, check, env):
            original_worker(command, check, env)
            self.final.write_bytes(b"replacement")

        self.mock_worker.side_effect = replacing_worker
        with self.assertRaisesRegex(ValueError, "changed during"):
            evaluation.evaluate_run(self.run, 0, 2)
        self.assertFalse(list(self.root.rglob("evaluation.json")))

    def test_declared_training_checksum_is_checked_before_worker(self):
        self.manifest["checkpoints"][-1]["policy_sha256"] = evaluation.file_hash(self.final)
        evaluation.atomic_json(self.manifest_path, self.manifest)
        self.final.write_bytes(b"changed after training")
        with self.assertRaisesRegex(ValueError, "training manifest"):
            evaluation.evaluate_run(self.run, 0, 2)
        self.mock_worker.assert_not_called()

    def test_missing_diagnostic_checkpoints_are_explicit(self):
        self.manifest["checkpoints"] = [self.manifest["checkpoints"][-1]]
        with self.assertRaisesRegex(ValueError, "checkpoint-eval-episodes 0"):
            evaluation.checkpoint_requests(self.manifest, 2, 3, 10000)
        self.manifest["checkpoints"][0]["epoch"] = 1
        requests = evaluation.checkpoint_requests(self.manifest, 2, 3, 10000)
        self.assertEqual(len(requests), 1)

    def test_only_new_completed_protocol_runs_are_accepted(self):
        for key, value in (("protocol", "walker-reference-v0"), ("status", "training"),
                           ("algorithm", "iql")):
            broken = copy.deepcopy(self.manifest)
            broken[key] = value
            evaluation.atomic_json(self.manifest_path, broken)
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.evaluate_run(self.run, 0, 1)
        self.mock_worker.assert_not_called()

    def test_batch_cli_ignores_existing_legacy_manifests(self):
        legacy = self.root / "runs" / "legacy"
        legacy.mkdir()
        evaluation.atomic_json(legacy / "run_manifest.json", {"status": "complete"})
        with redirect_stdout(io.StringIO()), patch.object(evaluation, "evaluate_run", return_value=None) as run:
            evaluation.main(["--root", str(self.root), "--final-eval-episodes", "0"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], self.run)

    def test_exactly_one_final_checkpoint_and_valid_counts_required(self):
        for count in (-1,):
            with self.assertRaises(ValueError):
                evaluation.evaluate_run(self.run, count, 3)
        broken = copy.deepcopy(self.manifest)
        broken["checkpoints"][1]["final"] = False
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation.checkpoint_requests(broken, 2, 3, 10000)
        with self.assertRaisesRegex(ValueError, "uint32"):
            evaluation.checkpoint_requests(self.manifest, 2, 3, 2**32)


@unittest.skipUnless(os.environ.get("WALKER_COMPOSITION_INTEGRATION") == "1",
                     "Set WALKER_COMPOSITION_INTEGRATION=1 for frozen-runtime CPU evaluation")
class FrozenRuntimeIntegrationTests(unittest.TestCase):
    def test_random_policy_reference_physics_metrics_and_cache_reuse(self):
        with tempfile.TemporaryDirectory(prefix="walker-evaluation-test-") as temporary:
            run_dir = Path(temporary) / "runs" / "fixture"
            run_dir.mkdir(parents=True)
            # Construct a new random actor, never load or evaluate a user's checkpoint.
            script = textwrap.dedent("""
                import json, sys
                from pathlib import Path
                import gym, d4rl, torch
                from offlinerlkit.modules import ActorProb, TanhDiagGaussian
                from offlinerlkit.nets import MLP
                from walker_reference import environment_info, TASK
                destination = Path(sys.argv[1])
                torch.manual_seed(7)
                actor = ActorProb(MLP(17, [256, 256]), TanhDiagGaussian(
                    256, 6, unbounded=True, conditioned_sigma=True, max_mu=1.0), device='cpu')
                torch.save({'actor.' + k: v for k, v in actor.state_dict().items()},
                           destination / 'policy.pth')
                env = gym.make(TASK)
                try:
                    (destination / 'physics.json').write_text(json.dumps(environment_info(env)))
                finally:
                    env.close()
            """)
            subprocess.run([str(evaluation.DEFAULT_REFERENCE_ROOT / "benchmark-python"),
                            "-c", script, str(run_dir)], check=True, cwd=evaluation.ROOT,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            manifest = {"protocol": evaluation.RUN_PROTOCOL, "version": 1, "status": "trained",
                        "algorithm": "mobile", "seed": 10000, "training_id": "fixture",
                        "environment": json.loads((run_dir / "physics.json").read_text()),
                        "checkpoints": [{"epoch": 1, "step": 1, "final": True,
                                         "policy_path": str(run_dir / "policy.pth")}]}
            evaluation.atomic_json(run_dir / "run_manifest.json", manifest)
            before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            result = evaluation.evaluate_run(run_dir, 0, 1, reuse_eval=True)
            report = json.loads(result.read_text())
            self.assertEqual(report["records"][0]["reset_seeds"], [1010000])
            self.assertEqual(report["expert"]["reset_seeds"], [1010000])
            self.assertEqual(report["environment"]["foot_friction"], [0.9, 1.9])
            self.assertEqual(len(report["expert"]["performance"]), 1)
            self.assertNotEqual(report["expert"]["performance"], report["expert"]["returns"])
            self.assertTrue(list((Path(temporary) / "evals" / "experts").glob("*.json")))
            expert_cache = next((Path(temporary) / "evals" / "experts").glob("*.json"))
            expert_mtime = expert_cache.stat().st_mtime_ns
            second_run = Path(temporary) / "runs" / "fixture-two"
            second_run.mkdir()
            second_manifest = {**manifest, "training_id": "fixture-two"}
            evaluation.atomic_json(second_run / "run_manifest.json", second_manifest)
            second_result = evaluation.evaluate_run(second_run, 0, 1, reuse_eval=True)
            self.assertEqual(report["expert"], json.loads(second_result.read_text())["expert"])
            self.assertEqual(expert_mtime, expert_cache.stat().st_mtime_ns)
            with patch.object(evaluation.subprocess, "run", side_effect=AssertionError("cache missed")):
                self.assertEqual(evaluation.evaluate_run(run_dir, 0, 1, reuse_eval=True), result)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run_dir.iterdir()})


if __name__ == "__main__":
    unittest.main()

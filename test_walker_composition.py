"""Orchestration regression tests; all artifacts live in temporary directories."""

from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

import numpy as np

import walker_composition as sweep
from walker_reference import array_hashes


PROOF = {"environment": {"model_xml_sha256": "reference", "foot_friction": [.9, 1.9]},
         "reference_verification": {"dataset_sha256": "medium", "source_commits": {"kit": "frozen"}}}


def fake_dataset():
    n = 40
    return {"observations": np.arange(n * 17, dtype=np.float32).reshape(n, 17),
            "actions": np.zeros((n, 6), np.float32),
            "next_observations": np.arange(n * 17, dtype=np.float32).reshape(n, 17) + .5,
            "rewards": np.arange(n, dtype=np.float32), "terminals": np.zeros(n, bool),
            "timeouts": np.zeros(n, bool), "episode_ids": np.repeat(np.arange(10), 4)}


def fake_collection(args, operation, spec, output, workdir):
    if operation == "preflight":
        sweep.write_json(output, PROOF)
        return
    data = fake_dataset()
    np.savez(output / "dataset.npz", **data)
    sweep.write_json(output / "metadata.json", {
        **PROOF, "dataset_spec": spec, "array_sha256": array_hashes(data),
        "episode_sources": ["clean"] * 5 + ["noisy"] * 5,
        "expert": {"actor_sha256": "expert"},
        "requested_quotas": sweep.requested_quotas(spec)})


def fake_training(args, algo, seed, data, run_dir):
    for epoch in sweep.walker_sweep.checkpoint_epochs(args) + [args.epoch]:
        policy = run_dir / ("model/policy.pth" if epoch == args.epoch
                            else f"checkpoint/step_{epoch * args.step_per_epoch}/policy.pth")
        policy.parent.mkdir(parents=True, exist_ok=True)
        policy.write_bytes(f"test fixture {algo} {seed} {epoch}".encode())


class GridTests(unittest.TestCase):
    def test_defaults_match_reference_except_requested_collection_and_split(self):
        args = sweep.parse_args([])
        old = sweep.walker_sweep.parse_args([])
        for name in sweep.TRAIN_ARGUMENTS:
            self.assertEqual(getattr(args, name), getattr(old, name), name)
        self.assertEqual(args.num_samples, 999995)
        self.assertEqual(args.test_fraction, .2)
        self.assertEqual(args.seed, [10000])
        self.assertFalse(args.smoke)
        self.assertEqual(sweep.training_config(args, "mobile")["dynamics_max_epochs"], 30)
        self.assertIsNone(sweep.training_config(args, "mopo")["dynamics_max_epochs"])
        self.assertEqual(sweep.training_config(args, "mobile")["rollout_freq"], 1000)

    def test_four_seed_full_grid_has_36_unique_datasets(self):
        args = sweep.parse_args(["--seed", "10000", "10100", "10200", "10300"])
        specs = sweep.dataset_grid(args)
        self.assertEqual(len(specs), 36)
        self.assertEqual(sum(spec["setting"] == "clean" for spec in specs), 4)
        self.assertEqual([sweep.requested_quotas(spec) for spec in specs[:5]],
                         [{"clean": 999995, "other": 0}, {"clean": 749997, "other": 249999},
                          {"clean": 499998, "other": 499998}, {"clean": 249999, "other": 749997},
                          {"clean": 0, "other": 999995}])

    def test_clean_endpoint_identity_shared_across_independent_commands(self):
        specs = [sweep.dataset_grid(sweep.parse_args(["--settings", setting]))[0]
                 for setting in ("noise0.5", "clean-medium-v2", "noise1.0")]
        self.assertEqual(specs[0], specs[1])
        self.assertEqual(len({sweep.identity(sweep.dataset_identity(spec, PROOF)) for spec in specs}), 1)

    def test_seed_composition_split_physics_change_dataset_identity(self):
        spec = sweep.dataset_grid(sweep.parse_args([]))[1]
        original = sweep.identity(sweep.dataset_identity(spec, PROOF))
        for changes in ({"seed": 10100}, {"num_samples": 999994}, {"test_fraction": 0.},
                        {"clean_fraction": .5, "other_fraction": .5}, {"setting": "noise1.0"}):
            self.assertNotEqual(original, sweep.identity(sweep.dataset_identity({**spec, **changes}, PROOF)))
        self.assertNotEqual(original, sweep.identity(sweep.dataset_identity(spec, {**PROOF, "environment": {}})))

    def test_seed_independent_training_config_and_nontraining_flags(self):
        a = sweep.parse_args(["--seed", "10000", "--device", "cpu"])
        b = sweep.parse_args(["--seed", "10100", "--device", "cuda", "--quiet", "--eval",
                              "--checkpoint-eval-episodes", "10", "--final-eval-episodes", "100"])
        self.assertEqual(sweep.training_config(a, "mobile"), sweep.training_config(b, "mobile"))

    def test_invalid_inputs_fail_before_any_collection(self):
        cases = [["--num-samples", "1000000"], ["--composition", ".5", ".6"],
                 ["--composition", "nan", ".5"], ["--composition", "-1", "2"],
                 ["--test-fraction", "nan"], ["--test-fraction", "1"], ["--real-ratio", "nan"],
                 ["--epoch", "0"], ["--num-samples", "0"], ["--actor-lr", "0"],
                 ["--critic-lr", "inf"], ["--penalty-coef", "-1"],
                 ["--seed", "-1"], ["--checkpoint-eval-episodes", "-1"], ["--dynamics-max-epochs", "-1"]]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep.parse_args(argv)
        # The finite medium cap must not restrict a generated-only experiment.
        sweep.parse_args(["--settings", "noise0.5", "--num-samples", "1000000"])

    def test_dry_run_has_no_files_workers_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "untouched"
            with patch.object(sweep, "data_worker", side_effect=AssertionError("worker")), \
                 patch.object(sweep.walker_sweep, "train", side_effect=AssertionError("training")), \
                 redirect_stdout(io.StringIO()) as captured:
                sweep.main(["--dry-run", "--storage-root", str(root), "--seed", "10000", "10100", "10200", "10300"])
            self.assertFalse(root.exists())
            report = json.loads(captured.getvalue())
            self.assertEqual(report["requested_points"], 40)
            self.assertEqual(report["distinct_training_runs"], 36)


class CacheAndTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = sweep.parse_args(["--storage-root", self.temporary.name, "--device", "cpu",
                                     "--settings", "noise0.5", "--composition", ".5", ".5",
                                     "--epoch", "10", "--step-per-epoch", "2", "--quiet"])
        self.spec = sweep.dataset_grid(self.args)[0]

    def prepare(self):
        with patch.object(sweep, "data_worker", side_effect=fake_collection), redirect_stdout(io.StringIO()):
            return sweep.prepare_dataset(self.args, self.spec, PROOF)

    def test_collection_cached_and_split_exactly_matches_existing_splitter(self):
        directory, metadata = self.prepare()
        expected = sweep.walker_sweep.split_data(fake_dataset(), .2, 10000)
        for name, part in zip(("train.npz", "test.npz"), expected):
            actual = sweep.load_npz(directory / name)
            for key, values in part.items():
                np.testing.assert_array_equal(actual[key], values)
        self.assertFalse((directory / "dataset.npz").exists())
        self.assertFalse(set(metadata["split"]["train"]["episode_ids"]) & set(metadata["split"]["test"]["episode_ids"]))
        self.assertEqual(metadata["split"]["train"]["transitions"], 32)
        self.assertEqual(metadata["split"]["test"]["transitions"], 8)
        with patch.object(sweep, "data_worker", side_effect=AssertionError("must reuse")):
            again, _ = sweep.prepare_dataset(self.args, self.spec, PROOF)
        self.assertEqual(directory, again)

    def test_corrupted_cache_rejected_never_overwritten(self):
        directory, _ = self.prepare()
        bad = directory / "train.npz"
        bad.write_bytes(b"intentional corrupt fixture")
        with patch.object(sweep, "data_worker", side_effect=AssertionError("must not recollect")), \
             self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            sweep.prepare_dataset(self.args, self.spec, PROOF)
        self.assertEqual(bad.read_bytes(), b"intentional corrupt fixture")

    def test_failed_collection_not_published_and_staging_removed(self):
        def fail(*args):
            raise RuntimeError("fixture failure")

        with patch.object(sweep, "data_worker", side_effect=fail), redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "fixture failure"):
            sweep.prepare_dataset(self.args, self.spec, PROOF)
        self.assertTrue(all(path.name.endswith(".lock") for path in (self.args.storage_root / "datasets").iterdir()))

    def test_concurrent_dataset_requests_collect_once(self):
        barrier = Barrier(2)

        def prepare():
            barrier.wait(timeout=10)
            return sweep.prepare_dataset(self.args, self.spec, PROOF)[0]

        with patch.object(sweep, "data_worker", side_effect=fake_collection) as collect, \
             redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prepare) for _ in range(2)]
            directories = [future.result(timeout=10) for future in futures]
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(directories[0], directories[1])

    def test_concurrent_run_requests_train_once(self):
        directory, metadata = self.prepare()
        barrier = Barrier(2)

        def train():
            barrier.wait(timeout=10)
            return sweep.run_one(self.args, "mobile", directory, metadata, {})

        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training) as trainer, \
             redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train) for _ in range(2)]
            directories = [future.result(timeout=10) for future in futures]
        self.assertEqual(trainer.call_count, 1)
        self.assertEqual(directories[0], directories[1])

    def test_collection_provenance_mismatch_rejected_before_split(self):
        def wrong(*args):
            fake_collection(*args)
            path = args[3] / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["environment"] = {}
            sweep.write_json(path, metadata)

        with patch.object(sweep, "data_worker", side_effect=wrong), redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "provenance changed"):
            sweep.prepare_dataset(self.args, self.spec, PROOF)

    def test_training_only_receives_training_split_and_reuses_exact_result(self):
        directory, metadata = self.prepare()
        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training) as train, \
             patch("walker_composition_eval.evaluate_run", side_effect=AssertionError("no --eval")), \
             redirect_stdout(io.StringIO()):
            run = sweep.run_one(self.args, "mobile", directory, metadata, {})
            before = {path.name: path.read_bytes() for path in run.rglob("*") if path.is_file()}
            repeated = sweep.run_one(self.args, "mobile", directory, metadata, {})
        self.assertEqual(train.call_count, 1)
        self.assertEqual(run, repeated)
        self.assertIs(train.call_args.args[0], self.args)
        train_data = train.call_args.args[3]
        self.assertEqual(array_hashes(train_data), metadata["split"]["train"]["array_sha256"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in run.rglob("*") if path.is_file()})
        manifest = json.loads((run / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "trained")
        self.assertEqual(len(manifest["checkpoints"]), 1)
        self.assertTrue(manifest["checkpoints"][0]["final"])

    def test_checkpoint_schedule_and_eval_are_only_requested(self):
        directory, metadata = self.prepare()
        self.args.checkpoint_eval_episodes = 2
        self.args.eval = True
        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training), \
             patch("walker_composition_eval.evaluate_run") as evaluate, redirect_stdout(io.StringIO()):
            run = sweep.run_one(self.args, "mobile", directory, metadata, {})
        manifest = json.loads((run / "run_manifest.json").read_text())
        self.assertEqual([record["epoch"] for record in manifest["checkpoints"]], list(range(1, 11)))
        self.assertEqual(manifest["status"], "complete")
        evaluate.assert_called_once_with(run, checkpoint_eval_episodes=2, final_eval_episodes=20,
                                         reference_root=self.args.reference_root, reuse_eval=False)

    def test_failed_training_preserved_not_replaced(self):
        directory, metadata = self.prepare()
        with patch.object(sweep.walker_sweep, "train", side_effect=RuntimeError("bad training")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "bad training"):
            sweep.run_one(self.args, "mobile", directory, metadata, {})
        manifests = list((self.args.storage_root / "runs").rglob("run_manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")
        original = manifests[0].read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "not a reusable"):
            sweep.run_one(self.args, "mobile", directory, metadata, {})
        self.assertEqual(manifests[0].read_bytes(), original)

    def test_eval_failure_keeps_completed_training_reusable(self):
        directory, metadata = self.prepare()
        self.args.eval = True
        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training), \
             patch("walker_composition_eval.evaluate_run", side_effect=RuntimeError("eval failed")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "eval failed"):
            sweep.run_one(self.args, "mobile", directory, metadata, {})
        manifest = json.loads(next((self.args.storage_root / "runs").rglob("run_manifest.json")).read_text())
        self.assertEqual(manifest["status"], "trained")

    def test_checkpoint_corruption_detected_before_reuse(self):
        directory, metadata = self.prepare()
        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training), redirect_stdout(io.StringIO()):
            run = sweep.run_one(self.args, "mobile", directory, metadata, {})
        (run / "model/policy.pth").write_bytes(b"changed checkpoint")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            sweep.run_one(self.args, "mobile", directory, metadata, {})

    def test_missing_checkpoints_are_not_silently_reconstructed_or_retrained(self):
        directory, metadata = self.prepare()
        with patch.object(sweep.walker_sweep, "train", side_effect=fake_training), redirect_stdout(io.StringIO()):
            run = sweep.run_one(self.args, "mobile", directory, metadata, {})
        before = (run / "run_manifest.json").read_bytes()
        self.args.checkpoint_eval_episodes = 2
        with patch.object(sweep.walker_sweep, "train", side_effect=AssertionError("must not retrain")), \
             self.assertRaisesRegex(RuntimeError, "earlier checkpoints cannot be reconstructed"):
            sweep.run_one(self.args, "mobile", directory, metadata, {})
        self.assertEqual((run / "run_manifest.json").read_bytes(), before)

    def test_preflight_happens_before_any_dataset_or_training(self):
        with patch.object(sweep.walker_sweep, "learner_provenance", return_value={"source_sha256": {}}), \
             patch.object(sweep, "data_worker", side_effect=RuntimeError("preflight failed")) as worker, \
             patch.object(sweep, "prepare_dataset", side_effect=AssertionError("collection started")), \
             self.assertRaisesRegex(RuntimeError, "preflight failed"):
            sweep.main(["--device", "cpu", "--storage-root", self.temporary.name])
        self.assertEqual(worker.call_args.args[1], "preflight")
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

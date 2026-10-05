"""Temporary-only tests for the independent model-free composition launcher."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

import walker_composition_mf as mf
from test_walker_composition import PROOF, fake_collection, fake_training


class ArgumentsTests(unittest.TestCase):
    def test_same_dataset_grid_and_model_free_only_configuration(self):
        argv = ["--seed", "10000", "10100", "10200", "10300"]
        args = mf.parse_args(argv)
        old = mf.composition.parse_args(argv)
        self.assertEqual(mf.composition.dataset_grid(args), mf.composition.dataset_grid(old))
        self.assertEqual(len(mf.composition.dataset_grid(args)), 36)
        self.assertEqual(args.algos, ["iql", "td3bc"])
        for algo in args.algos:
            config = mf.training_config(args, algo)
            self.assertEqual(config["implementation"], mf.IMPLEMENTATION)
            self.assertFalse(set(config) & {"real_ratio", "rollout_length", "penalty_coef", "dynamics_max_epochs"})
            other = "td3bc_" if algo == "iql" else "iql_"
            self.assertFalse(any(key.startswith(other) for key in config))

    def test_nontraining_and_other_algorithm_arguments_do_not_change_config(self):
        original = mf.parse_args([])
        changed = mf.parse_args(["--seed", "12", "--quiet", "--eval", "--reuse-eval", "--device", "cpu",
                                 "--checkpoint-eval-episodes", "10", "--final-eval-episodes", "100"])
        for algo in original.algos:
            self.assertEqual(mf.training_config(original, algo), mf.training_config(changed, algo))
        changed = mf.parse_args(["--td3bc-alpha", ".25"])
        self.assertEqual(mf.training_config(original, "iql"), mf.training_config(changed, "iql"))
        self.assertNotEqual(mf.training_config(original, "td3bc"), mf.training_config(changed, "td3bc"))
        changed = mf.parse_args(["--iql-expectile", ".8"])
        self.assertEqual(mf.training_config(original, "td3bc"), mf.training_config(changed, "td3bc"))
        self.assertNotEqual(mf.training_config(original, "iql"), mf.training_config(changed, "iql"))

    def test_invalid_inputs_and_model_based_flags_rejected(self):
        cases = [["--algos", "mobile"], ["--real-ratio", ".5"], ["--rollout-length", "1"],
                 ["--epoch", "0"], ["--step-per-epoch", "0"], ["--batch-size", "-1"],
                 ["--num-samples", "0"], ["--num-samples", "1000000"],
                 ["--composition", ".5", ".6"], ["--composition", "nan", ".5"],
                 ["--test-fraction", "nan"], ["--test-fraction", "1"],
                 ["--iql-expectile", "nan"], ["--iql-expectile", "1"],
                 ["--iql-temperature", "inf"], ["--iql-learning-rate", "0"],
                 ["--td3bc-alpha", "nan"], ["--td3bc-learning-rate", "-1"],
                 ["--iql-hidden-dims", "0"], ["--td3bc-hidden-dims", "-1"],
                 ["--seed", "-1"], ["--checkpoint-eval-episodes", "-1"], ["--final-eval-episodes", "-1"]]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mf.parse_args(argv)
        mf.parse_args(["--settings", "noise0.5", "--num-samples", "1000000"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            mf.training_config(mf.parse_args([]), "mobile")

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "absent"
            with patch.object(mf.composition, "data_worker", side_effect=AssertionError("worker")), \
                 patch.object(mf, "train", side_effect=AssertionError("training")), \
                 redirect_stdout(io.StringIO()) as output:
                mf.main(["--storage-root", str(root), "--dry-run", "--seed", "10000", "10100", "10200", "10300"])
            report = json.loads(output.getvalue())
            self.assertFalse(root.exists())
            self.assertEqual(report["distinct_datasets"], 36)
            self.assertEqual(report["distinct_training_runs"], 72)
            self.assertEqual(report["requested_points"], 80)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = mf.parse_args(["--storage-root", self.temporary.name, "--device", "cpu",
                                  "--settings", "noise0.5", "--composition", ".5", ".5",
                                  "--epoch", "10", "--step-per-epoch", "2", "--quiet"])
        spec = mf.composition.dataset_grid(self.args)[0]
        with patch.object(mf.composition, "data_worker", side_effect=fake_collection), redirect_stdout(io.StringIO()):
            self.directory, self.metadata = mf.composition.prepare_dataset(self.args, spec, PROOF)

    def run_one(self, algo="iql"):
        return mf.run_one(self.args, algo, self.directory, self.metadata, {})

    def test_training_only_reads_train_split_and_reuses_checkpoint(self):
        with patch.object(mf, "train", side_effect=fake_training) as train, redirect_stdout(io.StringIO()):
            run = self.run_one()
            before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            self.assertEqual(run, self.run_one())
        self.assertEqual(train.call_count, 1)
        self.assertEqual(run.parent.name, "model_free_runs")
        self.assertFalse((self.args.storage_root / "runs").exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()})
        self.assertEqual(mf.composition.array_hashes(train.call_args.args[3]),
                         self.metadata["split"]["train"]["array_sha256"])
        manifest = json.loads((run / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "trained")
        self.assertEqual([item["epoch"] for item in manifest["checkpoints"]], [10])

    def test_algorithms_have_separate_runs_same_dataset(self):
        with patch.object(mf, "train", side_effect=fake_training), redirect_stdout(io.StringIO()):
            runs = [self.run_one(algo) for algo in mf.ALGORITHMS]
        self.assertNotEqual(*runs)
        for run in runs:
            manifest = json.loads((run / "run_manifest.json").read_text())
            self.assertEqual(manifest["dataset_id"], self.metadata["dataset_id"])

    def test_concurrent_same_run_trains_once(self):
        barrier = Barrier(2)

        def run():
            barrier.wait(timeout=10)
            return self.run_one()

        with patch.object(mf, "train", side_effect=fake_training) as train, \
             redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run) for _ in range(2)]
            runs = [future.result(timeout=10) for future in futures]
        self.assertEqual(train.call_count, 1)
        self.assertEqual(*runs)

    def test_failure_kept_and_never_automatically_retrained(self):
        with patch.object(mf, "train", side_effect=RuntimeError("fixture failure")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.run_one()
        path = next((self.args.storage_root / "model_free_runs").glob("*/run_manifest.json"))
        original = path.read_bytes()
        self.assertEqual(json.loads(original)["status"], "failed")
        with patch.object(mf, "train", side_effect=AssertionError("retrained")), \
             self.assertRaisesRegex(RuntimeError, "not a reusable"):
            self.run_one()
        self.assertEqual(path.read_bytes(), original)

    def test_missing_diagnostics_and_corrupt_checkpoint_fail_without_retraining(self):
        with patch.object(mf, "train", side_effect=fake_training), redirect_stdout(io.StringIO()):
            run = self.run_one()
        self.args.checkpoint_eval_episodes = 1
        with patch.object(mf, "train", side_effect=AssertionError("retrained")), \
             self.assertRaisesRegex(RuntimeError, "checkpoint-eval-episodes 0"):
            self.run_one()
        self.args.checkpoint_eval_episodes = 0
        (run / "model/policy.pth").write_bytes(b"intentional corrupt fixture")
        with patch.object(mf, "train", side_effect=AssertionError("retrained")), \
             self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.run_one()

    def test_eval_dispatch_and_eval_failure_preserve_training(self):
        self.args.eval = True
        self.args.checkpoint_eval_episodes = 2
        with patch.object(mf, "train", side_effect=fake_training), \
             patch("walker_composition_mf_eval.evaluate_run", side_effect=RuntimeError("eval failed")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "eval failed"):
            self.run_one()
        path = next((self.args.storage_root / "model_free_runs").glob("*/run_manifest.json"))
        self.assertEqual(json.loads(path.read_text())["status"], "trained")
        with patch.object(mf, "train", side_effect=AssertionError("retrained")), \
             patch("walker_composition_mf_eval.evaluate_run") as evaluate, redirect_stdout(io.StringIO()):
            run = self.run_one()
        evaluate.assert_called_once_with(run, checkpoint_eval_episodes=2, final_eval_episodes=20,
                                         reference_root=self.args.reference_root, reuse_eval=False)
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual([c["epoch"] for c in manifest["checkpoints"]], list(range(1, 11)))

    def test_preflight_failure_prevents_collection_and_training(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(mf.walker_sweep, "learner_provenance", return_value={"source_sha256": {}}), \
             patch.object(mf.composition, "data_worker", side_effect=RuntimeError("preflight")), \
             patch.object(mf.composition, "prepare_dataset", side_effect=AssertionError("collection")), \
             patch.object(mf, "train", side_effect=AssertionError("training")), \
             self.assertRaisesRegex(RuntimeError, "preflight"):
            mf.main(["--device", "cpu", "--storage-root", temporary])


if __name__ == "__main__":
    unittest.main()

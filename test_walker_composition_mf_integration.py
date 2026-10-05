"""Opt-in real-reference collection, CPU training, evaluation, reuse, and plots.

Run with WALKER_COMPOSITION_INTEGRATION=1 and one CPU thread. All newly created
data, policies, evaluations, and plots live in an automatically removed tempdir.
The MOPO/MOBILE comparison fixtures are random valid actors, not trained results.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import walker_composition as composition
import walker_composition_eval as existing_eval
import walker_composition_mf as model_free
import walker_composition_mf_eval as model_free_eval
import walker_composition_plot as plotting
import walker_sweep


ROOT = Path(__file__).resolve().parent
PROTECTED_FILES = (
    "walker_composition.py", "walker_composition_data.py", "walker_composition_eval.py",
    "walker_composition_plot.py", "walker_sweep.py", "walker_reference.py", "policies.py",
    "sweep.py", "eval.py", "plot.py", "rollout.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(path):
    return {str(item.relative_to(path)): digest(item)
            for item in sorted(Path(path).rglob("*")) if item.is_file()}


def read_json(path):
    return json.loads(Path(path).read_text())


@unittest.skipUnless(os.environ.get("WALKER_COMPOSITION_INTEGRATION") == "1",
                     "opt-in actual reference collection/evaluation integration")
class ModelFreeReferenceIntegrationTests(unittest.TestCase):
    def test_real_compositions_training_evaluation_reuse_and_mixed_algorithm_plots(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        original_sources = {name: digest(ROOT / name) for name in PROTECTED_FILES}
        with tempfile.TemporaryDirectory(prefix="walker-mf-integration-") as temporary, \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                     "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                     "PYTHONDONTWRITEBYTECODE": "1"}):
            storage = Path(temporary)
            argv = [
                "--storage-root", str(storage), "--device", "cpu",
                "--settings", "noise0.5", "clean-medium-v2", "--algos", "iql", "td3bc",
                "--composition", "1", "0", "--composition", ".5", ".5",
                "--composition", "0", "1", "--seed", "73", "--num-samples", "2000",
                "--test-fraction", ".2", "--epoch", "2", "--step-per-epoch", "2",
                "--batch-size", "8", "--iql-hidden-dims", "16", "16",
                "--td3bc-hidden-dims", "16", "16", "--checkpoint-eval-episodes", "1",
                "--final-eval-episodes", "1", "--quiet", "--eval", "--reuse-eval",
            ]
            args = model_free.parse_args(argv)
            real_train = model_free.train
            training_calls = []

            def checked_train(*positional, **keywords):
                before = tree_hashes(storage / "datasets")
                result = real_train(*positional, **keywords)
                self.assertEqual(tree_hashes(storage / "datasets"), before)
                training_calls.append(positional[1])
                return result

            with patch.object(model_free, "train", side_effect=checked_train):
                model_free.main(argv)

            self.assertEqual(training_calls.count("iql"), 5)
            self.assertEqual(training_calls.count("td3bc"), 5)
            datasets = sorted((storage / "datasets").glob("*/metadata.json"))
            manifests = sorted((storage / "model_free_runs").glob("*/run_manifest.json"))
            reports = sorted((storage / "evals").glob("*/*/evaluation.json"))
            self.assertEqual((len(datasets), len(manifests), len(reports)), (5, 10, 10))
            self.assertFalse((storage / "runs").exists())
            immutable_data = tree_hashes(storage / "datasets")

            for path in datasets:
                metadata = read_json(path)
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["dataset_spec"]["seed"], 73)
                self.assertEqual(metadata["dataset_spec"]["num_samples"], 2000)
                self.assertEqual(metadata["environment"]["foot_friction"], [.9, 1.9])
                train = composition.load_npz(path.parent / "train.npz")
                test = composition.load_npz(path.parent / "test.npz")
                train_episodes = set(train["episode_ids"].tolist())
                test_episodes = set(test["episode_ids"].tolist())
                self.assertTrue(train_episodes)
                self.assertTrue(test_episodes)
                self.assertFalse(train_episodes & test_episodes)
                for name, data in (("train", train), ("test", test)):
                    self.assertEqual(composition.array_hashes(data), metadata["split"][name]["array_sha256"])
                    self.assertEqual(digest(path.parent / f"{name}.npz"), metadata["file_sha256"][f"{name}.npz"])

            for path in manifests:
                manifest = read_json(path)
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["runtime"]["device"], "cpu")
                self.assertEqual([item["epoch"] for item in manifest["checkpoints"]], [1, 2])
                self.assertEqual([item["final"] for item in manifest["checkpoints"]], [False, True])
                if manifest["algorithm"] == "td3bc":
                    data = composition.load_npz(manifest["train_dataset_path"])
                    for checkpoint in manifest["checkpoints"]:
                        state = torch.load(checkpoint["policy_path"], map_location="cpu", weights_only=True)
                        np.testing.assert_array_equal(state["observation_mean"].numpy(),
                                                      data["observations"].mean(0, keepdims=True))
                        np.testing.assert_array_equal(state["observation_std"].numpy(),
                                                      data["observations"].std(0, keepdims=True) + 1e-3)

            self.check_reports(reports)
            self.render_and_check(storage, ("iql", "td3bc"), [0, .5, 1], "model-free")

            # Completed fits/evaluations are reusable with all training and worker
            # launches forbidden, without invoking the legitimate main preflight.
            before_reuse = tree_hashes(storage)
            with patch.object(model_free, "train", side_effect=AssertionError("unexpected retraining")), \
                 patch.object(model_free_eval.subprocess, "run", side_effect=AssertionError("unexpected worker")):
                for path in manifests:
                    manifest = read_json(path)
                    directory = Path(manifest["dataset_dir"])
                    model_free.run_one(args, manifest["algorithm"], directory,
                                       read_json(directory / "metadata.json"), {"test": "reuse"})
            self.assertEqual(tree_hashes(storage), before_reuse)

            # Use genuine current-format MB checkpoints and the unchanged legacy
            # evaluator to prove that common plot/evaluation protocol is compatible.
            expert_before = tree_hashes(storage / "evals" / "experts")
            self.assertEqual(len(list((storage / "evals" / "experts").glob("*.json"))), 1)
            mb_args = composition.parse_args([
                "--storage-root", str(storage), "--device", "cpu", "--epoch", "2",
                "--step-per-epoch", "2", "--seed", "73", "--num-samples", "2000",
                "--test-fraction", ".2", "--composition", ".5", ".5",
                "--checkpoint-eval-episodes", "1", "--final-eval-episodes", "1",
                "--quiet", "--eval", "--reuse-eval",
            ])
            with patch.object(walker_sweep, "train", side_effect=self.write_model_based_fixture):
                for path in datasets:
                    metadata = read_json(path)
                    if metadata["dataset_spec"]["other_fraction"] == .5:
                        for algorithm in ("mopo", "mobile"):
                            composition.run_one(mb_args, algorithm, path.parent, metadata,
                                                {"test": "untrained architecture fixture"})
            self.assertEqual(len(list((storage / "runs").glob("*/run_manifest.json"))), 4)
            all_reports = sorted((storage / "evals").glob("*/*/evaluation.json"))
            self.assertEqual(len(all_reports), 14)
            self.check_reports(all_reports)
            self.assertEqual(tree_hashes(storage / "evals" / "experts"), expert_before)
            self.assertEqual(tree_hashes(storage / "datasets"), immutable_data)
            self.render_and_check(storage, ("td3bc", "iql", "mopo", "mobile"), [.5], "all-four")
            self.assertEqual(len(list((storage / "plots").rglob("*.png"))), 12)
            self.assertEqual({name: digest(ROOT / name) for name in PROTECTED_FILES}, original_sources)

    def check_reports(self, paths):
        expert_reports = []
        for path in paths:
            report = read_json(path)
            existing_eval.validate_evaluation(report, Path(report["run_manifest_path"]),
                                              report["evaluation_config"], report["config_id"])
            records = report["records"]
            self.assertEqual(len(records), 2)
            self.assertEqual([item["reset_seeds"] for item in records], [[73], [1000073]])
            self.assertEqual([item["episodes"] for item in records], [1, 1])
            for record in records:
                for name in ("returns", "normalized_scores", "performance", "lengths"):
                    self.assertEqual(len(record[name]), 1)
                    self.assertTrue(np.isfinite(record[name]).all())
                self.assertGreaterEqual(record["lengths"][0], 1)
                self.assertLessEqual(record["lengths"][0], 1000)
            expert_reports.append(report["expert"])
        self.assertTrue(all(report == expert_reports[0] for report in expert_reports))

    def render_and_check(self, storage, algorithms, fractions, suffix):
        for setting in ("noise0.5", "clean-medium-v2"):
            cohort = {
                "version": 2, "setting": setting, "seeds": [73],
                "ablation": {"name": "other_fraction", "values": fractions},
                "match": {"dataset_spec.num_samples": 2000, "dataset_spec.test_fraction": .2},
                "series": [{"algo": algorithm} for algorithm in algorithms],
            }
            for metric in ("performance", "raw_return", "normalized_return"):
                result = plotting.summarize(storage / "evals", cohort, metric)
                self.assertEqual([item["label"] for item in result["series"]],
                                 [algorithm.upper() for algorithm in algorithms])
                self.assertTrue(all(len(item["points"]) == len(fractions) for item in result["series"]))
                self.assertTrue(all(len(point["history"]) == 2
                                    for item in result["series"] for point in item["points"]))
                if metric == "performance":
                    output = storage / "plots" / f"{setting}-{suffix}"
                    plotting.render(result, output)
                    self.assertEqual(len(list(output.glob("*.png"))), len(fractions) + 1)
                    self.assertEqual(read_json(output / "plot_summary.json"), result)

    @staticmethod
    def write_model_based_fixture(args, algorithm, seed, data, directory):
        from policies import build_prob_actor

        torch.manual_seed(seed + (1 if algorithm == "mopo" else 2))
        actor, _ = build_prob_actor(17, 6, 1., [256, 256], "cpu", 1e-4)
        state = {"actor." + key: value for key, value in actor.state_dict().items()}
        for epoch in walker_sweep.checkpoint_epochs(args) + [args.epoch]:
            path = directory / ("model/policy.pth" if epoch == args.epoch
                                else f"checkpoint/step_{epoch * args.step_per_epoch}/policy.pth")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)


if __name__ == "__main__":
    unittest.main()

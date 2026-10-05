"""Additive model-free support must preserve old caches and plotting semantics."""

from contextlib import redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plot import bootstrap_mean
from test_walker_composition import PROOF, fake_collection, fake_training
import test_walker_composition_plot as plot_fixtures
import walker_composition as composition
import walker_composition_plot as plotting


class ExistingRunCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.options = ["--storage-root", self.temporary.name, "--device", "cpu",
                        "--settings", "noise0.5", "--composition", ".5", ".5",
                        "--epoch", "10", "--step-per-epoch", "2", "--quiet"]
        self.mb_args = composition.parse_args(self.options)
        self.spec = composition.dataset_grid(self.mb_args)[0]
        with patch.object(composition, "data_worker", side_effect=fake_collection), \
                redirect_stdout(io.StringIO()):
            self.directory, self.metadata = composition.prepare_dataset(self.mb_args, self.spec, PROOF)
        self.mf = importlib.import_module("walker_composition_mf")
        self.mf_args = self.mf.parse_args(self.options)

    @staticmethod
    def snapshot(root):
        return {str(path.relative_to(root)): composition.sha256(path)
                for path in root.rglob("*") if path.is_file()}

    def test_model_free_resolves_identical_dataset_and_split_without_collection(self):
        self.assertEqual(self.mf.composition.dataset_grid(self.mf_args), [self.spec])
        before = self.snapshot(self.directory)
        with patch.object(composition, "data_worker", side_effect=AssertionError("must reuse")):
            directory, metadata = self.mf.composition.prepare_dataset(self.mf_args, self.spec, PROOF)
        self.assertEqual(directory, self.directory)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_model_free_training_does_not_change_existing_mb_runs_or_identities(self):
        old_functions = {name: getattr(composition, name)
                         for name in ("parse_args", "training_config", "run_one", "prepare_dataset")}
        old_configs = {algo: composition.training_config(self.mb_args, algo)
                       for algo in ("mopo", "mobile")}
        with patch.object(composition.walker_sweep, "train", side_effect=fake_training), \
                redirect_stdout(io.StringIO()):
            old_runs = {algo: composition.run_one(self.mb_args, algo, self.directory, self.metadata, {})
                        for algo in old_configs}
        old_run_snapshot = self.snapshot(self.mb_args.storage_root / "runs")
        old_data_snapshot = self.snapshot(self.mb_args.storage_root / "datasets")
        for algo in ("td3bc", "iql"):
            with self.subTest(algo=algo), \
                    patch.object(composition.walker_sweep, "train", side_effect=AssertionError("MB training")), \
                    patch.object(self.mf, "train", side_effect=fake_training) as train, \
                    redirect_stdout(io.StringIO()):
                run = self.mf.run_one(self.mf_args, algo, self.directory, self.metadata, {})
                again = self.mf.run_one(self.mf_args, algo, self.directory, self.metadata, {})
            self.assertEqual(run, again)
            self.assertEqual(run.parent, self.mf_args.storage_root / "model_free_runs")
            self.assertEqual(train.call_count, 1)
            self.assertEqual(composition.array_hashes(train.call_args.args[3]),
                             self.metadata["split"]["train"]["array_sha256"])
        for algo, run in old_runs.items():
            with patch.object(composition.walker_sweep, "train", side_effect=AssertionError("must reuse")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(composition.run_one(self.mb_args, algo, self.directory, self.metadata, {}), run)
            self.assertEqual(composition.training_config(self.mb_args, algo), old_configs[algo])
        self.assertEqual(self.snapshot(self.mb_args.storage_root / "runs"), old_run_snapshot)
        self.assertEqual(self.snapshot(self.mb_args.storage_root / "datasets"), old_data_snapshot)
        self.assertTrue(all(getattr(composition, name) is function for name, function in old_functions.items()))


class MixedAlgorithmPlotCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = plot_fixtures.CompositionPlotTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.cohort = copy.deepcopy(self.fixture.cohort)
        self.original = copy.deepcopy(plotting.summarize(self.root, self.cohort))

    def add_series(self, algo, epoch=3000, real_ratio=None):
        config = {"epoch": epoch}
        if real_ratio is not None:
            config["real_ratio"] = real_ratio
        if algo in ("iql", "td3bc"):
            mf = importlib.import_module("walker_composition_mf")
            config = mf.training_config(mf.parse_args(["--epoch", str(epoch)]), algo)
        filters = {"training_config.real_ratio": real_ratio} if real_ratio is not None else {}
        self.cohort["series"].append({"algo": algo, "match": filters})
        for fraction in self.cohort["ablation"]["values"]:
            for seed in self.cohort["seeds"]:
                path = self.fixture.write_run(fraction, seed, suffix=f"-{algo}-{real_ratio}")
                self.fixture.edit(path / "run_manifest.json",
                                  lambda obj: obj.update(algorithm=algo, training_config=config))

                def update_epochs(obj):
                    for records in (obj["records"], obj["evaluation_config"]["checkpoints"]):
                        for record in records:
                            record["epoch"] = epoch if record["final"] else epoch // 10
                            record["step"] = record["epoch"] * 1000

                self.fixture.edit(path / "evaluation.json", update_epochs)

    def test_existing_mobile_summary_is_identical_with_unselected_new_algorithms_present(self):
        old_cohort = copy.deepcopy(self.cohort)
        for algo in ("iql", "td3bc", "mopo"):
            self.add_series(algo)
        self.assertEqual(plotting.summarize(self.root, old_cohort), self.original)

    def test_four_algorithms_and_two_mobile_ratios_use_unchanged_statistics_and_render(self):
        self.add_series("td3bc", epoch=1000)
        self.add_series("iql", epoch=2000)
        self.add_series("mopo")
        self.add_series("mobile", real_ratio=0.5)
        expected = bootstrap_mean([np.array([1, 3, 1, 3]), np.array([5, 7, 9, 11])])
        result = plotting.summarize(self.root, self.cohort)
        self.assertEqual([line["label"] for line in result["series"]],
                         ["MOBILE (real ratio=0.05)", "TD3BC", "IQL", "MOPO", "MOBILE (real ratio=0.50)"])
        self.assertIs(plotting.bootstrap_mean, bootstrap_mean)
        self.assertEqual(result["bootstrap_replicates"], 10000)
        self.assertEqual(result["bootstrap_percentiles"], [10., 90.])
        for line in result["series"]:
            for point in line["points"]:
                self.assertEqual((point["mean"], point["low"], point["high"]), expected)
                self.assertEqual(point["seeds"], [10000, 10100])
                self.assertAlmostEqual(point["actual_fraction"], 0.55 if point["requested_fraction"] else 0)
        self.assertEqual(result["series"][1]["points"][0]["history"][-1]["epoch"], 1000)
        self.assertEqual(result["series"][2]["points"][0]["history"][-1]["epoch"], 2000)
        output = self.root / "plots"
        plotting.render(result, output)
        self.assertEqual(len(list(output.glob("*.png"))), 3)
        self.assertEqual(json.loads((output / "plot_summary.json").read_text()), result)

    def test_mobile_names_use_selected_manifests_when_ratios_are_not_filters(self):
        self.cohort["series"][0]["match"] = {"training_config.epoch": 3000}
        for path in self.fixture.paths.values():
            self.fixture.edit(path / "run_manifest.json", lambda obj: obj["training_config"].update(real_ratio=0))
        self.add_series("mobile", epoch=4000, real_ratio=.5)
        self.cohort["series"][-1]["match"] = {"training_config.epoch": 4000}

        mobile_only = plotting.summarize(self.root, self.cohort)
        self.assertEqual([line["label"] for line in mobile_only["series"]],
                         ["real ratio=0.00 (epoch=3000)", "real ratio=0.50 (epoch=4000)"])
        self.add_series("iql")
        mixed = plotting.summarize(self.root, self.cohort)
        self.assertEqual([line["label"] for line in mixed["series"]],
                         ["MB-MOBILE (epoch=3000)", "Hybrid-MOBILE (epoch=4000)", "IQL"])
        self.assertEqual(mixed["series"][0]["points"], self.original["series"][0]["points"])
        for original_line, renamed_line in zip(mobile_only["series"], mixed["series"]):
            self.assertEqual(original_line["points"], renamed_line["points"])

    def test_missing_manifest_ratio_does_not_infer_a_mobile_variant(self):
        self.cohort["series"][0]["match"] = {}
        for path in self.fixture.paths.values():
            self.fixture.edit(path / "run_manifest.json", lambda obj: obj["training_config"].pop("real_ratio"))
        self.add_series("iql")
        result = plotting.summarize(self.root, self.cohort)
        self.assertEqual([line["label"] for line in result["series"]], ["MOBILE", "IQL"])
        self.assertEqual(result["series"][0]["points"], self.original["series"][0]["points"])

    def test_all_metrics_work_for_model_free_and_model_based_lines(self):
        self.add_series("iql")
        self.add_series("td3bc")
        self.add_series("mopo")
        for metric, expected in (("performance", 5), ("raw_return", 50), ("normalized_return", 10)):
            result = plotting.summarize(self.root, self.cohort, metric)
            self.assertTrue(all(point["mean"] == expected for line in result["series"] for point in line["points"]))

    def test_model_free_evaluation_protocol_mismatch_is_rejected(self):
        self.add_series("iql")
        for path in self.root.glob("*-iql-None/evaluation.json"):
            self.fixture.edit(path, lambda obj: obj["evaluation_config"].update(metric_version=2))
        with self.assertRaisesRegex(ValueError, "Mixed evaluation configurations"):
            plotting.summarize(self.root, self.cohort)

    def test_new_cohort_filters_match_actual_launcher_defaults_and_mb_recipes(self):
        mf = importlib.import_module("walker_composition_mf")
        for name in ("clean_noisy", "clean_medium"):
            path = Path(__file__).parent / "walker_composition_cohorts" / f"all_algorithms_{name}.json"
            cohort = plotting.validate_cohort(plotting.read_json(path))
            self.assertEqual(cohort["seeds"], [10000, 10100, 10200, 10300])
            self.assertEqual(cohort["ablation"]["values"], [0, .25, .5, .75, 1])
            self.assertEqual(cohort["eval_match"], {"checkpoint_eval_episodes": 10, "final_eval_episodes": 20})
            self.assertEqual([line["algo"] for line in cohort["series"]], ["td3bc", "iql", "mopo", "mobile", "mobile"])
            self.assertEqual(
                [line["match"].get("training_config.real_ratio") for line in cohort["series"]],
                [None, None, 0.0, 0.05, 0.0],
            )
            self.assertEqual(plotting.series_labels(cohort), ["TD3BC", "IQL", "MOPO", "Hybrid-MOBILE", "MB-MOBILE"])
            for series in cohort["series"]:
                algo = series["algo"]
                if algo in ("td3bc", "iql"):
                    config = mf.training_config(mf.parse_args([]), algo)
                else:
                    if algo == "mopo":
                        options = ["--real-ratio", "0", "--rollout-length", "1", "--rollout-batch-size", "250000",
                                   "--penalty-coef", "2.5", "--dynamics-max-epochs", "0"]
                    else:
                        options = ["--real-ratio", str(series["match"]["training_config.real_ratio"])]
                    config = composition.training_config(composition.parse_args(options), algo)
                candidate = {"dataset_spec": {"num_samples": 999995, "test_fraction": .2}, "training_config": config}
                with self.subTest(cohort=name, algorithm=algo):
                    self.assertTrue(plotting.matches(candidate, cohort["match"]))
                    self.assertTrue(plotting.matches(candidate, series["match"]))


if __name__ == "__main__":
    unittest.main()

"""Cache compatibility checks for per-source transition collection quotas."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sweep


class TransitionQuotaCacheTests(unittest.TestCase):
    def generated_schema(self, **overrides):
        return {
            "version": sweep.DATASET_SCHEMA_VERSION,
            "source": "generated",
            "env_name": "Reacher-v5",
            "expert_path": "/tmp/Reacher-v5.zip",
            "max_timesteps": 1000,
            "num_samples": 1000,
            "noise_scale": 0.5,
            "prop_clean_expert": 0.5,
            "prop_noisy_expert": 0.5,
            "prop_random": 0.0,
            "prop_expert": 1.0,
            "deterministic": True,
            "seed": 10000,
            "test_fraction": 0.2,
            **overrides,
        }

    def metadata(self, clean=500, noisy=500, random=0):
        return {
            "num_transitions": clean + noisy + random,
            "num_clean_expert_transitions": clean,
            "num_noisy_expert_transitions": noisy,
            "num_random_transitions": random,
        }

    def write_cache(self, path, schema, metadata):
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({
            **metadata, "dataset_schema": schema, "test_fraction": schema["test_fraction"],
        }), encoding="utf-8")
        (path / "train.npz").touch()
        (path / "test.npz").touch()

    def write_run(self, path, schema, metadata_path):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text(json.dumps({
            "training_schema": {"dataset": schema},
            "dataset_metadata_path": str(metadata_path),
        }), encoding="utf-8")

    def test_legacy_metadata_satisfying_quotas_remains_eligible(self):
        schema = self.generated_schema()
        self.assertTrue(sweep.dataset_meets_transition_quotas(self.metadata(), schema))
        self.assertTrue(sweep.dataset_meets_transition_quotas(
            self.metadata(clean=550, noisy=530), schema,
        ))

    def test_total_surplus_does_not_compensate_for_source_shortfall(self):
        schema = self.generated_schema()
        for counts in ((499, 2000), (2000, 499)):
            with self.subTest(counts=counts):
                self.assertFalse(sweep.dataset_meets_transition_quotas(
                    self.metadata(clean=counts[0], noisy=counts[1]), schema,
                ))

    def test_positive_source_counts_are_required(self):
        schema = self.generated_schema()
        for key in ("num_clean_expert_transitions", "num_noisy_expert_transitions"):
            metadata = self.metadata()
            del metadata[key]
            with self.subTest(key=key):
                self.assertFalse(sweep.dataset_meets_transition_quotas(metadata, schema))

    def test_zero_share_source_may_be_absent_but_must_not_have_data(self):
        schema = self.generated_schema()
        metadata = self.metadata()
        del metadata["num_random_transitions"]
        self.assertTrue(sweep.dataset_meets_transition_quotas(metadata, schema))
        self.assertFalse(sweep.dataset_meets_transition_quotas(
            self.metadata(random=1), schema,
        ))

    def test_random_source_has_its_own_quota(self):
        schema = self.generated_schema(
            prop_clean_expert=0.25, prop_noisy_expert=0.25,
            prop_random=0.5, prop_expert=0.5,
        )
        self.assertTrue(sweep.dataset_meets_transition_quotas(
            self.metadata(clean=250, noisy=250, random=500), schema,
        ))
        self.assertFalse(sweep.dataset_meets_transition_quotas(
            self.metadata(clean=500, noisy=500, random=499), schema,
        ))

    def test_noninteger_quotas_round_up_independently(self):
        schema = self.generated_schema(num_samples=1001)
        self.assertFalse(sweep.dataset_meets_transition_quotas(
            self.metadata(clean=500, noisy=501), schema,
        ))
        self.assertTrue(sweep.dataset_meets_transition_quotas(
            self.metadata(clean=501, noisy=501), schema,
        ))

    def test_clean_minari_requires_both_source_quotas(self):
        schema = {"source": "clean-minari", "num_samples": 1000, "minari_fraction": 0.25}
        metadata = {"num_clean_expert_transitions": 750, "num_minari_transitions": 250}
        self.assertTrue(sweep.dataset_meets_transition_quotas(metadata, schema))
        for bad in (
            {"num_clean_expert_transitions": 749, "num_minari_transitions": 1000},
            {"num_clean_expert_transitions": 1000, "num_minari_transitions": 249},
            {"num_clean_expert_transitions": 1000},
            {"num_minari_transitions": 1000},
        ):
            with self.subTest(metadata=bad):
                self.assertFalse(sweep.dataset_meets_transition_quotas(bad, schema))

    def test_pure_minari_mixture_does_not_require_or_accept_clean_data(self):
        schema = {"source": "clean-minari", "num_samples": 1000, "minari_fraction": 1.0}
        self.assertTrue(sweep.dataset_meets_transition_quotas(
            {"num_minari_transitions": 1000}, schema,
        ))
        self.assertFalse(sweep.dataset_meets_transition_quotas(
            {"num_minari_transitions": 1000, "num_clean_expert_transitions": 1}, schema,
        ))

    def test_non_mixture_sources_are_unchanged(self):
        for source in ("minari", "robomimic", "test"):
            with self.subTest(source=source):
                self.assertTrue(sweep.dataset_meets_transition_quotas({}, {"source": source}))

    def test_cache_lookup_skips_newer_underquota_cache_and_reuses_legacy(self):
        schema = self.generated_schema()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self.write_cache(parent / "20260901", schema, self.metadata())
            self.write_cache(parent / "20260902", schema, self.metadata(clean=499, noisy=2000))
            dataset = {"marker": "legacy"}
            with patch("sweep.rollout.load_dataset", return_value=dataset) as load:
                found = sweep.find_cached_dataset(parent, schema)
            self.assertIsNotNone(found)
            self.assertIs(found[0], dataset)
            self.assertEqual(Path(found[1]["train_dataset_path"]).parent, parent / "20260901")
            self.assertEqual(load.call_count, 2)
            self.assertTrue(all(call.args[0].parent == parent / "20260901" for call in load.call_args_list))

    def test_cache_lookup_rejects_unknown_positive_source_counts(self):
        schema = self.generated_schema()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self.write_cache(parent / "20260901", schema, {})
            with patch("sweep.rollout.load_dataset") as load:
                self.assertIsNone(sweep.find_cached_dataset(parent, schema))
            load.assert_not_called()

    def test_trained_run_lookup_cannot_bypass_source_quotas(self):
        schema = self.generated_schema()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, metadata in (
                ("20260901", self.metadata()),
                ("20260902", self.metadata(clean=499, noisy=2000)),
            ):
                data = root / "datasets" / variant
                self.write_cache(data, schema, metadata)
                self.write_run(root / "runs" / variant, schema, data / "metadata.json")
            with patch("sweep.run_is_complete", return_value=True):
                found = sweep.find_trained_run(root / "runs", {"dataset": schema})
            self.assertEqual(found, root / "runs" / "20260901")

    def test_trained_run_missing_or_unreadable_metadata_is_not_reused(self):
        schema = self.generated_schema()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            self.write_run(root / "runs" / "20260901", schema, metadata_path)
            with patch("sweep.run_is_complete", return_value=True):
                self.assertIsNone(sweep.find_trained_run(root / "runs", {"dataset": schema}))
                metadata_path.write_text("not JSON", encoding="utf-8")
                self.assertIsNone(sweep.find_trained_run(root / "runs", {"dataset": schema}))

    def test_pure_minari_run_reuse_does_not_require_mixture_metadata(self):
        schema = {"source": "minari", "dataset_id": "mujoco/reacher/expert-v0"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root / "20260901", schema, root / "unused-metadata.json")
            with patch("sweep.run_is_complete", return_value=True):
                self.assertEqual(sweep.find_trained_run(root, {"dataset": schema}), root / "20260901")

    def test_clean_endpoint_lookup_excludes_underquota_data(self):
        schema = self.generated_schema(prop_clean_expert=1.0, prop_noisy_expert=0.0)
        args = SimpleNamespace(max_timesteps=1000, seed=10000, test_fraction=0.2, algos=["none"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_parent = root / "datasets" / "a-valid"
            self.write_cache(valid_parent / "20260901", schema, self.metadata(clean=1000, noisy=0))
            self.write_cache(root / "datasets" / "z-underquota" / "20260902", schema,
                             self.metadata(clean=999, noisy=0))
            found = sweep.find_generated_clean_dataset(
                root / "datasets", root / "trained", "Reacher-v5",
                Path(schema["expert_path"]), 1000, args,
            )
            self.assertEqual(found, (valid_parent, valid_parent.name, schema))


if __name__ == "__main__":
    unittest.main()

"""Collection parity and complete-trajectory tests; no expensive training."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import walker_composition_data as data
from walker_reference import convert_dataset


def spec(setting="noise0.5", clean=0.5, n=999995, seed=10000):
    return {"protocol": data.PROTOCOL, "setting": setting, "seed": seed, "num_samples": n,
            "clean_fraction": clean, "other_fraction": 1. - clean, "test_fraction": 0.2}


class Actor:
    def deterministic_action(self, observation):
        return np.tanh(observation[:6] + np.array([0., 1., -1., 5., -5., .5])).astype(np.float32)

    def predict(self, observation, deterministic=True):
        return self.deterministic_action(observation), None


class LegacyEnv:
    def __init__(self, horizon=None):
        self.horizon = horizon
        self.reset_seeds = []
        self.closed = False

    def seed(self, seed):
        self.reset_seeds.append(seed)

    def reset(self):
        self.observation = np.random.default_rng(self.reset_seeds[-1]).normal(size=17)
        self.length = self.horizon or 3 + self.reset_seeds[-1] % 4
        self.steps = 0
        return self.observation.copy()

    def step(self, action):
        self.steps += 1
        self.observation = self.observation + np.resize(action, 17) * .01 + .1
        done = self.steps == self.length
        info = {"TimeLimit.truncated": done and self.steps == 1000}
        return self.observation.copy(), float(action.sum()), done, info

    def close(self):
        self.closed = True


class ModernEnv:
    def __init__(self):
        self.legacy = LegacyEnv()
        self.action_space = type("Space", (), {"shape": (6,), "low": -np.ones(6), "high": np.ones(6)})()

    def reset(self, seed):
        self.legacy.seed(seed)
        return self.legacy.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.legacy.step(action)
        timeout = info.get("TimeLimit.truncated", False)
        return obs, reward, bool(done and not timeout), timeout, info

    def close(self):
        self.legacy.close()


def raw_episodes(lengths):
    n = sum(lengths)
    obs = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 17), dtype=np.float32)
    return {"observations": obs, "next_observations": obs + 1,
            "actions": np.zeros((n, 6), np.float32), "rewards": np.ones(n, np.float32),
            "terminals": np.isin(np.arange(n), np.cumsum(lengths) - 1),
            "timeouts": np.zeros(n, bool)}


class QuotaTests(unittest.TestCase):
    def test_exact_requested_endpoint_and_interior_quotas(self):
        expected = [(999995, 0), (749997, 249999), (499998, 499998), (249999, 749997), (0, 999995)]
        for clean, pair in zip((1., .75, .5, .25, 0.), expected):
            self.assertEqual(tuple(data.requested_quotas(spec(clean=clean)).values()), pair)

    def test_quota_math_matches_shared_collector_including_float_boundary(self):
        from rollout import transition_quotas
        for n in (1, 10, 100, 999995):
            for clean in (0., .1, .25, .5, .55, .7, 1.):
                expected = transition_quotas(n, [clean, 1-clean])
                self.assertEqual(list(data.requested_quotas(spec(clean=clean, n=n)).values()), expected.tolist())

    def test_invalid_requests_rejected_and_testing_seeds_supported(self):
        for field, value in (("seed", -1), ("seed", 2**32), ("seed", True), ("seed", 1.5),
                             ("num_samples", 0), ("num_samples", True), ("num_samples", 2.3),
                             ("clean_fraction", float("nan")), ("other_fraction", -0.2),
                             ("clean_fraction", .4), ("test_fraction", 1.), ("test_fraction", float("inf")),
                             ("protocol", "old"), ("setting", "medium-v2")):
            request = spec()
            request[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                data.requested_quotas(request)
        with self.assertRaises(ValueError):
            data.requested_quotas(spec(setting="clean"))
        for seed in (10000, 10100, 10200, 10300):
            data.requested_quotas(spec(seed=seed))


class CollectionTests(unittest.TestCase):
    def test_exact_rng_actions_reset_seeds_and_arrays_match_ordinary_collector(self):
        import rollout
        for clean in (0., .25, .5, .75, 1.):
            modern_envs, legacy_envs = [], []

            def modern_factory(_):
                env = ModernEnv()
                modern_envs.append(env)
                return env

            def legacy_factory():
                env = LegacyEnv()
                legacy_envs.append(env)
                return env

            with patch("rollout.gym.make", side_effect=modern_factory), \
                 patch("rollout.load_expert_policy", return_value=Actor()):
                expected, _ = rollout.collect_dataset("Walker2d-v5", "unused", max_timesteps=1000,
                    num_samples=57, noise_scale=.5, prop_clean_expert=clean,
                    prop_noisy_expert=1-clean, seed=10000)
            actual, names, parts, sources, _ = data.assemble_raw(spec(clean=clean, n=57), Actor(), legacy_factory)
            for key in data.RAW_KEYS:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{clean=}, {key=}")
            self.assertEqual([env.reset_seeds for env in legacy_envs],
                             [env.legacy.reset_seeds for env in modern_envs])
            self.assertTrue(all(env.closed for env in legacy_envs))
            self.assertEqual(len(sources), len(np.unique(expected["episode_ids"])))
            for name, part in zip(names, parts):
                quota = data.requested_quotas(spec(clean=clean, n=57))["clean" if name == "clean" else "other"]
                self.assertGreaterEqual(len(part["rewards"]), quota)
                self.assertLess(len(part["rewards"]), quota + 6)

    def test_clean_medium_order_and_clean_rng_are_independent_of_medium_selection(self):
        medium = raw_episodes([3, 5])
        request = spec(setting="clean-medium-v2", n=12)
        mixed, names, _, sources, medium_ids = data.assemble_raw(request, Actor(), LegacyEnv, medium, [4, 2])
        clean, _, _, _, _ = data.assemble_raw(spec(setting="clean", clean=1., n=6), Actor(), LegacyEnv)
        self.assertEqual(names, ["medium", "clean"])
        self.assertEqual(sources[:2], ["medium", "medium"])
        self.assertEqual(medium_ids[:2], [4, 2])
        self.assertTrue(all(value is None for value in medium_ids[2:]))
        for key in data.RAW_KEYS:
            np.testing.assert_array_equal(mixed[key][:8], medium[key])
            np.testing.assert_array_equal(mixed[key][8:], clean[key])

    def test_pure_medium_never_accesses_actor_or_collection_environment(self):
        raw = raw_episodes([3, 4])
        def never():
            self.fail("Generated-data environment should not be created")
        actual, names, _, sources, ids = data.assemble_raw(spec("clean-medium-v2", clean=0., n=7),
                                                         None, never, raw, [3, 9])
        self.assertEqual(names, ["medium"])
        self.assertEqual(sources, ["medium", "medium"])
        self.assertEqual(ids, [3, 9])
        for key in raw:
            np.testing.assert_array_equal(actual[key], raw[key])

    def test_reference_timeout_precedence_and_complete_episode_overshoot(self):
        env = LegacyEnv(horizon=1000)
        raw, episodes = data.collect_source(env, Actor(), 1, .5, np.random.default_rng(10100))
        self.assertEqual(episodes, 1)
        self.assertEqual(len(raw["rewards"]), 1000)
        self.assertTrue(raw["timeouts"][-1])
        self.assertFalse(raw["terminals"].any())
        self.assertLessEqual(float(np.abs(raw["actions"]).max()), 1.)

    def test_repeated_collection_is_identical_but_different_seeds_change_data(self):
        first = data.assemble_raw(spec(n=30), Actor(), LegacyEnv)[0]
        same = data.assemble_raw(spec(n=30), Actor(), LegacyEnv)[0]
        different = data.assemble_raw(spec(n=30, seed=10100), Actor(), LegacyEnv)[0]
        for key in first:
            np.testing.assert_array_equal(first[key], same[key])
        self.assertFalse(np.array_equal(first["observations"], different["observations"]))

    def test_failure_closes_collection_environment(self):
        env = LegacyEnv()
        with patch.object(env, "step", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                data.assemble_raw(spec(n=2), Actor(), lambda: env)
        self.assertTrue(env.closed)


class MediumSelectionAndConversionTests(unittest.TestCase):
    def test_without_replacement_prefix_and_capacity_rejection(self):
        ranges = np.array([[0, 3], [3, 8], [8, 15], [15, 19]])
        order = np.random.default_rng(10000).permutation(4)
        for quota in (1, 8, 18, 19):
            chosen = data.medium_selection(ranges, quota, 10000)
            self.assertEqual(chosen, order[:len(chosen)].tolist())
            self.assertEqual(len(chosen), len(set(chosen)))
            sizes = (ranges[:, 1] - ranges[:, 0])[chosen]
            self.assertGreaterEqual(sizes.sum(), quota)
            self.assertLess(sizes[:-1].sum(), quota)
        self.assertEqual(set(data.medium_selection(ranges, 19, 10000)), set(range(4)))
        for quota in (0, -1, 20):
            with self.assertRaises(ValueError):
                data.medium_selection(ranges, quota, 10000)

    def test_single_conversion_and_source_ids_cross_the_source_boundary(self):
        from offlinerlkit.utils.load_dataset import qlearning_dataset
        request = spec("clean-medium-v2", n=12)
        medium = raw_episodes([3, 5])
        medium["terminals"][-1] = False
        medium["timeouts"][-1] = True
        raw, names, parts, sources, _ = data.assemble_raw(request, Actor(), LegacyEnv, medium, [4, 2])
        converted = convert_dataset(None, raw, qlearning_dataset)
        expected = qlearning_dataset(None, dataset=raw)
        for key in expected:
            np.testing.assert_array_equal(converted[key], expected[key])
        counts = data.source_counts(converted, raw, names, parts, sources)
        self.assertEqual(counts["medium"], {"raw_transitions": 8, "processed_transitions": 7, "episodes": 2})
        self.assertEqual(counts["clean"]["processed_transitions"], len(parts[1]["rewards"]) - 1)
        self.assertEqual(sum(record["processed_transitions"] for record in counts.values()), len(converted["rewards"]))
        self.assertEqual(converted["episode_ids"][7], 2)


@unittest.skipUnless(os.environ.get("WALKER_REFERENCE_TESTS") == "1", "opt-in reference collection integration")
class LegacyRuntimeTests(unittest.TestCase):
    root = Path("/home/shekhe/walker-benchmark")
    worker = Path(__file__).with_name("walker_composition_data.py")

    def run_worker(self, operation, request, directory):
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request))
        output = directory / "output"
        if operation == "collect":
            output.mkdir()
        subprocess.run([str(self.root / "benchmark-python"), str(self.worker), "--reference-root", str(self.root),
                        operation, "--request", str(request_path), "--output", str(output)],
                       check=True, capture_output=True, text=True, timeout=180,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        return output

    def test_preflight_full_medium_capacity_and_reference_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_worker("preflight", {"datasets": [spec("clean-medium-v2", clean=0.)]}, Path(tmp))
            report = json.loads(output.read_text())
            self.assertEqual(report["medium_complete_transitions"], 999995)
            self.assertEqual(report["medium_complete_episodes"], 1190)
            self.assertEqual(report["environment"]["foot_friction"], [.9, 1.9])
            self.assertIsNone(report["expert"])

    def test_full_medium_endpoint_exact_conversion_unique_episodes_and_split(self):
        import h5py
        from offlinerlkit.utils.load_dataset import qlearning_dataset
        from rollout import split_dataset
        from walker_sweep import split_data

        request = spec("clean-medium-v2", clean=0.)
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_worker("collect", request, Path(tmp))
            meta = json.loads((output / "metadata.json").read_text())
            indices = meta["original_medium_episode_ids"]
            self.assertEqual(meta["raw_transitions"], 999995)
            self.assertEqual(meta["source_counts"]["medium"]["episodes"], 1190)
            self.assertEqual(len(indices), 1190)
            self.assertEqual(len(set(indices)), 1190)
            self.assertEqual(set(indices), set(range(1190)))
            self.assertEqual(indices, np.random.default_rng(10000).permutation(1190).tolist())
            self.assertEqual(meta["episode_sources"], ["medium"] * 1190)
            self.assertIsNone(meta["expert"])
            self.assertEqual(meta["actual_other_episode_fraction"], 1.)
            self.assertEqual(meta["actual_other_transition_fraction"], 1.)

            # Reconstruct the requested raw ordering directly from the canonical
            # source, excluding the five unfinished rows; no source duplication.
            with h5py.File(meta["medium_source"]["path"], "r") as handle:
                terminal, timeout = handle["terminals"][:], handle["timeouts"][:]
                ends = np.flatnonzero(terminal | timeout) + 1
                ranges = np.column_stack([np.r_[0, ends[:-1]], ends])
                self.assertEqual(int(ends[-1]), 999995)
                self.assertEqual(len(handle["rewards"]) - int(ends[-1]), 5)
                rows = np.concatenate([np.arange(start, end) for start, end in ranges[indices]])
                self.assertEqual(len(np.unique(rows)), 999995)
                self.assertEqual(int(rows.min()), 0)
                self.assertEqual(int(rows.max()), 999994)
                selected = {key: handle[key][:][rows] for key in data.RAW_KEYS}
            expected = convert_dataset(None, selected, qlearning_dataset)
            with np.load(output / "dataset.npz", allow_pickle=False) as saved:
                actual = {key: saved[key] for key in saved.files}
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(np.unique(actual["episode_ids"]), np.arange(1190))
            self.assertEqual(len(actual["rewards"]), meta["processed_transitions"])
            self.assertEqual(meta["source_counts"]["medium"]["processed_transitions"], len(actual["rewards"]))
            self.assertFalse(actual["timeouts"].any())
            del selected, expected, rows

            expected_parts = split_dataset(actual, test_fraction=.2, seed=10000)
            actual_parts = split_data(actual, fraction=.2, seed=10000)
            for expected_part, actual_part in zip(expected_parts, actual_parts):
                for key in actual:
                    np.testing.assert_array_equal(actual_part[key], expected_part[key])
            train_ids, test_ids = [set(part["episode_ids"]) for part in actual_parts]
            self.assertEqual((len(train_ids), len(test_ids)), (952, 238))
            self.assertFalse(train_ids & test_ids)
            self.assertEqual(train_ids | test_ids, set(range(1190)))
            self.assertEqual(sum(len(part["rewards"]) for part in actual_parts), len(actual["rewards"]))
            # Source episode identities remain disjoint too, not only assigned IDs.
            self.assertFalse({indices[i] for i in train_ids} & {indices[i] for i in test_ids})

    def test_small_reference_collection_both_families(self):
        for setting in ("noise0.5", "clean-medium-v2"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as tmp:
                output = self.run_worker("collect", spec(setting, n=10), Path(tmp))
                meta = json.loads((output / "metadata.json").read_text())
                self.assertEqual(meta["expert"]["actor_sha256"], data.EXPERT_ACTOR_SHA256)
                self.assertGreaterEqual(meta["raw_transitions"], 10)
                self.assertEqual(meta["dataset_spec"]["seed"], 10000)
                expected_order = ["clean", "noisy"] if setting == "noise0.5" else ["medium", "clean"]
                self.assertEqual(meta["source_order"], expected_order)
                with np.load(output / "dataset.npz") as handle:
                    self.assertEqual(len(handle["rewards"]), meta["processed_transitions"])
                    self.assertEqual(set(np.unique(handle["episode_ids"])), set(range(len(meta["episode_sources"]))))
                    self.assertFalse(handle["timeouts"].any())


if __name__ == "__main__":
    unittest.main()

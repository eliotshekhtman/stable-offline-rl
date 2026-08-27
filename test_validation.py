import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

import eval as evaluation
import rollout
import sweep
import validation


class TinyBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(3, 1, bias=False)


def make_dataset() -> dict[str, np.ndarray]:
    observations = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    next_observations = observations.copy()
    next_observations[:3, 0] += 1.0
    next_observations[3:, 0] += 1.0
    return {
        "observations": observations,
        "actions": np.zeros((6, 1), dtype=np.float32),
        "next_observations": next_observations,
        "rewards": np.arange(6, dtype=np.float32),
        "terminals": np.asarray([False, False, True, False, False, True]),
        "timeouts": np.zeros(6, dtype=bool),
        "episode_ids": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    }


def sha256(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


class ValidationTests(unittest.TestCase):
    def test_validation_backfills_without_changing_run_identity_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_parent = root / "trained" / "bc_test"
            run_dir = run_parent / "variant"
            checkpoint_dir = run_dir / "checkpoint" / "step_0"
            model_dir = run_dir / "model"
            record_dir = run_dir / "record"
            checkpoint_dir.mkdir(parents=True)
            model_dir.mkdir()
            record_dir.mkdir()

            dataset = make_dataset()
            dataset_dir = root / "dataset"
            paths = {}
            for name in ("full", "train", "test"):
                path = dataset_dir / f"{name}.npz"
                rollout.save_dataset(dataset, path)
                paths[name] = path.resolve()
            metadata_path = dataset_dir / "metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")

            policy = TinyBCPolicy()
            with torch.no_grad():
                policy.actor.weight.fill_(0.25)
            checkpoint_path = checkpoint_dir / "policy.pth"
            model_path = model_dir / "policy.pth"
            torch.save(policy.state_dict(), checkpoint_path)
            torch.save(policy.state_dict(), model_path)
            (record_dir / "policy_training_progress.csv").write_text(
                "timestep\n", encoding="utf-8"
            )

            training_schema = {
                "version": sweep.TRAINING_SCHEMA_VERSION,
                "identity": "unchanged",
            }
            manifest = {
                "env_name": "Pendulum-v1",
                "algo": "bc",
                "dataset_source": "minari",
                "chunk_length": 1,
                "base_discount": 0.99,
                "macro_discount": 0.99,
                "seed": 7,
                "batch_size": 2,
                "training_schema": training_schema,
                "model_dir": str(model_dir.resolve()),
                "full_dataset_path": str(paths["full"]),
                "train_dataset_path": str(paths["train"]),
                "test_dataset_path": str(paths["test"]),
                "dataset_metadata_path": str(metadata_path.resolve()),
                "checkpoints": [
                    {
                        "step": 0,
                        "requested_percent": 0,
                        "actual_percent": 0.0,
                        "policy_path": str(checkpoint_path.resolve()),
                    }
                ],
            }
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            before = {
                path: sha256(path)
                for path in (checkpoint_path, model_path, manifest_path)
            }

            self.assertEqual(
                sweep.find_trained_run(run_parent, training_schema),
                run_dir,
            )
            with patch.object(
                evaluation,
                "load_policy_and_dynamics",
                return_value=(policy, None, None, None),
            ):
                output_path = validation.validate_run(run_dir, "cpu")

            with output_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["validation_schema_version"]), 1)
            self.assertEqual(int(rows[0]["step"]), 0)
            self.assertEqual(int(float(rows[0]["heldout/num_transitions"])), 6)
            self.assertTrue(np.isfinite(float(rows[0]["heldout/action_mse"])))
            self.assertEqual(
                sweep.find_trained_run(run_parent, training_schema),
                run_dir,
            )
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest)

            with patch.object(
                validation.rollout,
                "load_dataset",
                side_effect=AssertionError("complete validation should be reused"),
            ):
                self.assertEqual(validation.validate_run(run_dir, "cpu"), output_path)

    def test_policy_dataset_uses_column_targets_and_training_reward_statistics(self):
        train_dataset = {
            "observations": np.zeros((2, 2), dtype=np.float32),
            "actions": np.zeros((2, 1), dtype=np.float32),
            "next_observations": np.zeros((2, 2), dtype=np.float32),
            "rewards": np.asarray([1.0, 3.0], dtype=np.float32),
            "terminals": np.zeros(2, dtype=bool),
        }
        heldout_dataset = {
            **train_dataset,
            "rewards": np.asarray([100.0, 200.0], dtype=np.float32),
        }
        manifest = {
            "algo": "mopo",
            "training_schema": {
                "model_based": {
                    "manipulation_settings": {"reward_normalization": "zscore"}
                }
            },
        }

        prepared = validation.prepare_policy_dataset(
            manifest, None, train_dataset, heldout_dataset, None, None
        )

        self.assertEqual(prepared["rewards"].shape, (2, 1))
        self.assertEqual(prepared["terminals"].shape, (2, 1))
        expected = (heldout_dataset["rewards"] - 2.0) / 1.001
        np.testing.assert_allclose(prepared["rewards"][:, 0], expected)

    def test_validation_cache_requires_matching_schema_and_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.csv"
            path.write_text(
                "validation_schema_version,step\n1,0\n", encoding="utf-8"
            )
            checkpoints = [{"step": 0}]

            self.assertTrue(validation.validation_is_complete(path, checkpoints))
            self.assertFalse(
                validation.validation_is_complete(path, [{"step": 100}])
            )
            path.write_text(
                "validation_schema_version,step\n0,0\n", encoding="utf-8"
            )
            self.assertFalse(validation.validation_is_complete(path, checkpoints))


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

import policies
import sweep
from offlinerlkit.policy import RAMBOPolicy


class DummyEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self.spec = SimpleNamespace(id="HalfCheetah-v5")


class RetiredAlgorithmTests(unittest.TestCase):
    @staticmethod
    def parse(*extra):
        with patch.object(
            sys, "argv", ["sweep.py", "--env", "Reacher-v5", *extra]
        ):
            return sweep.parse_args()

    def test_cli_rejects_retired_algorithms(self):
        for algo in ("dql", "rambo"):
            with self.subTest(algo=algo), self.assertRaises(SystemExit):
                self.parse("--algos", algo)

    def test_registries_distinguish_training_from_legacy_loading(self):
        self.assertNotIn("dql", policies.TRAINABLE_MODEL_FREE_ALGOS)
        self.assertIn("dql", policies.LOADABLE_MODEL_FREE_ALGOS)
        self.assertIn("dql", policies.MODEL_FREE_ALGOS)
        self.assertNotIn("rambo", policies.TRAINABLE_MODEL_BASED_ALGOS)
        self.assertIn("rambo", policies.LOADABLE_MODEL_BASED_ALGOS)
        self.assertIn("rambo", policies.MODEL_BASED_ALGOS)

    def test_programmatic_training_rejects_legacy_only_algorithms(self):
        for algo in ("dql", "rambo"):
            with self.subTest(algo=algo), self.assertRaisesRegex(
                ValueError, "retained only for loading legacy runs"
            ):
                sweep.train_algo(
                    algo=algo,
                    env_name="unused",
                    primitive_dataset={},
                    chunk_dataset={},
                    chunk_length=1,
                    run_dir=Path("unused"),
                    eval_dir=Path("unused"),
                    split_paths={},
                    training_schema={},
                    args=SimpleNamespace(),
                )

    def test_retired_algorithm_tuning_options_are_removed_from_cli(self):
        args = self.parse()
        for attribute in (
            "dql_eta",
            "dql_weight_temperature",
            "dql_reward_normalization",
            "dynamics_update_freq",
            "adv_batch_size",
            "adv_weight",
            "bc_epoch",
            "bc_batch_size",
        ):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(args, attribute))

        for option in ("--dql-eta", "--adv-weight", "--bc-epoch"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self.parse(option, "1")

    def test_normal_imports_do_not_import_dql(self):
        script = "import sys; import policies, sweep; assert 'dql' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_manifest_omits_retired_algorithm_fields(self):
        args = self.parse()
        args.seed = 7
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            sweep.save_run_manifest(
                run_dir=run_dir,
                eval_dir=root / "eval",
                algo="mopo",
                env_name="Reacher-v5",
                split_paths={"dataset_tag": "test"},
                training_schema={"model_based": {}},
                chunk_length=1,
                macro_discount=0.99,
                args=args,
            )

            saved = json.loads((run_dir / "run_manifest.json").read_text())

        for field in ("adv_weight", "adv_batch_size", "dql_config"):
            self.assertNotIn(field, saved)
        self.assertEqual(saved["rollout_length"], args.rollout_length)

    def test_dql_builder_is_retained_and_imported_on_demand(self):
        legacy_policy = object()
        builder = Mock(return_value=legacy_policy)
        dql_module = types.ModuleType("dql")
        dql_module.build_dql_policy = builder
        args = SimpleNamespace(epoch=3, step_per_epoch=4, device="cpu")

        with patch.dict(sys.modules, {"dql": dql_module}):
            policy, scheduler = policies.build_model_free_policy(
                "dql",
                DummyEnv(),
                buffer=object(),
                args=args,
                discount=0.99,
                dql_config={"legacy": True},
            )

        self.assertIs(policy, legacy_policy)
        self.assertIsNone(scheduler)
        self.assertEqual(builder.call_args.kwargs["total_steps"], 12)
        self.assertEqual(builder.call_args.kwargs["config"], {"legacy": True})

    def test_rambo_builder_is_retained_for_legacy_checkpoints(self):
        args = SimpleNamespace(
            device="cpu",
            epoch=1,
            model_manipulation_settings=False,
            model_actor_learning_rate=None,
            model_critic_learning_rate=3e-4,
            mopo_penalty_coef=0.5,
            adv_weight=3e-4,
            rollout_length=5,
            adv_batch_size=16,
        )

        policy, dynamics, scheduler = policies.build_model_based_policy(
            "rambo",
            DummyEnv(),
            args,
            discount=0.99,
            obs_mean=np.zeros((1, 2), dtype=np.float32),
            obs_std=np.ones((1, 2), dtype=np.float32),
        )

        self.assertIsInstance(policy, RAMBOPolicy)
        self.assertIs(policy.dynamics, dynamics)
        self.assertIsNone(scheduler)


if __name__ == "__main__":
    unittest.main()

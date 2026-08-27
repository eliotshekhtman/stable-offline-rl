import sys
import unittest
from unittest.mock import patch

import numpy as np

from policies import termination_fn_never
import sweep


class MobileShiftArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        with patch.object(sys, "argv", ["sweep.py", "--env", "Reacher-v5", *extra]):
            return sweep.parse_args()

    def test_default_and_override(self):
        defaults = self.parse()
        self.assertEqual(defaults.mobile_return_shift, 30.0)
        self.assertEqual(defaults.mopo_penalty_coef, 0.5)
        self.assertEqual(defaults.mobile_penalty_coef, 1.5)
        self.assertIsNone(defaults.model_actor_learning_rate)
        self.assertEqual(defaults.model_critic_learning_rate, 3e-4)
        self.assertFalse(defaults.model_manipulation_settings)
        self.assertEqual(
            self.parse("--mobile-return-shift", "17.5").mobile_return_shift,
            17.5,
        )

    def test_output_and_evaluation_defaults(self):
        args = self.parse()
        self.assertFalse(args.quiet)
        self.assertEqual(args.checkpoint_eval_episodes, 20)
        self.assertEqual(args.final_eval_episodes, 100)
        self.assertTrue(self.parse("--quiet").quiet)

    def test_negative_shift_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--mobile-return-shift", "-1")

    def test_negative_model_penalty_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--mopo-penalty-coef", "-1")
        with self.assertRaises(SystemExit):
            self.parse("--mobile-penalty-coef", "-1")
        with self.assertRaises(SystemExit):
            self.parse("--model-actor-learning-rate", "0")
        with self.assertRaises(SystemExit):
            self.parse("--model-critic-learning-rate", "0")

    def test_model_actor_learning_rate_is_recorded(self):
        args = self.parse("--model-actor-learning-rate", "3e-5")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["model_based"]["actor_learning_rate"], 3e-5)

    def test_model_critic_learning_rate_is_recorded(self):
        args = self.parse("--model-critic-learning-rate", "1e-4")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["model_based"]["critic_learning_rate"], 1e-4)

    def test_reacher_mobile_schema_records_shifted_clamp(self):
        args = self.parse("--mobile-return-shift", "20")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "Reacher-v5", {"source": "test"}, 1, args
        )

        self.assertEqual(
            schema["mobile"],
            {"return_shift": 20.0, "clamp_target_q": True},
        )

    def test_non_reacher_schema_is_unchanged(self):
        args = self.parse()
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "HalfCheetah-v5", {"source": "test"}, 1, args
        )

        self.assertNotIn("mobile", schema)

    def test_lift_model_based_schema_records_synthetic_termination(self):
        args = self.parse()
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(
            schema["model_based"]["synthetic_termination"],
            "never",
        )

    def test_manipulation_model_schema_records_changed_training(self):
        args = self.parse(
            "--model-manipulation-settings",
            "--mobile-penalty-coef", "1.0",
        )
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["mobile"]["penalty_coef"], 1.0)
        self.assertEqual(schema["mobile"]["num_critics"], 10)
        self.assertTrue(schema["mobile"]["max_q_backup"])
        self.assertEqual(
            schema["model_based"]["manipulation_settings"]["reward_normalization"],
            "zscore",
        )

    def test_lift_synthetic_transitions_never_terminate(self):
        next_obs = np.zeros((3, 19), dtype=np.float32)
        next_obs[:, 11] = [0.83, 0.84, 0.85]

        np.testing.assert_array_equal(
            termination_fn_never(np.zeros((3, 19)), None, next_obs),
            [[False], [False], [False]],
        )


if __name__ == "__main__":
    unittest.main()

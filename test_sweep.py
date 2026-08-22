import sys
import unittest
from unittest.mock import patch

import sweep


class MobileShiftArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        with patch.object(sys, "argv", ["sweep.py", "--env", "Reacher-v5", *extra]):
            return sweep.parse_args()

    def test_default_and_override(self):
        self.assertEqual(self.parse().mobile_return_shift, 30.0)
        self.assertEqual(
            self.parse("--mobile-return-shift", "17.5").mobile_return_shift,
            17.5,
        )

    def test_negative_shift_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--mobile-return-shift", "-1")

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


if __name__ == "__main__":
    unittest.main()

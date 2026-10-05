"""Shared display names must never alter algorithm identities or ratios."""

import unittest

from plot_labels import mobile_names, real_ratio_label


class MobileNamesTests(unittest.TestCase):
    def test_context_rules(self):
        cases = [
            (["iql", "mobile", "mobile"], [None, 0.5, 0],
             [None, "Hybrid-MOBILE", "MB-MOBILE"]),
            (["mobile"] * 5, [0, 0.25, 0.5, 0.75, 1],
             ["real ratio=0.00", "real ratio=0.25", "real ratio=0.50",
              "real ratio=0.75", "real ratio=1.00"]),
            (["mobile"], [0], [None]),
            (["iql", "mobile"], [None, 0.5], [None, None]),
            (["mopo", "mobile", "mobile"], [0, 0.25, 0.5], [None] * 3),
            (["iql", "mobile", "mobile", "mobile"], [None, 0, 0.25, 0.5],
             [None, "MB-MOBILE", "Hybrid-MOBILE (real ratio=0.25)",
              "Hybrid-MOBILE (real ratio=0.50)"]),
            (["iql", "mobile", "mobile"], [None, None, 0.5], [None] * 3),
        ]
        for algorithms, ratios, expected in cases:
            with self.subTest(algorithms=algorithms, ratios=ratios):
                before = (list(algorithms), list(ratios))
                self.assertEqual(mobile_names(algorithms, ratios), expected)
                self.assertEqual((algorithms, ratios), before)

    def test_ratio_precision_does_not_turn_positive_into_zero(self):
        self.assertEqual(real_ratio_label(0.05), "real ratio=0.05")
        self.assertEqual(real_ratio_label(0.001), "real ratio=0.001")
        self.assertEqual(mobile_names(["mobile", "mobile"], [0, 0.001]),
                         ["real ratio=0.00", "real ratio=0.001"])

    def test_invalid_ratios_are_not_silently_classified(self):
        for ratio in (True, -0.1, 1.1, float("nan"), float("inf"), "0"):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "real ratio"):
                mobile_names(["mobile"], [ratio])
        with self.assertRaisesRegex(ValueError, "one real ratio"):
            mobile_names(["mobile"], [])


if __name__ == "__main__":
    unittest.main()

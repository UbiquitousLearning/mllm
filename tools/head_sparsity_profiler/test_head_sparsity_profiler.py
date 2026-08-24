from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import head_sparsity_profiler as profiler


class FakeTokenizer:
    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [len(word) for word in text.split()]}


class CalibrationTest(unittest.TestCase):
    def test_ae_line_filter_concatenation_and_complete_windows(self) -> None:
        text = "short\nalpha beta gamma delta\nepsilon zeta eta theta\n"
        self.assertEqual(
            profiler.build_calibration_windows(FakeTokenizer(), text, 3),
            [[5, 4, 5], [5, 7, 4]],
        )

    def test_sample_cap(self) -> None:
        text = "one two three four five six seven eight\n"
        self.assertEqual(
            profiler.build_calibration_windows(FakeTokenizer(), text, 2, 2),
            [[3, 3], [5, 4]],
        )


class ConversionTest(unittest.TestCase):
    def test_parse_unsorted_ae_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heads = root / "heads.txt"
            layers = root / "layers.txt"
            heads.write_text(
                "qwen 1 1 8\nqwen 0 0 2\nqwen 1 0 7\nqwen 0 1 3\n",
                encoding="utf-8",
            )
            layers.write_text("qwen 1 5\nqwen 0 4\n", encoding="utf-8")
            model, head_values, layer_values = profiler.read_ae_results(heads, layers)
            self.assertEqual(model, "qwen")
            self.assertEqual(head_values, [[2.0, 3.0], [7.0, 8.0]])
            self.assertEqual(layer_values, [4.0, 5.0])

    def test_reject_duplicate_and_incomplete_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heads = root / "heads.txt"
            layers = root / "layers.txt"
            layers.write_text("m 0 1\n", encoding="utf-8")
            heads.write_text("m 0 0 1\nm 0 0 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                profiler.read_ae_results(heads, layers)

    def test_ae_importance_and_budget_without_caps(self) -> None:
        ratios, importance, leftover = profiler.allocate_ae_retentions(
            [[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0], 0.25, 102400.0
        )
        self.assertEqual(importance, [[10.0, 20.0], [60.0, 80.0]])
        expected = [[10.0 / 170.0, 20.0 / 170.0], [60.0 / 170.0, 80.0 / 170.0]]
        for actual_row, expected_row in zip(ratios, expected):
            for actual, wanted in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, wanted)
        self.assertAlmostEqual(sum(map(sum, ratios)), 1.0)
        self.assertEqual(leftover, 0.0)

    def test_ae_sequential_cap_redistribution(self) -> None:
        ratios, _, leftover = profiler.allocate_ae_retentions(
            [[100.0, 1.0, 1.0, 1.0]], [1.0], 0.5, 1e9
        )
        self.assertEqual(ratios[0][0], 1.0)
        for value in ratios[0][1:]:
            self.assertAlmostEqual(value, 1.0 / 3.0)
        self.assertAlmostEqual(sum(ratios[0]), 2.0)
        self.assertAlmostEqual(leftover, 0.0)

    def test_ae_float_evaluation_order_is_stable(self) -> None:
        heads = [
            [1.0 + (layer * 12 + head) / 1000.0 for head in range(12)]
            for layer in range(28)
        ]
        layers = [1.0 + layer / 10.0 for layer in range(28)]
        ratios, _, _ = profiler.allocate_ae_retentions(heads, layers)
        digest = hashlib.sha256(
            profiler.format_retention_profile(ratios).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            digest,
            "473271b37b9c34fd6e9b9f795d8c8b0814a3735e2b45c1386363e7544bba53cc",
        )

    def test_profile_format_matches_ae(self) -> None:
        self.assertEqual(
            profiler.format_retention_profile([[1.0, 0.2], [0.125, 0.75]]),
            "1.0 0.2 \n0.125 0.75 \n",
        )

    def test_convert_measurement_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measurements.json"
            output = root / "profile.txt"
            source.write_text(
                json.dumps(
                    {
                        "model": "tiny",
                        "head_perplexity": [[2.0, 4.0]],
                        "layer_perplexity": [3.0],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                profiler.main(
                    [
                        "convert",
                        "--measurements",
                        str(source),
                        "--average-retention",
                        "0.5",
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            values = [float(value) for value in output.read_text().split()]
            self.assertEqual(len(values), 2)
            self.assertTrue(math.isclose(sum(values) / 2, 0.5))
            report = json.loads(
                output.with_suffix(".txt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["method"], "ShadowNPU-AE-offline-get_ratios")


if __name__ == "__main__":
    unittest.main()

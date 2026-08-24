import json
import tempfile
import unittest
from pathlib import Path

import hmx_scale_calibrator as calibrator


SAMPLE_LOG = """
[HMX_INT8_BUCKET] op=x layer=2 key_len=256 q=1 k=1 output=9 requant=0.1 matrices=2: 0(q=0.1,k=0.2) 1(q=0.2,k=0.4)
[HMX_INT8_OUTPUT_SCALE] op=x layer=2 key_len=256 matrix=0 query_begin=0 requant=0.01 output=2 q_scale=0.1 k_scale=0.2 peak=112 attempts=2
[HMX_INT8_OUTPUT_SCALE] op=x layer=2 key_len=256 matrix=1 query_begin=0 requant=0.02 output=4 q_scale=0.2 k_scale=0.4 peak=108 attempts=1
[HMX_INT8_RECALL] op=x layer=2 key_len=256 matrix=0 recall=0.95 unique_scores=100 zero_fraction=0.01 saturation_fraction=0 score_mismatch_fraction=0 measured_q=0.1 measured_k=0.2 bucket_q=0.1 bucket_k=0.2 used_q=0.1 used_k=0.2
QUALITY_RESULT expected_code=NPU-7391 extracted_code=NPU-7391 retrieval_exact=1
"""


class ScaleCalibratorTest(unittest.TestCase):
    def _log(self, root: Path, model: str = "qwen2_1p5b") -> Path:
        path = root / "run.log"
        path.write_text(SAMPLE_LOG, encoding="utf-8")
        Path(str(path) + ".json").write_text(
            json.dumps({"model": model}), encoding="utf-8"
        )
        return path

    def test_parse_prefers_bucket_qk_over_recall_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(Path(directory))
            parsed = calibrator.parse_calibration_logs(
                [path], "qwen2_1p5b", require_model_sidecar=True
            )
        self.assertEqual(len(parsed.qk), 2)
        self.assertEqual(len(parsed.output), 2)
        self.assertEqual(len(parsed.recall), 1)
        self.assertEqual(parsed.qk[1].layer, 2)
        self.assertTrue(parsed.quality_results[0])

    def test_profile_is_model_bound_and_emits_model_specific_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(Path(directory))
            parsed = calibrator.parse_calibration_logs([path], "qwen2_1p5b")
            profile = calibrator.make_model_scale_profile(
                parsed, "qwen2_1p5b", 128, [0.5, 1.0, 2.0], 1.0, 1.05,
                "abc",
            )
        self.assertEqual(profile["model"], "qwen2_1p5b")
        self.assertEqual(profile["model_sha256"], "abc")
        self.assertAlmostEqual(profile["q"]["base_scale"], 0.15)
        for actual, expected in zip(
            profile["q"]["buckets"], [0.075, 0.15, 0.3]
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(profile["output"]["fixed_scale"], 4.2)
        self.assertAlmostEqual(profile["output"]["target_requant_scale"], 0.01)

    def test_rejects_cross_model_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._log(Path(directory), model="wrong_model")
            with self.assertRaisesRegex(ValueError, "belongs to"):
                calibrator.parse_calibration_logs([path], "qwen2_1p5b")

    def test_cli_writes_profile_catalog_and_build_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = self._log(root)
            output = root / "scale.json"
            status = calibrator.main(
                [
                    "calibrate", "--log", str(log), "--model", "qwen2_1p5b",
                    "--head-dim", "128", "--require-model-sidecar",
                    "--output", str(output),
                ]
            )
            self.assertEqual(status, 0)
            profile = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(profile["model"], "qwen2_1p5b")
            self.assertIn("qwen2_1p5b 128", output.with_suffix(".catalog.txt").read_text())
            build_env = output.with_suffix(".build.env").read_text()
            self.assertIn("HMX_OPERATOR_TARGET_REQUANT_SCALE=", build_env)
            self.assertIn("HMX_OPERATOR_Q_SCALES=", build_env)


if __name__ == "__main__":
    unittest.main()

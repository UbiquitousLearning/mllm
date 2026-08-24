import argparse
import tempfile
import unittest
from pathlib import Path

import pipeline_latency_profiler as profiler


class PipelineLatencyProfilerTest(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path, list[tuple[float, float]]]:
        pairs = [(float(q), float(k)) for q in (1, 2, 4) for k in (1, 2, 4)]
        manifest = root / "manifest.tsv"
        lines = [
            "model\theads\toperator_so\tdsp_skel\tm\tk\tn\tq_scale\tk_scale\toutput_scale\trequant_scale"
        ]
        for q_scale, k_scale in pairs:
            lines.append(
                "qwen2_1p5b\t12\top.so\tskel.so\t128\t128\t4160\t"
                f"{q_scale}\t{k_scale}\t1\t1"
            )
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        heads = root / "heads.txt"
        heads.write_text(
            "".join(
                " ".join((["1.0"] if layer < 2 else ["0.2"]) * 12) + "\n"
                for layer in range(28)
            ),
            encoding="utf-8",
        )
        return manifest, heads, pairs

    def make_raw(
        self, root: Path, pairs: list[tuple[float, float]], repetitions: int
    ) -> Path:
        raw = root / "raw.tsv"
        lines: list[str] = []
        for _ in range(repetitions):
            for q_scale, k_scale in pairs:
                lines.append(f"npu\t128\t12\t{q_scale}\t{k_scale}\t100")
            for layer in range(2, 28):
                for head in range(12):
                    lines.append(f"topk\t128\t{layer}\t{head}\t0.2\t30")
                    lines.append(f"sparse\t128\t{layer}\t{head}\t0.2\t50")
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return raw

    def test_inspect_reports_exact_smoke_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, heads, _ = self.make_inputs(root)
            result = profiler.inspect_inputs(
                manifest, heads, "qwen2_1p5b", 128, 128, 3
            )
        self.assertEqual(result["npu_head_counts"], [12])
        self.assertEqual(result["aggregate_rows"]["npu"], 9)
        self.assertEqual(result["aggregate_rows"]["topk"], 312)
        self.assertEqual(result["aggregate_rows"]["sparse"], 312)
        self.assertEqual(result["aggregate_rows"]["total"], 633)
        self.assertEqual(result["minimum_raw_rows"], 1899)

    def test_aggregate_and_strict_validate_complete_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, heads, pairs = self.make_inputs(root)
            raw = self.make_raw(root, pairs, 3)
            binary = root / "profile_hmx_pipeline_latency"
            binary.write_bytes(b"test-binary")
            output = root / "runtime.tsv"
            args = argparse.Namespace(
                raw=raw,
                head_raw=None,
                manifest=manifest,
                head_profile=heads,
                binary=binary,
                model="qwen2_1p5b",
                device="test-device",
                main_cpu="5",
                topk_cpus="0,1",
                sparse_cpus="5",
                query_len=128,
                max_key_len=128,
                minimum_samples=3,
                output=output,
            )
            result = profiler.aggregate(args)
            second = profiler.validate_profile(
                output,
                manifest,
                heads,
                binary,
                "qwen2_1p5b",
                "test-device",
                "5",
                "0,1",
                "5",
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["rows"], {"npu": 9, "topk": 312, "sparse": 312})
        self.assertEqual(second["profile_sha256"], result["profile_sha256"])

    def test_validate_rejects_missing_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, heads, pairs = self.make_inputs(root)
            raw = self.make_raw(root, pairs, 1)
            binary = root / "binary"
            binary.write_bytes(b"x")
            output = root / "runtime.tsv"
            args = argparse.Namespace(
                raw=raw,
                head_raw=None,
                manifest=manifest,
                head_profile=heads,
                binary=binary,
                model="qwen2_1p5b",
                device="device",
                main_cpu="5",
                topk_cpus="0,1",
                sparse_cpus="5",
                query_len=128,
                max_key_len=128,
                minimum_samples=1,
                output=output,
            )
            profiler.aggregate(args)
            original = output.read_text(encoding="utf-8").splitlines()
            data_index = next(
                index for index, line in enumerate(original) if line.startswith("sparse\t")
            )
            missing = root / "missing.tsv"
            missing.write_text(
                "\n".join(original[:data_index] + original[data_index + 1 :]) + "\n",
                encoding="utf-8",
            )
            duplicate = root / "duplicate.tsv"
            duplicate.write_text(
                "\n".join(original + [original[data_index]]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sparse grid is incomplete"):
                profiler.validate_profile(missing)
            with self.assertRaisesRegex(ValueError, "duplicate sparse row"):
                profiler.validate_profile(duplicate)

    def test_rejects_non_qwen_v1_head_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen2.5.txt"
            path.write_text(
                "".join(" ".join(["0.2"] * 16) + "\n" for _ in range(24)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "28x12"):
                profiler.load_head_profile(path)

    def test_rejects_non_dense_prefix_not_profiled_by_native_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, heads, _ = self.make_inputs(root)
            rows = heads.read_text(encoding="utf-8").splitlines()
            rows[1] = " ".join(["0.9"] * 12)
            heads.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profiles L2--L27"):
                profiler.inspect_inputs(
                    manifest, heads, "qwen2_1p5b", 128, 128, 3
                )

    def test_device_command_quotes_session_environment(self):
        args = argparse.Namespace(
            topk_cpus="0,1",
            sparse_cpus="5",
            main_cpu="6",
            warmup=3,
            repetitions=20,
            max_key_len=4160,
            query_len=128,
            stages="all",
            topk_mode="cooperative",
            env=[
                "LD_PRELOAD=/data/local/tmp/lib redirect.so",
                "ADSP_LIBRARY_PATH=/data/local/tmp/bank;/vendor/lib/rfsa/adsp",
            ],
            device_binary="/data/local/tmp/profile_hmx_pipeline_latency",
            device_manifest="/data/local/tmp/bank/manifest.tsv",
            device_head_profile="/data/local/tmp/heads.txt",
            device_raw="/data/local/tmp/raw.tsv",
        )
        command, environment = profiler.make_device_command(args)
        self.assertEqual(environment["MLLM_HMX_PIPELINE_TOPK_WORKERS"], "2")
        self.assertEqual(environment["MLLM_HMX_PIPELINE_SPARSE_WORKERS"], "1")
        self.assertIn("'LD_PRELOAD=/data/local/tmp/lib redirect.so'", command)
        self.assertIn("'ADSP_LIBRARY_PATH=/data/local/tmp/bank;/vendor/lib/rfsa/adsp'", command)


if __name__ == "__main__":
    unittest.main()

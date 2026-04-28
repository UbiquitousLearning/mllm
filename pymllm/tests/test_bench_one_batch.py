from __future__ import annotations

import pytest
import torch

from pymllm.configs.global_config import GlobalConfig
from pymllm.bench_one_batch import (
    BenchArgs,
    BenchSetting,
    generate_settings,
    make_profile_trace_path,
    make_synthetic_input_ids,
    parse_args,
    summarize_latencies,
)


@pytest.fixture(autouse=True)
def _reset_global_config():
    GlobalConfig.reset()
    yield
    GlobalConfig.reset()


def test_parse_args_accepts_server_config_and_list_bench_args(tmp_path):
    model_dir = tmp_path / "model"
    result_file = tmp_path / "bench.jsonl"
    model_dir.mkdir()

    cfg, bench_args = parse_args(
        [
            "--server.model_path",
            str(model_dir),
            "--server.dtype",
            "float16",
            "--quantization.method",
            "compressed-tensors",
            "--run-name",
            "unit",
            "--batch-size",
            "1",
            "4",
            "--input-len",
            "256",
            "512",
            "--output-len",
            "8",
            "16",
            "--result-filename",
            str(result_file),
            "--profile-stage",
            "decode",
            "--profile-activities",
            "CPU",
            "GPU",
        ]
    )

    assert cfg.server.model_path == model_dir
    assert cfg.server.tokenizer_path == model_dir
    assert cfg.server.dtype == "float16"
    assert cfg.quantization.method == "compressed-tensors"
    assert bench_args.run_name == "unit"
    assert bench_args.batch_size == [1, 4]
    assert bench_args.input_len == [256, 512]
    assert bench_args.output_len == [8, 16]
    assert bench_args.result_filename == result_file
    assert bench_args.profile_stage == "decode"
    assert bench_args.profile_activities == ["CPU", "GPU"]


def test_generate_settings_has_stable_batch_input_output_order(tmp_path):
    args = BenchArgs(
        batch_size=[1, 2],
        input_len=[256, 512],
        output_len=[8],
        result_filename=tmp_path / "out.jsonl",
    )

    assert generate_settings(args) == [
        BenchSetting(batch_size=1, input_len=256, output_len=8),
        BenchSetting(batch_size=1, input_len=512, output_len=8),
        BenchSetting(batch_size=2, input_len=256, output_len=8),
        BenchSetting(batch_size=2, input_len=512, output_len=8),
    ]


def test_make_synthetic_input_ids_is_seeded_int32_and_vocab_capped():
    first = make_synthetic_input_ids(
        batch_size=2,
        input_len=4,
        vocab_size=50_000,
        seed=123,
        device="cpu",
    )
    second = make_synthetic_input_ids(
        batch_size=2,
        input_len=4,
        vocab_size=50_000,
        seed=123,
        device="cpu",
    )

    assert first.shape == (2, 4)
    assert first.dtype == torch.int32
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < 10_000


def test_summarize_latencies_matches_sglang_style_metrics():
    setting = BenchSetting(batch_size=2, input_len=256, output_len=4)

    result = summarize_latencies(
        setting=setting,
        prefill_latency=0.5,
        decode_latencies=[0.1, 0.2, 0.3],
        run_name="unit",
        device="cuda",
        dtype="torch.float16",
        cuda_graph=True,
    )

    assert result["run_name"] == "unit"
    assert result["batch_size"] == 2
    assert result["input_len"] == 256
    assert result["output_len"] == 4
    assert result["prefill_latency"] == 0.5
    assert result["prefill_throughput"] == pytest.approx(1024.0)
    assert result["median_decode_latency"] == pytest.approx(0.2)
    assert result["median_decode_throughput"] == pytest.approx(10.0)
    assert result["total_latency"] == pytest.approx(1.1)
    assert result["overall_throughput"] == pytest.approx((260 * 2) / 1.1)
    assert result["device"] == "cuda"
    assert result["dtype"] == "torch.float16"
    assert result["cuda_graph"] is True


def test_make_profile_trace_path_is_deterministic_and_sanitized(tmp_path):
    path = make_profile_trace_path(
        output_dir=tmp_path,
        prefix="pymllm_profile",
        run_name="qwen3/vl w8a8",
        setting=BenchSetting(batch_size=1, input_len=256, output_len=8),
        stage="decode",
    )

    assert path.parent == tmp_path
    assert path.name == "pymllm_profile_qwen3_vl_w8a8_bs1_in256_out8_decode.trace.json"

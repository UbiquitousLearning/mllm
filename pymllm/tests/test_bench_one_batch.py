from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from pymllm.configs.global_config import GlobalConfig
from pymllm.bench_one_batch import (
    BenchArgs,
    BenchSetting,
    PymllmBenchRunner,
    generate_settings,
    make_multimodal_bench_input_from_processor_output,
    make_profile_trace_path,
    make_synthetic_input_ids,
    make_vit_prefill_metrics,
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
            "--image",
            str(tmp_path / "image.jpg"),
            "--prompt",
            "What is in this image?",
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
    assert bench_args.image_path == tmp_path / "image.jpg"
    assert bench_args.prompt == "What is in this image?"


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


def test_generate_settings_uses_processed_prompt_length_for_image_mode(tmp_path):
    args = BenchArgs(
        batch_size=[1, 2],
        input_len=[256, 512],
        output_len=[8],
        result_filename=tmp_path / "out.jsonl",
        image_path=tmp_path / "image.jpg",
    )

    assert generate_settings(args) == [
        BenchSetting(batch_size=1, input_len=0, output_len=8),
        BenchSetting(batch_size=2, input_len=0, output_len=8),
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


def test_make_vit_prefill_metrics_reports_seconds_and_tps():
    result = make_vit_prefill_metrics(vit_prefill_ms=12.5, vit_prefill_tokens=25)

    assert result == {
        "vit_prefill_latency": pytest.approx(0.0125),
        "vit_prefill_ms": pytest.approx(12.5),
        "vit_prefill_tokens": 25,
        "vit_prefill_throughput": pytest.approx(2000.0),
        "vit_prefill_tps": pytest.approx(2000.0),
    }


def test_make_multimodal_bench_input_repeats_processor_output_per_batch():
    bench_input = make_multimodal_bench_input_from_processor_output(
        {
            "input_ids": torch.tensor([[1, 5, 5, 2]], dtype=torch.int64),
            "pixel_values": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        },
        batch_size=3,
        image_token_id=5,
        device="cpu",
    )

    assert bench_input.input_ids.shape == (3, 4)
    assert bench_input.input_ids.dtype == torch.int32
    assert bench_input.vit_prefill_tokens == 6
    torch.testing.assert_close(
        bench_input.pixel_values,
        torch.arange(6, dtype=torch.float32).reshape(2, 3).repeat(3, 1),
    )
    torch.testing.assert_close(
        bench_input.image_grid_thw,
        torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]], dtype=torch.int64),
    )


def test_extend_attaches_multimodal_inputs_and_returns_vit_metrics():
    class _ReqPool:
        def alloc(self, batch_size):
            return list(range(batch_size))

        def write(self, index, value):
            del index, value

        def clear(self):
            pass

    class _KvPool:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def clear(self):
            pass

    class _Runner:
        def __init__(self):
            self.device = "cpu"
            self.dtype = torch.float32
            self.req_to_token_pool = _ReqPool()
            self.token_to_kv_pool_allocator = _KvPool()
            self.gdn_pool = None
            self.last_forward_batch = None

        def prepare_forward_batch_extend(self, **kwargs):
            return SimpleNamespace(batch_size=kwargs["req_pool_indices"].shape[0])

        def forward(self, forward_batch):
            self.last_forward_batch = forward_batch
            forward_batch.vit_prefill_ms = 4.0
            forward_batch.vit_prefill_tokens = 8
            return object()

        def sample(self, logits_output, forward_batch, **kwargs):
            del logits_output, forward_batch, kwargs
            return torch.tensor([7], dtype=torch.int32)

    fake_runner = _Runner()
    bench_runner = PymllmBenchRunner(fake_runner)
    pixel_values = torch.ones((2, 3), dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)

    result = bench_runner.extend(
        torch.tensor([[1, 5, 5, 2]], dtype=torch.int32),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        benchmark_vision_timing=True,
    )

    assert result.vit_prefill_ms == pytest.approx(4.0)
    assert result.vit_prefill_tokens == 8
    torch.testing.assert_close(fake_runner.last_forward_batch.pixel_values, pixel_values)
    torch.testing.assert_close(
        fake_runner.last_forward_batch.image_grid_thw,
        image_grid_thw,
    )
    assert fake_runner.last_forward_batch.benchmark_vision_timing is True


def test_make_profile_trace_path_is_deterministic_and_sanitized(tmp_path):
    path = make_profile_trace_path(
        output_dir=tmp_path,
        prefix="pymllm_profile",
        run_name="qwen3/vl w8a8",
        setting=BenchSetting(batch_size=1, input_len=256, output_len=8),
        stage="decode",
    )

    assert path.parent == tmp_path
    assert (
        path.name
        == "pymllm_profile_qwen3_vl_w8a8_bs1_in256_out8_decode.trace.json.gz"
    )

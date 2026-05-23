from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from pymllm.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    _compute_cu_seqlens_from_grid,
    _run_with_synchronized_wall_timing,
)


class _AddLayer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, positions, hidden_states, forward_batch, **kwargs):
        del positions, forward_batch, kwargs
        return hidden_states + self.value


class _CarryLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch, **kwargs):
        del positions, forward_batch, kwargs
        return hidden_states + 10.0, hidden_states + 100.0


class _TensorLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch):
        del positions, forward_batch
        return hidden_states * 2.0


class _FinalNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_residual = None

    def forward(self, hidden_states, residual=None):
        self.seen_residual = residual
        if residual is None:
            return hidden_states
        return hidden_states + residual


class _Mode:
    def is_extend(self) -> bool:
        return True

    def is_decode(self) -> bool:
        return False


class _FakeVisual(nn.Module):
    def forward(self, pixel_values, grid_thw):
        del pixel_values, grid_thw
        return torch.ones((1, 2), dtype=torch.float32)


class _FakeTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        input_deepstack_embeds=None,
    ):
        del positions, forward_batch, input_deepstack_embeds
        if input_embeds is not None:
            return input_embeds
        return self.embed_tokens(input_ids)


def _make_vl_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        rope_scaling={"mrope_section": [1, 1, 0], "mrope_interleaved": True},
        max_position_embeddings=32,
        vocab_size=8,
    )
    vision_config = SimpleNamespace(
        depth=0,
        hidden_size=2,
        intermediate_size=4,
        num_heads=1,
        in_channels=3,
        patch_size=1,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=2,
        num_position_embeddings=4,
        deepstack_visual_indexes=[],
    )
    return SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=5,
        video_token_id=6,
        vision_start_token_id=4,
        tie_word_embeddings=False,
    )


def test_text_model_adds_deepstack_after_decoder_layer():
    model = Qwen3VLTextModel(
        vocab_size=8,
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    model.layers = nn.ModuleList([_AddLayer(10.0)])
    model.norm = nn.Identity()

    input_embeds = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    input_deepstack_embeds = torch.tensor(
        [[0.5, 1.5], [2.5, 3.5]],
        dtype=torch.float32,
    )

    hidden_states = model(
        input_ids=torch.tensor([0, 1], dtype=torch.int64),
        positions=torch.zeros((3, 2), dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=input_embeds,
        input_deepstack_embeds=input_deepstack_embeds,
    )

    torch.testing.assert_close(
        hidden_states,
        input_embeds + 10.0 + input_deepstack_embeds,
    )


def test_text_model_deepstack_resets_residual_carry_before_injection():
    model = Qwen3VLTextModel(
        vocab_size=8,
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    final_norm = _FinalNorm()
    model.layers = nn.ModuleList([_CarryLayer()])
    model.norm = final_norm

    input_embeds = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    input_deepstack_embeds = torch.tensor(
        [[0.5, 1.5], [2.5, 3.5]],
        dtype=torch.float32,
    )

    hidden_states = model(
        input_ids=torch.tensor([0, 1], dtype=torch.int64),
        positions=torch.zeros((3, 2), dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=input_embeds,
        input_deepstack_embeds=input_deepstack_embeds,
    )

    assert final_norm.seen_residual is None
    torch.testing.assert_close(
        hidden_states,
        input_embeds + 10.0 + input_embeds + 100.0 + input_deepstack_embeds,
    )


def test_text_model_materializes_residual_before_tensor_returning_layer():
    model = Qwen3VLTextModel(
        vocab_size=8,
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    model.layers = nn.ModuleList([_CarryLayer(), _TensorLayer()])
    model.norm = nn.Identity()

    input_embeds = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )

    hidden_states = model(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        positions=torch.zeros((3, 3), dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=input_embeds,
    )

    torch.testing.assert_close(
        hidden_states,
        (input_embeds + 10.0 + input_embeds + 100.0) * 2.0,
    )


def test_forward_rejects_mismatched_image_token_and_feature_counts():
    model = Qwen3VLForConditionalGeneration(_make_vl_config())
    model.visual = _FakeVisual()

    forward_batch = SimpleNamespace(
        forward_mode=_Mode(),
        batch_size=1,
        extend_start_loc=torch.tensor([0], dtype=torch.int64),
        extend_seq_lens=torch.tensor([5], dtype=torch.int64),
        pixel_values=torch.zeros((1, 3), dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int64),
    )

    with pytest.raises(
        ValueError,
        match="Image features and image tokens do not match",
    ):
        model(
            input_ids=torch.tensor([1, 4, 5, 5, 2], dtype=torch.int64),
            positions=torch.arange(5, dtype=torch.int64),
            forward_batch=forward_batch,
        )


def test_forward_records_vit_prefill_tps_when_benchmark_timing_enabled():
    model = Qwen3VLForConditionalGeneration(_make_vl_config())
    model.visual = _FakeVisual()
    model.model = _FakeTextModel()

    forward_batch = SimpleNamespace(
        forward_mode=_Mode(),
        batch_size=1,
        extend_start_loc=torch.tensor([0], dtype=torch.int64),
        extend_seq_lens=torch.tensor([4], dtype=torch.int64),
        pixel_values=torch.zeros((1, 3), dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        benchmark_vision_timing=True,
    )

    model(
        input_ids=torch.tensor([1, 4, 5, 2], dtype=torch.int64),
        positions=torch.arange(4, dtype=torch.int64),
        forward_batch=forward_batch,
    )

    assert forward_batch.vit_prefill_ms is not None
    assert forward_batch.vit_prefill_ms >= 0.0
    assert forward_batch.vit_prefill_tokens == 1
    assert forward_batch.vit_prefill_tps >= 0.0


def test_vision_timing_includes_host_side_work_when_benchmark_enabled():
    def host_heavy_fn():
        time.sleep(0.02)
        return torch.tensor([1.0])

    result, elapsed_ms = _run_with_synchronized_wall_timing(
        host_heavy_fn,
        device=torch.device("cpu"),
        enabled=True,
    )

    torch.testing.assert_close(result, torch.tensor([1.0]))
    assert elapsed_ms >= 15.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_vision_timing_includes_host_side_work_when_benchmark_enabled():
    device = torch.device("cuda")

    def host_heavy_cuda_fn():
        time.sleep(0.02)
        return torch.ones((1,), device=device)

    result, elapsed_ms = _run_with_synchronized_wall_timing(
        host_heavy_cuda_fn,
        device=device,
        enabled=True,
    )

    assert result.device.type == "cuda"
    assert elapsed_ms >= 15.0


def test_cuda_vision_timing_uses_wall_clock_not_event_elapsed(monkeypatch):
    class _FakeCudaEvent:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def record(self):
            pass

        def elapsed_time(self, other):
            del other
            return 0.0

    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def host_heavy_fn():
        time.sleep(0.02)
        return torch.tensor([1.0])

    result, elapsed_ms = _run_with_synchronized_wall_timing(
        host_heavy_fn,
        device=torch.device("cuda"),
        enabled=True,
    )

    torch.testing.assert_close(result, torch.tensor([1.0]))
    assert elapsed_ms >= 15.0


def test_vision_interpolation_indices_match_sglang_hf():
    model = Qwen3VLVisionModel(
        depth=0,
        hidden_size=2,
        intermediate_size=4,
        num_heads=1,
        in_channels=3,
        patch_size=1,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )

    np.testing.assert_allclose(
        model._get_interpolation_indices(3),
        np.linspace(0, 3, 3, dtype=np.float32),
    )


def test_vision_cu_seqlens_expands_temporal_frames_like_sglang_hf():
    cu_seqlens = _compute_cu_seqlens_from_grid(
        torch.tensor([[2, 3, 5], [1, 2, 2]], dtype=torch.int64)
    )

    torch.testing.assert_close(
        cu_seqlens,
        torch.tensor([0, 15, 30, 34], dtype=torch.int32),
    )

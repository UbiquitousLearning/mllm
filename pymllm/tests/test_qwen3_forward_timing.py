from __future__ import annotations

from types import SimpleNamespace

import torch

from pymllm.executor.model_runner import LogitsProcessorOutput
from pymllm.models.qwen3 import Qwen3ForCausalLM


class _Mode:
    def __init__(self, *, is_extend: bool, is_decode: bool):
        self._is_extend = is_extend
        self._is_decode = is_decode

    def is_extend(self) -> bool:
        return self._is_extend

    def is_decode(self) -> bool:
        return self._is_decode


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        attention_bias=False,
        vocab_size=32,
        tie_word_embeddings=False,
        hidden_act="silu",
    )


def test_forward_extend_sets_prefill_timing_and_prunes_hidden_states(monkeypatch):
    cfg = _make_config()
    model = Qwen3ForCausalLM(cfg)

    def fake_forward(input_ids, positions, forward_batch, input_embeds=None):
        del positions, forward_batch, input_embeds
        return torch.ones((input_ids.shape[0], cfg.hidden_size), dtype=torch.float32)

    monkeypatch.setattr(model.model, "forward", fake_forward)

    fb = SimpleNamespace(
        forward_mode=_Mode(is_extend=True, is_decode=False),
        extend_start_loc=torch.tensor([0, 3], dtype=torch.int64),
        extend_seq_lens=torch.tensor([3, 2], dtype=torch.int64),
        llm_prefill_ms=None,
        llm_decode_ms=None,
    )

    out = model.forward(
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        forward_batch=fb,
    )

    assert isinstance(out, LogitsProcessorOutput)
    assert out.next_token_logits.shape == (2, cfg.vocab_size)
    assert fb.llm_prefill_ms is not None
    assert fb.llm_prefill_ms >= 0.0
    assert fb.llm_decode_ms is None


def test_forward_decode_sets_decode_timing(monkeypatch):
    cfg = _make_config()
    model = Qwen3ForCausalLM(cfg)

    def fake_forward(input_ids, positions, forward_batch, input_embeds=None):
        del positions, forward_batch, input_embeds
        return torch.ones((input_ids.shape[0], cfg.hidden_size), dtype=torch.float32)

    monkeypatch.setattr(model.model, "forward", fake_forward)

    fb = SimpleNamespace(
        forward_mode=_Mode(is_extend=False, is_decode=True),
        llm_prefill_ms=None,
        llm_decode_ms=None,
    )

    out = model.forward(
        input_ids=torch.tensor([7, 8], dtype=torch.int64),
        positions=torch.tensor([11, 12], dtype=torch.int64),
        forward_batch=fb,
    )

    assert isinstance(out, LogitsProcessorOutput)
    assert out.next_token_logits.shape == (2, cfg.vocab_size)
    assert fb.llm_prefill_ms is None
    assert fb.llm_decode_ms is not None
    assert fb.llm_decode_ms >= 0.0

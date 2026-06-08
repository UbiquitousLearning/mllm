from __future__ import annotations

from types import SimpleNamespace

import torch

from pymllm.executor.model_runner import LogitsProcessorOutput, ModelRunner


def test_sample_uses_cpu_greedy_flag_without_tensor_reduction(monkeypatch):
    runner = SimpleNamespace(device="cpu")
    logits_output = LogitsProcessorOutput(
        next_token_logits=torch.tensor([[1.0, 3.0], [5.0, 2.0]])
    )
    forward_batch = SimpleNamespace(batch_size=2)

    def fail_count_nonzero(*args, **kwargs):
        raise AssertionError("greedy sampling should not reduce temperature tensors")

    monkeypatch.setattr(torch, "count_nonzero", fail_count_nonzero)

    next_token_ids = ModelRunner.sample(
        runner,
        logits_output,
        forward_batch,
        temperatures=torch.zeros((2,), dtype=torch.float32),
        is_all_greedy=True,
    )

    assert next_token_ids.tolist() == [1, 0]
    assert next_token_ids.dtype == torch.int32

"""CUDA-graph accelerated forward pass for decode steps.

Captures CUDA graphs for a set of discrete batch sizes so that the decode
forward pass can be replayed without CPU-side kernel-launch overhead.

Simplified from sglang's ``CudaGraphRunner`` for pymllm's single-GPU
architecture.  Handles:

* Pre-allocated input buffers (avoids per-step allocations)
* CUDA-graph capture for each batch size
* Optional ``torch.compile`` integration
* Graph replay with padding to the nearest captured batch size

Typical lifecycle::

    runner = CudaGraphRunner(model_runner)   # captures all batch sizes

    # --- inside the inference loop ---
    if runner.can_run(forward_batch):
        logits_output = runner.replay(forward_batch)
    else:
        logits_output = model_runner.forward(forward_batch)

Integration with :class:`~pymllm.executor.model_runner.ModelRunner`
-------------------------------------------------------------------
The ``ModelRunner`` owns the ``CudaGraphRunner`` and delegates decode
batches to it when the batch size is within the captured range.  The
``CudaGraphRunner`` calls ``attn_backend.init_forward_metadata_*_cuda_graph``
directly (bypassing the normal ``init_forward_metadata`` path) so that
FlashInfer's per-batch planning is recorded inside the graph.
"""

from __future__ import annotations

import bisect
import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch

from pymllm.engine.forward_batch import ForwardBatch, ForwardMode
from pymllm.executor.model_runner import LogitsProcessorOutput

if TYPE_CHECKING:
    from pymllm.executor.model_runner import ModelRunner
    from pymllm.layers.attention.attention_backend import AttentionBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global CUDA-graph memory pool (shared across all CudaGraphRunner instances)
# ---------------------------------------------------------------------------

_global_graph_memory_pool: Optional[tuple] = None


def get_global_graph_memory_pool() -> Optional[tuple]:
    """Return the shared CUDA graph memory pool handle."""
    return _global_graph_memory_pool


def set_global_graph_memory_pool(pool: tuple) -> None:
    """Set the shared CUDA graph memory pool handle."""
    global _global_graph_memory_pool
    _global_graph_memory_pool = pool


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

# Flag indicating whether we are currently capturing a CUDA graph.
_is_capture_mode: bool = False


def is_capture_mode() -> bool:
    """Return ``True`` if a CUDA-graph capture is in progress."""
    return _is_capture_mode


@contextmanager
def model_capture_mode():
    """Context manager that sets the global capture-mode flag."""
    global _is_capture_mode
    _is_capture_mode = True
    try:
        yield
    finally:
        _is_capture_mode = False


@contextmanager
def freeze_gc():
    """Freeze the garbage collector during CUDA-graph capture.

    GC activity during capture can interfere with the recorded stream
    ordering.  This context manager collects garbage before capture,
    freezes all surviving objects, and unfreezes + re-collects afterwards.
    """
    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
        gc.collect()


# ---------------------------------------------------------------------------
# Pre-allocated input buffers
# ---------------------------------------------------------------------------


@dataclass
class _InputBuffers:
    """Pre-allocated GPU tensors used as CUDA-graph inputs.

    During graph capture these buffers are used as-is.  During replay the
    real batch data is copied into the first ``batch_size`` rows while the
    remaining padding rows retain their fill values.
    """

    input_ids: torch.Tensor  # [max_bs] int64
    req_pool_indices: torch.Tensor  # [max_bs] int32
    seq_lens: torch.Tensor  # [max_bs] int32
    seq_lens_cpu: torch.Tensor  # [max_bs] int32 (CPU)
    out_cache_loc: torch.Tensor  # [max_bs] int64
    positions: torch.Tensor  # [max_bs] int64
    mrope_position_deltas: torch.Tensor  # [max_bs] int64

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        seq_len_fill_value: int,
    ) -> "_InputBuffers":
        """Allocate all buffers for the given maximum batch size."""
        with torch.device(device):
            input_ids = torch.zeros((max_bs,), dtype=torch.int64)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((max_bs,), dtype=torch.int64)
            positions = torch.zeros((max_bs,), dtype=torch.int64)
            mrope_position_deltas = torch.zeros((max_bs,), dtype=torch.int64)

        # seq_lens_cpu must be a real CPU tensor.
        seq_lens_cpu = torch.full(
            (max_bs,),
            seq_len_fill_value,
            dtype=torch.int32,
            device="cpu",
        )

        return cls(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_position_deltas=mrope_position_deltas,
        )

    def populate(
        self,
        forward_batch: ForwardBatch,
        padded_bs: int,
        seq_len_fill_value: int,
    ) -> None:
        """Copy real batch data into the pre-allocated buffers.

        Any padding slots (``[real_bs : padded_bs]``) are filled with safe
        defaults so that the captured graph does not access invalid memory.
        """
        real_bs = forward_batch.batch_size

        # Reset padding slots when the padded size exceeds the real size.
        if padded_bs != real_bs:
            self.seq_lens.fill_(seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.mrope_position_deltas.zero_()

        self.input_ids[:real_bs].copy_(forward_batch.input_ids)
        self.req_pool_indices[:real_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:real_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:real_bs].copy_(forward_batch.out_cache_loc)
        self.positions[:real_bs].copy_(forward_batch.positions)

        # Copy M-RoPE position deltas (used by Qwen3-VL for multimodal).
        if forward_batch.mrope_position_deltas is not None:
            self.mrope_position_deltas[:real_bs].copy_(
                forward_batch.mrope_position_deltas
            )
        else:
            self.mrope_position_deltas[:real_bs].zero_()

        if forward_batch.seq_lens_cpu is not None:
            if padded_bs != real_bs:
                self.seq_lens_cpu.fill_(seq_len_fill_value)
            self.seq_lens_cpu[:real_bs].copy_(forward_batch.seq_lens_cpu)


# ---------------------------------------------------------------------------
# Batch-size schedule
# ---------------------------------------------------------------------------


def _default_capture_batch_sizes(max_bs: int) -> List[int]:
    """Return a list of batch sizes to capture.

    Uses the same schedule as sglang (non-speculative)::

        [1, 2, 4, 8, 12, 16, 24, 32, 40, …, 256, 272, 288, …, 512, 544, …]

    Capped at *max_bs*.
    """
    bs_list = (
        [1, 2, 4, 8, 12]
        + list(range(16, 257, 8))
        + list(range(272, 512, 16))
        + list(range(512, max_bs + 1, 32))
    )
    bs_list = sorted(set(bs for bs in bs_list if bs <= max_bs))
    if not bs_list:
        bs_list = [1]
    return bs_list


# ---------------------------------------------------------------------------
# CudaGraphRunner
# ---------------------------------------------------------------------------


class CudaGraphRunner:
    """Captures and replays CUDA graphs for decode-step forward passes.

    This class is the pymllm equivalent of sglang's ``CudaGraphRunner``,
    stripped of distributed, speculative-decoding, LoRA, mamba, TBO, and
    piecewise-graph complexities.

    Parameters
    ----------
    model_runner
        The owning :class:`~pymllm.executor.model_runner.ModelRunner`.
        Must have been fully initialised before the ``CudaGraphRunner``
        is constructed.
    """

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.device = model_runner.device

        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: Dict[int, LogitsProcessorOutput] = {}

        self.enable_torch_compile: bool = (
            model_runner.server_config.enable_torch_compile
        )
        self.torch_compile_max_bs: int = model_runner.server_config.torch_compile_max_bs

        # -----------------------------------------------------------
        # Batch-size schedule
        # -----------------------------------------------------------
        max_bs = model_runner.max_running_requests
        self.capture_bs: List[int] = _default_capture_batch_sizes(max_bs)
        self.compile_bs: List[int] = (
            [bs for bs in self.capture_bs if bs <= self.torch_compile_max_bs]
            if self.enable_torch_compile
            else []
        )
        self.max_bs: int = max(self.capture_bs)

        logger.info("CUDA graph capture batch sizes: %s", self.capture_bs)

        # -----------------------------------------------------------
        # Attention-backend CUDA-graph state
        # -----------------------------------------------------------
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs, self.max_bs)

        # Fill value for padded seq_lens so attention kernels don't div-by-0.
        self.seq_len_fill_value: int = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        # -----------------------------------------------------------
        # Pre-allocated input buffers
        # -----------------------------------------------------------
        self.buffers: _InputBuffers = _InputBuffers.create(
            device=torch.device(self.device),
            max_bs=self.max_bs,
            seq_len_fill_value=self.seq_len_fill_value,
        )

        # -----------------------------------------------------------
        # Optional torch.compile config
        # -----------------------------------------------------------
        if self.enable_torch_compile:
            _set_torch_compile_config()

        # -----------------------------------------------------------
        # Capture all batch sizes
        # -----------------------------------------------------------
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as exc:
            raise RuntimeError(
                f"CUDA graph capture failed: {exc}\n"
                "Possible fixes:\n"
                "  1. Reduce --server.mem_fraction_static (e.g. 0.7)\n"
                "  2. Reduce --server.max_running_requests\n"
                "  3. Disable CUDA graph with --server.disable_cuda_graph\n"
            ) from exc

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Return ``True`` if the batch can be run via CUDA graph replay.

        The batch must be a decode (or idle) batch whose size does not
        exceed the largest captured batch size.
        """
        return (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.batch_size <= self.max_bs
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> None:
        """Capture CUDA graphs for every batch size in ``capture_bs``.

        Iterates in reverse order (largest first) so that the GPU memory
        pool allocated for the largest graph is reused by smaller ones.
        """
        tic = time.perf_counter()
        before_mem = _get_avail_mem(self.device)
        logger.info("CUDA graph capture begin. avail mem=%.2f GB", before_mem)

        with freeze_gc():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for bs in reversed(self.capture_bs):
                    forward_fn = self._get_forward_fn(bs)
                    graph, output = self._capture_one_batch_size(bs, forward_fn, stream)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output

        after_mem = _get_avail_mem(self.device)
        logger.info(
            "CUDA graph capture end. elapsed=%.2f s, mem usage=%.2f GB, "
            "avail mem=%.2f GB",
            time.perf_counter() - tic,
            before_mem - after_mem,
            after_mem,
        )

    def _get_forward_fn(self, bs: int) -> Callable:
        """Return the forward callable for the given batch size.

        When ``torch.compile`` is enabled and *bs* is within the compile
        threshold, the model's forward method is wrapped with
        ``torch.compile``.
        """
        model_forward = self.model_runner.model.forward
        if self.enable_torch_compile and bs in self.compile_bs:
            return torch.compile(
                torch.no_grad()(model_forward),
                mode="max-autotune-no-cudagraphs",
            )
        return model_forward

    def _capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        stream: torch.cuda.Stream,
    ) -> tuple:
        """Capture a single CUDA graph for batch size *bs*.

        Steps:
        1. Build a ``ForwardBatch`` from the pre-allocated buffers.
        2. Tell the attention backend to plan for CUDA-graph capture.
        3. Run the forward pass twice for warmup.
        4. Capture the third run into a ``CUDAGraph``.

        Returns ``(graph, output_buffers)``.
        """
        buffers = self.buffers

        # Slice pre-allocated buffers to the capture size.
        input_ids = buffers.input_ids[:bs]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:bs]
        positions = buffers.positions[:bs]
        mrope_position_deltas = buffers.mrope_position_deltas[:bs]

        # Build ForwardBatch (DECODE mode).
        # mrope_position_deltas is set to the static buffer (initially zeros)
        # so that the graph captures the ``positions + deltas`` path.  During
        # replay the buffer is updated with real delta values.
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            return_logprob=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            mrope_position_deltas=mrope_position_deltas,
        )

        # Tell the attention backend to set up CUDA-graph-aware metadata.
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
        )

        # The single forward-pass function to be captured.
        def run_once():
            return forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
            )

        # Warmup (2 eager runs to stabilise cudnn / autotuner / etc.).
        for _ in range(2):
            torch.cuda.synchronize()
            run_once()

        # ----- Capture -----
        global _global_graph_memory_pool
        if _global_graph_memory_pool is None:
            _global_graph_memory_pool = torch.cuda.graph_pool_handle()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=_global_graph_memory_pool,
            stream=stream,
        ):
            output = run_once()

        return graph, output

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Replay a captured CUDA graph for the given decode batch.

        The batch is padded to the nearest captured size, inputs are copied
        into the pre-allocated buffers, the graph is replayed, and the
        output is sliced back to the real batch size.

        Parameters
        ----------
        forward_batch
            The decode batch from the scheduler.

        Returns
        -------
        LogitsProcessorOutput
            The logits for the real (un-padded) sequences.
        """
        real_bs = forward_batch.batch_size

        # Find the smallest captured bs >= real_bs.
        idx = bisect.bisect_left(self.capture_bs, real_bs)
        padded_bs = self.capture_bs[idx]

        # Copy real data into the static buffers.
        self.buffers.populate(
            forward_batch,
            padded_bs=padded_bs,
            seq_len_fill_value=self.seq_len_fill_value,
        )

        # Update the attention backend for replay.
        seq_lens_sum = (
            forward_batch.seq_lens_sum + (padded_bs - real_bs) * self.seq_len_fill_value
        )
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=padded_bs,
            req_pool_indices=self.buffers.req_pool_indices[:padded_bs],
            seq_lens=self.buffers.seq_lens[:padded_bs],
            seq_lens_sum=seq_lens_sum,
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=self.buffers.seq_lens_cpu[:padded_bs],
        )

        # Replay the graph.
        self.graphs[padded_bs].replay()

        # Retrieve output and slice to real batch size.
        output = self.output_buffers[padded_bs]

        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[:real_bs],
                hidden_states=(
                    output.hidden_states[:real_bs]
                    if output.hidden_states is not None
                    else None
                ),
            )
        elif isinstance(output, torch.Tensor):
            # Raw tensor output: assume [padded_bs, vocab_size].
            return LogitsProcessorOutput(
                next_token_logits=output[:real_bs],
            )
        else:
            # HuggingFace-style output with .logits attribute.
            if hasattr(output, "logits"):
                logits = output.logits
                if logits.dim() == 3:
                    return LogitsProcessorOutput(
                        next_token_logits=logits[:real_bs, -1, :],
                    )
                return LogitsProcessorOutput(
                    next_token_logits=logits[:real_bs],
                )
            raise TypeError(f"Unexpected CUDA graph output type: {type(output)}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all captured CUDA graphs and associated buffers."""
        for graph in self.graphs.values():
            del graph
        self.graphs.clear()
        self.output_buffers.clear()
        logger.info("CudaGraphRunner shutdown complete.")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_avail_mem(device: str) -> float:
    """Return available GPU memory in GB."""
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1 << 30)


def _set_torch_compile_config() -> None:
    """Set dynamo / inductor configs for optimal CUDA-graph + compile."""
    try:
        import torch._dynamo.config
        import torch._inductor.config

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024
    except ImportError:
        logger.warning("torch._dynamo / torch._inductor not available.")

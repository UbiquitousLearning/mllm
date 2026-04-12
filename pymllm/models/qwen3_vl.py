# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Adapted for pymllm
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3-VL model for pymllm.

Adapted from sglang's Qwen3-VL implementation for pymllm's single-GPU
inference architecture.  Uses pymllm layers (RadixAttention, RMSNorm, MLP)
and conforms to the pymllm forward interface::

    model.forward(input_ids, positions, forward_batch)

Designed for a single accelerator card — no tensor / pipeline parallelism.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymllm.layers import RMSNorm, apply_mrope
from pymllm.layers.attention.radix_attention import RadixAttention
from pymllm.layers.linear import Linear
from pymllm.layers.mlp import MLP

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------


class Qwen3VisionMLP(nn.Module):
    """MLP block for the vision encoder."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_act: str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        if hidden_act == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        elif hidden_act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D convolution patch embedding for video/image patchification."""

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3VisionAttention(nn.Module):
    """Multi-head self-attention for the vision encoder (no KV cache)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with variable-length sequences via cu_seqlens.

        Args:
            x: [total_tokens, embed_dim]
            cu_seqlens: [num_seqs + 1] cumulative sequence lengths
            rotary_pos_emb_cos: [total_tokens, rotary_dim]
            rotary_pos_emb_sin: [total_tokens, rotary_dim]
        """
        seq_len = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim).unbind(dim=1)

        # Apply rotary position embedding.
        # cos/sin are [total_tokens, head_dim // 2].
        # VisionAttention: double them to full head_dim and apply RoPE to
        # all head dimensions (the rotation pairs (q[i], q[i + head_dim//2])).
        cos = rotary_pos_emb_cos
        sin = rotary_pos_emb_sin
        if cos.shape[-1] * 2 == self.head_dim:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        cos = cos.unsqueeze(1).to(dtype=q.dtype, device=q.device)  # [seq, 1, head_dim]
        sin = sin.unsqueeze(1).to(dtype=q.dtype, device=q.device)  # [seq, 1, head_dim]

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Scaled dot-product attention per variable-length sequence
        output = torch.empty_like(q)
        num_seqs = cu_seqlens.shape[0] - 1
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            qi = q[start:end].transpose(0, 1).unsqueeze(0)  # [1, heads, seq, dim]
            ki = k[start:end].transpose(0, 1).unsqueeze(0)
            vi = v[start:end].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi)
            output[start:end] = oi.squeeze(0).transpose(0, 1)

        output = output.reshape(seq_len, self.embed_dim)
        return self.out_proj(output)


class Qwen3VisionBlock(nn.Module):
    """Single vision transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act: str = "silu",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Qwen3VisionAttention(embed_dim=dim, num_heads=num_heads)
        self.mlp = Qwen3VisionMLP(
            dim, intermediate_dim, hidden_act=hidden_act, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nn.Module):
    """Merges spatial patches to reduce sequence length.

    Groups ``spatial_merge_size ** 2`` consecutive patch tokens and projects
    them to the language model hidden dimension.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else context_dim, eps=norm_eps
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.act_fn(self.linear_fc1(x))
        return self.linear_fc2(x)


class Qwen3VLVisionModel(nn.Module):
    """Complete vision encoder for Qwen3-VL.

    Produces patch embeddings from raw pixel values, applies a stack of
    vision transformer blocks with 3D rotary embeddings, then merges
    spatial patches.  Supports "deep stack" where intermediate layer
    outputs are captured and concatenated to the final output.
    """

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Optional[List[int]] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_position_embeddings = num_position_embeddings
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.deepstack_visual_indexes = deepstack_visual_indexes
        # Total output dim = out_hidden_size * (1 main + N deepstack mergers)
        self.out_hidden_size = out_hidden_size * (1 + len(deepstack_visual_indexes))

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)

        head_dim = hidden_size // num_heads
        self._init_rope_cache(head_dim)

        self.blocks = nn.ModuleList(
            [
                Qwen3VisionBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_size,
                    hidden_act=hidden_act,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            dim=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            norm_eps=norm_eps,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    dim=out_hidden_size,
                    context_dim=hidden_size,
                    spatial_merge_size=spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_eps=norm_eps,
                )
                for _ in range(len(deepstack_visual_indexes))
            ]
        )

    def _init_rope_cache(self, head_dim: int, max_grid_size: int = 8192):
        """Precompute cos/sin cache for 2D rotary embeddings."""
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(max_grid_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cache", torch.sin(freqs), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    # -- Rotary position embedding helpers --

    @staticmethod
    def _rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        """Compute 2D rotary position IDs for a grid of *h* x *w* patches.

        The patches are re-ordered to group ``spatial_merge_size ** 2``
        neighbours together (matching the merger's token order).

        Returns tensor of shape ``[h*w, 2]`` with ``(height_pos, width_pos)``.
        """
        merge = spatial_merge_size
        h_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        w_ids = torch.arange(w).unsqueeze(0).expand(h, -1)

        h_ids = h_ids.reshape(h // merge, merge, w // merge, merge)
        w_ids = w_ids.reshape(h // merge, merge, w // merge, merge)

        h_ids = h_ids.permute(0, 2, 1, 3).flatten()
        w_ids = w_ids.permute(0, 2, 1, 3).flatten()

        return torch.stack([h_ids, w_ids], dim=-1)

    def rot_pos_emb(
        self, grid_thw: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary pos-emb cos/sin for all images/videos in the batch."""
        pos_ids = []
        for t, h, w in grid_thw:
            base = self._rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        cos_combined = self.cos_cache[pos_ids].flatten(1)
        sin_combined = self.sin_cache[pos_ids].flatten(1)
        return cos_combined, sin_combined

    # -- Position embedding interpolation --

    def _get_interpolation_indices(self, dim_size: int) -> np.ndarray:
        indices = (np.arange(dim_size, dtype=np.float32) + 0.5) * (
            self.num_grid_per_side / dim_size
        ) - 0.5
        return np.clip(indices, 0, self.num_grid_per_side - 1)

    def _calculate_indices_and_weights(
        self, h_idxs: np.ndarray, w_idxs: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute bilinear interpolation indices and weights."""
        side = self.num_grid_per_side
        h_f = np.floor(h_idxs).astype(np.int64)
        h_c = np.clip(h_f + 1, 0, side - 1)
        dh = h_idxs - h_f
        w_f = np.floor(w_idxs).astype(np.int64)
        w_c = np.clip(w_f + 1, 0, side - 1)
        dw = w_idxs - w_f

        indices = [
            (h_f[:, None] * side + w_f).flatten(),
            (h_f[:, None] * side + w_c).flatten(),
            (h_c[:, None] * side + w_f).flatten(),
            (h_c[:, None] * side + w_c).flatten(),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)).flatten(),
            ((1 - dh)[:, None] * dw).flatten(),
            (dh[:, None] * (1 - dw)).flatten(),
            (dh[:, None] * dw).flatten(),
        ]
        return indices, weights

    def _get_position_embedding(
        self,
        patch_pos_embeds: List[torch.Tensor],
        grid_ts: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ) -> torch.Tensor:
        """Tile and reorganize position embeddings to align with the merged token order."""
        result_parts = []
        merge = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result_parts.append(pos_embed)
        return torch.cat(result_parts, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Interpolate position embeddings via bilinear interpolation."""
        grid_thw_cpu = grid_thw.cpu().numpy()
        temporal_dims = grid_thw_cpu[:, 0].tolist()
        height_dims = grid_thw_cpu[:, 1].tolist()
        width_dims = grid_thw_cpu[:, 2].tolist()

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        patches_size = [h * w for h, w in zip(height_dims, width_dims)]
        total_patches = sum(patches_size)
        all_indices_np = np.zeros((4, total_patches), dtype=np.int64)
        all_weights_np = np.zeros((4, total_patches), dtype=np.float32)

        current_idx = 0
        for _t, h, w in zip(temporal_dims, height_dims, width_dims):
            h_idxs = self._get_interpolation_indices(h)
            w_idxs = self._get_interpolation_indices(w)
            indices, weights = self._calculate_indices_and_weights(h_idxs, w_idxs)
            end_idx = current_idx + h * w
            for i in range(4):
                all_indices_np[i, current_idx:end_idx] = indices[i]
                all_weights_np[i, current_idx:end_idx] = weights[i]
            current_idx = end_idx

        idx_tensor = torch.from_numpy(all_indices_np).to(device)
        weight_tensor = torch.from_numpy(all_weights_np).to(dtype=dtype, device=device)

        pos_embeds = self.pos_embed(idx_tensor.view(-1))
        pos_embeds = pos_embeds.view(4, total_patches, -1)
        patch_pos_embeds = (pos_embeds * weight_tensor.unsqueeze(-1)).sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split(patches_size)
        return self._get_position_embedding(
            list(patch_pos_embeds), temporal_dims, height_dims, width_dims
        )

    # -- Forward --

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder.

        Args:
            x: Pixel values, shape ``[total_patches, patch_dim]``.
            grid_thw: Grid dimensions ``[num_images, 3]`` with ``(T, H, W)``.

        Returns:
            Vision features of shape
            ``[num_merged_tokens, out_hidden_size * (1 + num_deepstack)]``.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds

        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = _compute_cu_seqlens_from_grid(grid_thw)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        deepstack_features = []
        ds_idx = 0

        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)

            if layer_num in self.deepstack_visual_indexes:
                # x is [total_tokens, hidden].  The merger expects the last
                # dim to be context_dim so it can group spatial_merge_size^2
                # tokens; reshape to [total_tokens, 1, hidden] so that the
                # `.view(-1, hidden_size)` inside the merger collapses the
                # spatial merge correctly.
                ds_feat = self.deepstack_merger_list[ds_idx](x.unsqueeze(1))
                deepstack_features.append(ds_feat)
                ds_idx += 1

        x = self.merger(x.unsqueeze(1))

        # Concatenate main + deepstack features along the feature dimension.
        # Result: [num_merged_tokens, out_hidden_size * (1 + num_deepstack)]
        hidden_states = torch.cat([x] + deepstack_features, dim=-1)
        return hidden_states


def _compute_cu_seqlens_from_grid(grid_thw: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths from grid dimensions."""
    grid_np = grid_thw.cpu().numpy()
    seq_lens = (grid_np[:, 0] * grid_np[:, 1] * grid_np[:, 2]).astype(np.int32)
    cu_seqlens = np.concatenate([[0], np.cumsum(seq_lens)])
    return torch.tensor(cu_seqlens, dtype=torch.int32)


def _build_cos_sin_cache(
    head_dim: int,
    rope_theta: float,
    max_pos: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a [max_pos, head_dim] cos/sin cache for M-RoPE.

    Layout: first ``head_dim // 2`` columns are cos values, second half are sin.
    Each row corresponds to one position index.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim // 2]
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(dtype)


def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, int]:
    """Compute M-RoPE 3-D position IDs for one sequence.

    For text tokens all three (temporal, height, width) indices are equal to
    the sequential counter.  For image tokens the indices follow the spatial
    grid ``(t, h, w)``.

    Args:
        input_ids: Token IDs for one sequence, shape ``[T]``.
        image_grid_thw: Grid dimensions for every image in the sequence,
            shape ``[num_images, 3]``.  ``None`` when there are no images.
        image_token_id: Token ID used as placeholder for image patches.
        vision_start_token_id: Token ID that precedes each image block.
        spatial_merge_size: Number of patches merged per spatial dimension
            (e.g. 2 → 2x2 merge, so llm_grid_h = H // 2).

    Returns:
        ``(position_ids, mrope_position_delta)`` where ``position_ids`` has
        shape ``[3, T]`` and ``mrope_position_delta`` is a Python ``int``
        equal to ``max_position_used + 1 - T``.
    """
    total_tokens = input_ids.shape[0]
    device = input_ids.device
    position_ids = torch.zeros(3, total_tokens, dtype=torch.long, device=device)

    if image_grid_thw is None or image_grid_thw.shape[0] == 0:
        pos = torch.arange(total_tokens, dtype=torch.long, device=device)
        position_ids[0] = pos
        position_ids[1] = pos
        position_ids[2] = pos
        return position_ids, 0

    input_ids_cpu = input_ids.cpu().tolist()
    grid_thw_list = image_grid_thw.cpu().tolist()

    llm_pos_ids_start = 0
    image_idx = 0
    i = 0

    while i < total_tokens:
        token = input_ids_cpu[i]

        if token == vision_start_token_id and image_idx < len(grid_thw_list):
            # The vision_start token itself gets a regular sequential position.
            position_ids[:, i] = llm_pos_ids_start
            llm_pos_ids_start += 1
            i += 1

            # Compute LLM-side grid dimensions (after spatial merging).
            t_g = int(grid_thw_list[image_idx][0])
            h_g = int(grid_thw_list[image_idx][1])
            w_g = int(grid_thw_list[image_idx][2])
            llm_grid_t = t_g
            llm_grid_h = h_g // spatial_merge_size
            llm_grid_w = w_g // spatial_merge_size
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w

            # Build per-patch 3-D indices.
            t_idx = (
                torch.arange(llm_grid_t, device=device)
                .view(-1, 1, 1)
                .expand(-1, llm_grid_h, llm_grid_w)
                .flatten()
            )
            h_idx = (
                torch.arange(llm_grid_h, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_idx = (
                torch.arange(llm_grid_w, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )

            img_start = i
            img_end = i + num_image_tokens
            position_ids[0, img_start:img_end] = t_idx + llm_pos_ids_start
            position_ids[1, img_start:img_end] = h_idx + llm_pos_ids_start
            position_ids[2, img_start:img_end] = w_idx + llm_pos_ids_start

            llm_pos_ids_start += max(llm_grid_t, llm_grid_h, llm_grid_w)
            i += num_image_tokens
            image_idx += 1
        else:
            # Text token (including vision_end and all non-image tokens).
            position_ids[:, i] = llm_pos_ids_start
            llm_pos_ids_start += 1
            i += 1

    mrope_position_delta = llm_pos_ids_start - total_tokens
    return position_ids, mrope_position_delta


# ---------------------------------------------------------------------------
# Text Decoder (Language Model)
# ---------------------------------------------------------------------------


class Qwen3VLAttention(nn.Module):
    """Attention layer for the Qwen3-VL text decoder.

    Uses QK-norm (per-head RMSNorm on Q and K before RoPE) and
    :class:`RadixAttention` for KV-cached inference.  Applies
    interleaved M-RoPE with a precomputed cos/sin cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        mrope_section: Tuple[int, int, int] = (24, 20, 20),
        mrope_interleaved: bool = True,
        max_position_embeddings: int = 32768,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.mrope_section = list(mrope_section)
        self.mrope_interleaved = mrope_interleaved

        def _get_qm(suffix):
            if quant_config is None:
                return None
            return quant_config.get_quant_method(
                layer=None, prefix=f"{prefix}.{suffix}" if prefix else suffix,
            )

        # When quantized, AWQ checkpoints store q/k/v separately so we
        # cannot fuse them into a single packed-int32 parameter.
        self.use_fused_qkv = quant_config is None

        if self.use_fused_qkv:
            self.qkv_proj = Linear(
                hidden_size, self.q_size + 2 * self.kv_size, bias=False,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = Linear(
                hidden_size, self.q_size, bias=False,
                quant_method=_get_qm("q_proj"),
            )
            self.k_proj = Linear(
                hidden_size, self.kv_size, bias=False,
                quant_method=_get_qm("k_proj"),
            )
            self.v_proj = Linear(
                hidden_size, self.kv_size, bias=False,
                quant_method=_get_qm("v_proj"),
            )

        # Output projection
        self.o_proj = Linear(
            num_heads * head_dim, hidden_size, bias=False,
            quant_method=_get_qm("o_proj"),
        )

        # QK normalization
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Precomputed M-RoPE cos/sin cache: [max_pos, head_dim]
        cos_sin = _build_cos_sin_cache(
            head_dim, rope_theta, max_position_embeddings, torch.float32
        )
        self.register_buffer("cos_sin_cache", cos_sin, persistent=False)

        # Radix attention (single-GPU: heads == tp_heads)
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        if self.use_fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Per-head QK normalization
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))

        # Apply M-RoPE. positions is [3, T] for prefill (3-D) or may arrive
        # as [T] for purely text-only batches; expand to [3, T] in that case.
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        q, k = apply_mrope(
            q,
            k,
            positions,
            self.cos_sin_cache.to(q.dtype),
            self.mrope_section,
            self.mrope_interleaved,
        )

        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        # Attention with KV cache
        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class Qwen3VLDecoderLayer(nn.Module):
    """Single decoder layer for the Qwen3-VL text model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        layer_id: int,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        mrope_section: Tuple[int, int, int] = (24, 20, 20),
        mrope_interleaved: bool = True,
        max_position_embeddings: int = 32768,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Qwen3VLAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn" if prefix else "self_attn",
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation="silu",
            use_fused_gate_up_proj=True,
            use_bias_gate_up=False,
            use_bias_down=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
        deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        # Add deepstack embeddings after residual (matches HF ordering)
        if deepstack_embeds is not None:
            hidden_states = hidden_states + deepstack_embeds

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3VLTextModel(nn.Module):
    """Qwen3-VL text backbone (embedding + decoder layers + final norm)."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        mrope_section: Tuple[int, int, int] = (24, 20, 20),
        mrope_interleaved: bool = True,
        max_position_embeddings: int = 32768,
        quant_config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                Qwen3VLDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    layer_id=layer_id,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                    mrope_section=mrope_section,
                    mrope_interleaved=mrope_interleaved,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=quant_config,
                    prefix=f"model.layers.{layer_id}",
                )
                for layer_id in range(num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
        input_embeds: Optional[torch.Tensor] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        for layer_idx, layer in enumerate(self.layers):
            ds_embeds = _get_deepstack_embeds(
                layer_idx, input_deepstack_embeds, self.hidden_size
            )
            hidden_states = layer(
                positions,
                hidden_states,
                forward_batch,
                deepstack_embeds=ds_embeds,
            )

        return self.norm(hidden_states)


def _get_deepstack_embeds(
    layer_idx: int,
    input_deepstack_embeds: Optional[torch.Tensor],
    hidden_size: int,
) -> Optional[torch.Tensor]:
    """Extract deepstack embeddings for a specific decoder layer."""
    if input_deepstack_embeds is None:
        return None
    num_deepstack = input_deepstack_embeds.shape[-1] // hidden_size
    if layer_idx >= num_deepstack:
        return None
    start = hidden_size * layer_idx
    return input_deepstack_embeds[:, start : start + hidden_size]


# ---------------------------------------------------------------------------
# Full Model: Qwen3VLForConditionalGeneration
# ---------------------------------------------------------------------------


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL multimodal model for conditional generation.

    Combines a vision encoder and text decoder.  During prefill, image/video
    tokens are replaced with visual features from the vision encoder.
    During decode, the model runs only the text decoder.

    Forward interface::

        logits = model.forward(input_ids, positions, forward_batch)
    """

    def __init__(self, config, quant_config=None) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        # Vision encoder — NOT quantized
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel(
                depth=getattr(vision_config, "depth", 27),
                hidden_size=getattr(vision_config, "hidden_size", 1152),
                hidden_act=getattr(vision_config, "hidden_act", "gelu_pytorch_tanh"),
                intermediate_size=getattr(vision_config, "intermediate_size", 4304),
                num_heads=getattr(vision_config, "num_heads", 16),
                in_channels=getattr(vision_config, "in_channels", 3),
                patch_size=getattr(vision_config, "patch_size", 16),
                spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
                temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
                out_hidden_size=getattr(vision_config, "out_hidden_size", 3584),
                num_position_embeddings=getattr(
                    vision_config, "num_position_embeddings", 2304
                ),
                deepstack_visual_indexes=getattr(
                    vision_config, "deepstack_visual_indexes", [8, 16, 24]
                ),
                norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            )
        else:
            self.visual = None

        # Text decoder
        hidden_size = getattr(text_config, "hidden_size", 4096)
        vocab_size = getattr(text_config, "vocab_size", 151936)

        # M-RoPE configuration -- mrope_section lives inside rope_scaling,
        # NOT as a top-level attribute of text_config.
        rope_scaling = getattr(text_config, "rope_scaling", None) or {}
        if isinstance(rope_scaling, dict):
            mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
            mrope_interleaved = rope_scaling.get("mrope_interleaved", True)
        else:
            mrope_section = getattr(rope_scaling, "mrope_section", [24, 20, 20])
            mrope_interleaved = getattr(rope_scaling, "mrope_interleaved", True)
        max_position_embeddings = getattr(text_config, "max_position_embeddings", 32768)

        self.model = Qwen3VLTextModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=getattr(text_config, "intermediate_size", 22016),
            num_hidden_layers=getattr(text_config, "num_hidden_layers", 32),
            num_attention_heads=getattr(text_config, "num_attention_heads", 32),
            num_key_value_heads=getattr(text_config, "num_key_value_heads", 32),
            head_dim=getattr(text_config, "head_dim", 128),
            rope_theta=getattr(text_config, "rope_theta", 5_000_000.0),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            mrope_section=tuple(mrope_section),
            mrope_interleaved=bool(mrope_interleaved),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        # LM head — following sglang's pattern: always use lm_head.weight
        # for matmul in forward(), so it works whether lm_head is nn.Embedding
        # (tied) or nn.Linear (untied).
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Token IDs for multimodal
        self.image_token_id = getattr(config, "image_token_id", 151655)
        self.video_token_id = getattr(config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)

        # Spatial merge size (needed for get_rope_index)
        self.spatial_merge_size = (
            getattr(vision_config, "spatial_merge_size", 2)
            if vision_config is not None
            else 2
        )

        # Deepstack config
        if vision_config is not None:
            ds_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
            self.num_deepstack_embeddings = len(ds_indexes)
        else:
            self.num_deepstack_embeddings = 0

        self._hidden_size = hidden_size

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Run forward pass for Qwen3-VL.

        Args:
            input_ids: Flattened input token IDs, shape ``[num_tokens]``.
            positions: Position IDs, shape ``[num_tokens]`` (1-D, from model
                runner).  Overridden internally with 3-D M-RoPE positions.
            forward_batch: :class:`ForwardBatch` with attention metadata.

        Returns:
            Logits tensor of shape ``[num_tokens, vocab_size]``.
        """
        pixel_values = getattr(forward_batch, "pixel_values", None)
        image_grid_thw = getattr(forward_batch, "image_grid_thw", None)

        # ------------------------------------------------------------------
        # Build 3-D M-RoPE positions
        # ------------------------------------------------------------------
        if forward_batch.forward_mode.is_extend():
            # Prefill: compute per-sequence 3-D position IDs from input_ids
            # and image grids, then store per-request deltas for future decode.
            mrope_positions_list: List[torch.Tensor] = []
            deltas: List[int] = []
            image_idx_offset = 0

            for i in range(forward_batch.batch_size):
                start = int(forward_batch.extend_start_loc[i].item())
                length = int(forward_batch.extend_seq_lens[i].item())
                seq_ids = input_ids[start : start + length]

                # Determine how many images belong to this sequence.
                num_img = int((seq_ids == self.vision_start_token_id).sum().item())
                if image_grid_thw is not None and num_img > 0:
                    thw_seq = image_grid_thw[
                        image_idx_offset : image_idx_offset + num_img
                    ]
                    image_idx_offset += num_img
                else:
                    thw_seq = None

                pos3d, delta = get_rope_index(
                    seq_ids,
                    thw_seq,
                    self.image_token_id,
                    self.vision_start_token_id,
                    self.spatial_merge_size,
                )
                mrope_positions_list.append(pos3d)
                deltas.append(delta)

            # Concatenate across sequences: [3, total_extend_tokens]
            positions = torch.cat(mrope_positions_list, dim=1)
            forward_batch.mrope_position_deltas = torch.tensor(
                deltas, dtype=torch.int64, device=input_ids.device
            )
        else:
            # Decode: each sequence emits exactly one token.  Apply the stored
            # per-request delta so the position matches the image extent.
            stored_deltas = getattr(forward_batch, "mrope_position_deltas", None)
            if stored_deltas is not None:
                pos_1d = forward_batch.positions + stored_deltas
            else:
                pos_1d = forward_batch.positions
            positions = pos_1d.unsqueeze(0).expand(3, -1)  # [3, batch_size]

        input_embeds = None
        input_deepstack_embeds = None
        vit_prefill_ms = None
        vit_prefill_tokens = None
        llm_prefill_ms = None
        llm_decode_ms = None

        if (
            pixel_values is not None
            and image_grid_thw is not None
            and self.visual is not None
            and not forward_batch.forward_mode.is_decode()
        ):
            # Run vision encoder
            _vit_t0 = time.perf_counter()
            vision_features = (
                self.visual(pixel_values, grid_thw=image_grid_thw)
            )
            vit_prefill_ms = (time.perf_counter() - _vit_t0) * 1000.0

            # Separate main embeddings and deepstack embeddings
            if self.num_deepstack_embeddings > 0:
                vision_embeds = vision_features[:, : self._hidden_size]
                deepstack_embeds = vision_features[:, self._hidden_size :]
            else:
                vision_embeds = vision_features
                deepstack_embeds = None

            # Get text embeddings and replace image tokens with vision features
            input_embeds = self.model.embed_tokens(input_ids)
            image_mask = input_ids == self.image_token_id
            if image_mask.any():
                vit_prefill_tokens = int(image_mask.sum().item())
                input_embeds[image_mask] = vision_embeds.to(input_embeds.dtype)

            # Build per-token deepstack embeddings
            if deepstack_embeds is not None and image_mask.any():
                input_deepstack_embeds = torch.zeros(
                    input_embeds.shape[0],
                    deepstack_embeds.shape[-1],
                    dtype=input_embeds.dtype,
                    device=input_embeds.device,
                )
                input_deepstack_embeds[image_mask] = deepstack_embeds.to(
                    input_embeds.dtype
                )

        # Text decoder
        _llm_t0 = time.perf_counter()
        hidden_states = (
            self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                input_deepstack_embeds=input_deepstack_embeds,
            )
        )
        _llm_ms = (time.perf_counter() - _llm_t0) * 1000.0

        if forward_batch.forward_mode.is_extend():
            llm_prefill_ms = _llm_ms
            forward_batch.vit_prefill_ms = vit_prefill_ms
            forward_batch.vit_prefill_tokens = vit_prefill_tokens
            forward_batch.llm_prefill_ms = llm_prefill_ms
            forward_batch.llm_decode_ms = None
        else:
            llm_decode_ms = _llm_ms
            forward_batch.llm_decode_ms = llm_decode_ms

        # Prune hidden_states before lm_head to avoid a wasteful
        # [total_tokens, vocab] matmul during prefill.
        # LogitsProcessor._get_pruned_states(): in extend mode only keep
        # the last token of each sequence; in decode mode all rows are
        # already one-per-sequence.
        if forward_batch.forward_mode.is_extend():
            if (
                forward_batch.extend_start_loc is not None
                and forward_batch.extend_seq_lens is not None
            ):
                last_index = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                hidden_states = hidden_states[last_index]
            else:
                hidden_states = hidden_states[-1:]

        # LM head: always use weight matrix directly for the linear
        # projection.  Works for both nn.Embedding (tied) and nn.Linear
        # (untied).
        logits = torch.matmul(
            hidden_states.to(self.lm_head.weight.dtype),
            self.lm_head.weight.T,
        )

        # Return LogitsProcessorOutput so that ModelRunner._process_logits
        # skips redundant last-token gathering.
        from pymllm.executor.model_runner import LogitsProcessorOutput

        return LogitsProcessorOutput(next_token_logits=logits)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from a HuggingFace checkpoint.

        Handles weight name remapping between HuggingFace Qwen3-VL
        checkpoints and this model's parameter names.
        """
        # When quantized, the model has separate q/k/v and gate/up projections
        # (no fused qkv_proj / gate_up_proj), so skip the stacking logic.
        if self.quant_config is not None:
            stacked_params_mapping = []
        else:
            stacked_params_mapping = [
                # (param_name, weight_name, shard_id)
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".up_proj", 1),
                (".gate_up_proj", ".gate_proj", 0),
            ]

        params_dict = dict(self.named_parameters())

        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # When weights are tied, lm_head.weight is the same tensor as
            # embed_tokens.weight — skip the duplicate from the checkpoint.
            if tie_word_embeddings and "lm_head.weight" in name:
                continue

            name = _remap_weight_name(name)

            # Handle language model stacked parameters (QKV, gate_up)
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                _load_stacked_weight(params_dict[name], loaded_weight, shard_id)
                handled = True
                break

            if handled:
                continue

            # Handle vision encoder QKV stacking
            if "visual" in name:
                for qkv_key in (".attn.q.", ".attn.k.", ".attn.v."):
                    if qkv_key not in name:
                        continue
                    qkv_name = name.replace(qkv_key, ".attn.qkv_proj.")
                    if qkv_name in params_dict:
                        shard = {"q": 0, "k": 1, "v": 2}[qkv_key[-2]]
                        _load_vision_qkv_weight(
                            params_dict[qkv_name], loaded_weight, shard
                        )
                        handled = True
                    break

            if handled:
                continue

            # Direct parameter loading
            if name in params_dict:
                param = params_dict[name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, loaded_weight)
                elif param.data.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                else:
                    logger.warning(
                        "Shape mismatch: param %s (%s) vs loaded (%s), skipping.",
                        name,
                        param.data.shape,
                        loaded_weight.shape,
                    )


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _remap_weight_name(name: str) -> str:
    """Remap HuggingFace weight names to pymllm parameter names."""
    # transformers >= v4.52: model.language_model.* -> model.*
    if name.startswith("model.language_model."):
        name = name.replace("model.language_model.", "model.", 1)
    # model.visual.* -> visual.*
    elif name.startswith("model.visual."):
        name = name.replace("model.visual.", "visual.", 1)

    # Vision attention param renaming (checkpoint -> pymllm names)
    if "visual" in name:
        name = name.replace("attn.qkv.", "attn.qkv_proj.")
        name = name.replace("attn.proj.", "attn.out_proj.")

    return name


def _load_stacked_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id,
) -> None:
    """Load one shard (q/k/v or gate/up) into a fused parameter.

    For QKV with GQA (grouped-query attention), Q has a different size
    from K and V.  The fused layout is ``[Q, K, V]`` where
    ``Q_size = total - 2 * KV_size``.  We must use cumulative offsets
    rather than ``idx * shard_size`` to handle the asymmetry correctly.
    """
    if isinstance(shard_id, str):
        # QKV fused layout: [Q, K, V]
        # Q may have a different size from K/V (GQA).
        total_size = param.data.shape[0]
        shard_size = loaded_weight.shape[0]
        if shard_id == "q":
            param.data[0:shard_size].copy_(loaded_weight)
        elif shard_id == "k":
            kv_size = shard_size
            q_size = total_size - 2 * kv_size
            param.data[q_size : q_size + kv_size].copy_(loaded_weight)
        elif shard_id == "v":
            kv_size = shard_size
            q_size = total_size - 2 * kv_size
            param.data[q_size + kv_size : q_size + 2 * kv_size].copy_(loaded_weight)
    else:
        # gate_up: 0 -> gate, 1 -> up (same size, idx*size is correct)
        shard_size = loaded_weight.shape[0]
        param.data[shard_id * shard_size : (shard_id + 1) * shard_size].copy_(
            loaded_weight
        )


def _load_vision_qkv_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_idx: int,
) -> None:
    """Load a Q, K, or V weight shard into a fused QKV parameter."""
    shard_size = param.data.shape[0] // 3
    start = shard_idx * shard_size
    param.data[start : start + shard_size].copy_(loaded_weight)

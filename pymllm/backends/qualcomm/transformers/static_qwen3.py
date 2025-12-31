import torch
from torch import nn
from torch.nn import functional as F
from pymllm.backends.qualcomm.transformers.core.qdq import QDQ_OP
from pymllm.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.backends.qualcomm.transformers.core.qlinear import (
    QLinearLPBQ,
    QLinearW8A16_PerChannelSym_PerTensorSym,
)


class Qwen3Config:
    def __init__(self):
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.bos_token_id = 151643
        self.eos_token_id = 151645
        self.head_dim = 128
        self.hidden_act = "silu"
        self.hidden_size = 2048
        self.initializer_range = 0.02
        self.intermediate_size = 6144
        self.max_position_embeddings = 40960
        self.max_window_layers = 28
        self.model_type = "qwen3"
        self.num_attention_heads = 16
        self.num_hidden_layers = 28
        self.num_key_value_heads = 8
        self.pad_token_id = 151643
        self.rms_norm_eps = 1e-06
        self.rope_scaling = None
        self.rope_theta = 1000000
        self.sliding_window = None
        self.tie_word_embeddings = True
        self.torch_dtype = "bfloat16"
        self.transformers_version = "4.51.0"
        self.use_cache = True
        self.use_sliding_window = False
        self.vocab_size = 151936


def generate_rope_cache(
    max_length: int,
    head_dim: int,
    rope_theta: float,
    dtype=torch.bfloat16,
    device="cpu",
):
    """
    Generate RoPE (Rotary Position Embedding) cache for given max_length.

    Args:
        max_length: Maximum sequence length
        head_dim: Dimension of each attention head
        rope_theta: RoPE theta parameter (frequency base)
        dtype: Data type for the embeddings
        device: Device to place the embeddings on

    Returns:
        tuple: (cos, sin) embeddings of shape [max_length, head_dim]
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(max_length, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QLinearLPBQ(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            block_size=32,
        )
        self.up_proj = QLinearLPBQ(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            block_size=32,
        )
        self.down_proj = QLinearLPBQ(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            block_size=32,
        )
        self.act_fn = nn.SiLU()

        # QDQ
        self.qdq_x = QDQ_OP["A16-PerTensor"]()
        self.qdq_up_result = QDQ_OP["A16-PerTensor"]()
        self.qdq_gate_result = QDQ_OP["A16-PerTensor"]()
        self.qdq_act = QDQ_OP["A16-PerTensor"]()
        self.qdq_middle = QDQ_OP["A16-PerTensor"]()

    def forward(self, x):
        """
        input:
            x: bf16, w/o fakequant
        output:
            o: bf16, w/o fakequant
        """
        x = self.qdq_x(x)
        up_result = self.qdq_up_result(self.up_proj(x))
        gate_result = self.qdq_gate_result(self.gate_proj(x))
        up_result = self.qdq_act(self.act_fn(up_result))
        o = self.qdq_middle(gate_result * up_result)
        o = self.down_proj(o)
        return o


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            block_size=32,
        )
        self.k_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            block_size=32,
        )
        self.v_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            block_size=32,
        )
        self.o_proj = QLinearLPBQ(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            block_size=32,
        )
        self.q_norm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # QDQ
        self.qdq_hidden_states = QDQ_OP["A16-PerTensor"]()
        self.qdq_0 = QDQ_OP["A16-PerTensor"]()
        self.qdq_1 = QDQ_OP["A16-PerTensor"]()
        self.qdq_2 = QDQ_OP["A16-PerTensor"]()
        self.qdq_3 = QDQ_OP["A16-PerTensor"]()
        self.qdq_4 = QDQ_OP["A8-PerTensor"]()
        self.qdq_5 = QDQ_OP["A16-PerTensor"]()
        self.qdq_6 = QDQ_OP["A16-PerTensor"]()
        self.qdq_7 = QDQ_OP["A16-PerTensor"]()
        self.qdq_8 = QDQ_OP["A16-PerTensor"]()
        self.qdq_9 = QDQ_OP["A16-PerTensor"]()
        self.qdq_10 = QDQ_OP["A16-PerTensor"]()
        self.qdq_11 = QDQ_OP["A16-PerTensor"]()
        self.qdq_12 = QDQ_OP["A16-PerTensor"]()
        self.qdq_13 = QDQ_OP["A16-PerTensor"]()
        self.qdq_14 = QDQ_OP["A8-PerTensor"]()

        self.qdq_rope_0 = QDQ_OP["A16-PerTensor"]()
        self.qdq_rope_1 = QDQ_OP["A16-PerTensor"]()
        self.qdq_rope_2 = QDQ_OP["A16-PerTensor"]()
        self.qdq_rope_3 = QDQ_OP["A16-PerTensor"]()
        self.qdq_rope_4 = QDQ_OP["A16-PerTensor"]()
        self.qdq_rope_5 = QDQ_OP["A16-PerTensor"]()

    def forward(
        self,
        hidden_states: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        causal_mask: torch.Tensor,
    ):
        """
        input:
            hidden_states: bf16, w/o fakequant
        output:
            o: bf16, w/o fakequant
        """
        bsz, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        quantized_hidden_states = self.qdq_hidden_states(hidden_states)

        # [B, H, S, D]
        query_states = (
            self.q_proj(quantized_hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_states = (
            self.k_proj(quantized_hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_states = (
            self.v_proj(quantized_hidden_states).view(hidden_shape).transpose(1, 2)
        )

        query_states = self.q_norm(self.qdq_0(query_states))
        query_states = self.qdq_1(query_states)

        key_states = self.k_norm(self.qdq_2(key_states))
        key_states = self.qdq_3(key_states)

        # ROPE Here
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # q_embed = (q * cos) + (rotate_half(q) * sin)
        # k_embed = (k * cos) + (rotate_half(k) * sin)
        cos_embedding = cos.unsqueeze(1)
        sin_embedding = sin.unsqueeze(1)
        rot_q = rotate_half(query_states)
        rot_k = rotate_half(key_states)
        query_states = self.qdq_rope_0(
            self.qdq_rope_1(query_states * cos_embedding)
            + self.qdq_rope_2(rot_q * sin_embedding)
        )
        key_states = self.qdq_rope_3(
            self.qdq_rope_4(key_states * cos_embedding)
            + self.qdq_rope_5(rot_k * sin_embedding)
        )

        key_states = self.qdq_4(key_states)
        key_states = key_states.transpose(2, 3)  # [B, H, D, S]
        key_states = repeat_kv(key_states, self.num_key_value_groups)

        attn = query_states @ key_states
        attn = self.qdq_5(attn)
        attn = attn / self.qdq_6(torch.ones(1, dtype=torch.bfloat16) * self.scaling)
        attn = self.qdq_7(attn)
        attn_min = torch.amin(attn, dim=-1, keepdim=True)
        attn_min = self.qdq_8(attn_min)
        attn_vv = attn_min - 20
        attn_vv = self.qdq_9(attn_vv)
        attn = torch.where(causal_mask == 0, attn, attn_vv)
        attn = self.qdq_10(attn)
        attn = F.softmax(attn, -1)
        attn = self.qdq_11(attn)
        y = attn @ self.qdq_14(self.qdq_13(value_states))
        y = self.qdq_12(y)
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        y = self.o_proj(y)
        return y


class Qwen3DecodeLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = QRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.qdq_0 = QDQ_OP["A16-PerTensor"]()
        self.qdq_1 = QDQ_OP["A16-PerTensor"]()
        self.qdq_2 = QDQ_OP["A16-PerTensor"]()
        self.qdq_3 = QDQ_OP["A16-PerTensor"]()

    def forward(
        self,
        hidden_states: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        causal_mask: torch.Tensor,
    ):
        """
        inputs:
            hidden_states: bf16, w/o fakequant
        outputs:
            hidden_states: bf16, w/o fakequant
        """
        hidden_states = self.qdq_0(hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            sin,
            cos,
            causal_mask,
        )
        hidden_states = self.qdq_2(residual + self.qdq_1(hidden_states))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.qdq_3(hidden_states)
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecodeLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = QRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.qdq_0 = QDQ_OP["A16-PerTensor"]()

    def forward(self, input_ids, sin, cos, causal_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, sin, cos, causal_mask)

        hidden_states = self.norm(self.qdq_0(hidden_states))
        return hidden_states


class Qwen3ForCausalLM:
    def __init__(self, config):
        self.config = config
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = QLinearW8A16_PerChannelSym_PerTensorSym(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.qdq_0 = QDQ_OP["A16-PerTensor"]()
        self.qdq_1 = QDQ_OP["A16-PerTensor"]()
        self.qdq_2 = QDQ_OP["A16-PerTensor"]()

        # Register sin and cos as buffers
        self.register_buffer("sin", None)
        self.register_buffer("cos", None)

        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        input_ids,
        position_ids,
        max_length,
    ):
        bsz, seq_len = input_ids.shape

        # Generate causal mask based on position_ids length
        # For prefill, we need a lower triangular mask
        causal_mask = 1 - torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.int8, device=input_ids.device)
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Generate or use registered RoPE embeddings
        if self.sin is None or self.cos is None or self.cos.shape[0] < max_length:
            cos, sin = generate_rope_cache(
                max_length,
                head_dim=self.config.head_dim,
                rope_theta=self.config.rope_theta,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
            # Register the generated embeddings
            self.sin = self.qdq_1(sin)
            self.cos = self.qdq_2(cos)

        if self.k_cache is None or self.v_cache is None:
            pass

        # Slice RoPE embeddings to current sequence length
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        out = self.model(input_ids, sin, cos, causal_mask)
        logits = self.lm_head(self.qdq_0(out))
        return logits

    def _update_kv_cache_by_copy(self):
        pass

    def _freeze_observer(self):
        pass

    def infer(self, model_path: str, prompt: str, max_length) -> str:
        pass

    def calibrate(self, model_path: str, dataset_path: str):
        pass

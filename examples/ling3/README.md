# Ling-3.0-tiny mobile CPU runner

This runner targets the pinned official checkpoint
`inclusionAI/Ling-3.0-tiny@a2ee06c0f2de5b171701aee7f73f70a1da75483b`.
The supported deployment envelope is batch 1, a 2048-token cache, float32
recurrent state, and KAI W4A32 linear weights on ARM64 macOS and Android.
Build the ARM CPU backend without `-ffast-math`: Ling-3's recurrent gates are
not compatible with relaxed IEEE semantics. For Android, the validated ISA
flags are `-march=armv8.2-a+fp16+fp16fml+dotprod+i8mm`; do not append
`-ffast-math`.

Validate the source checkpoint before conversion:

```bash
python3 validate_checkpoint.py /path/to/Ling-3.0-tiny \
  --observed-revision a2ee06c0f2de5b171701aee7f73f70a1da75483b
```

Convert with the repository V2 converter and
`quant_cfg_tiny_w4a32_kai.json`, using model name `Ling-3.0-tiny`, then seal
the output descriptor table:

```bash
python3 validate_converted_model.py /path/to/Ling-3.0-tiny.mllm \
  /path/to/Ling-3.0-tiny
```

Run one deterministic smoke request:

```bash
./mllm-ling3-runner \
  --model_path /path/to/Ling-3.0-tiny.mllm \
  --tokenizer_path /path/to/Ling-3.0-tiny/tokenizer.json \
  --config_path config_tiny_w4a32_kai.json \
  --prompt '你好，请用一句话介绍你自己。' \
  --max_new_tokens 8 --print_token_ids
```

The runner emits `LING3_RUN_START`, generated token IDs, and
`LING3_RUN_OK`. A successful build or tokenizer-only test is not a full-model
runtime result; device evidence must retain the converted model SHA256 and
the runner/library identities together.

For a longer correctness demo, pass the prompt directly and use a 64-token
generation limit:

```bash
./mllm-ling3-runner \
  --model_path /path/to/Ling-3.0-tiny.mllm \
  --tokenizer_path /path/to/Ling-3.0-tiny/tokenizer.json \
  --config_path config_tiny_w4a32_kai.json \
  --prompt '请用中文详细介绍 Ling-3.0-tiny 的混合注意力架构，并解释 KDA、MLA 和 MoE 各自的作用。' --disable_thinking \
  --max_new_tokens 64 --print_token_ids
```

The expected completion marker is
`LING3_RUN_OK prompt_tokens=49 generated_tokens=64`. This is a generation
correctness demo, not a perplexity, model-quality, or performance benchmark.

## Retained tests

The Ling-specific coverage follows the repository test layout (`ctest` labels
in parentheses):

| Layer | Target | Notes |
| --- | --- | --- |
| KDA kernel | `Mllm-Test-CPUKernel --gtest_filter='KimiDeltaAttentionKernelTest.*'` (`cpu-kernel`) | scalar-reference oracle, bitwise prefill/decode and serial/parallel checks |
| Public `nn::KimiDeltaAttention` | `Mllm-Test-Nn-KimiDeltaAttention` (`KimiDeltaAttentionFocused`) | eager reference, in-place state, chunked equivalence, trace + serialization |
| Causal convolution | upstream `Mllm-Test-Nn-CausalDepthwiseConv1D` and the `CausalDepthwiseConv*KernelTest` suites | Ling registers the upstream operation with the current-first order |
| Model | `Mllm-Test-Ling3-Config`, `Mllm-Test-Ling3-RoPE`, `Mllm-Test-Ling3-Tokenizer` (`ling3`) | set `MLLM_LING3_EXAMPLE_DIR` on a device; the tokenizer test skips unless `LING3_OFFICIAL_TOKENIZER` points at the official `tokenizer.json` |

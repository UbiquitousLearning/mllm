# Spark-X2.5-1.7B on CPU

This example implements `XHToken/Spark-X2.5-1.7B` at revision
`448e61eb392c00f2c403185c5b56d5e0665bfaab`: 28 dense decoder layers,
8 query / 2 KV heads of dimension 256, headwise sigmoid attention gates,
GELU-gated MLPs, and shared source embedding/output weights.
Every fourth layer uses full attention with 64-dimensional partial RoPE;
the other layers use a 512-token causal window and full-dimensional RoPE.

The runtime supports batch-1 CPU text generation, FP32 and ARM KAI W4A32.
The supplied configs cap context at 4096 tokens; `max_cache_length` can be
increased to 8192 for larger requests. The checkpoint's 1M positional range is
not a tested mobile capacity claim. Full-attention KV grows with context;
sliding layers retain only the 511 past tokens needed by their next forward.
Chunked prefill uses absolute positions and per-query window boundaries.

## Convert

Download the pinned checkpoint, including tokenizer.json and the index/shards.
Use separate paths for checkpoint, converted files, and build output.
The converter validates every tensor name, shape and source dtype, and refuses
to overwrite existing output or partial files.

```bash
python examples/spark2_5/convert.py \
  --checkpoint /path/to/Spark-X2.5-1.7B \
  --output /path/to/spark-fp32.mllm

# Requires the repository's built pymllm mobile binding; packing is portable
# and can run on an x86 host. Use the candidate's matching runtime libraries.
python examples/spark2_5/convert.py \
  --checkpoint /path/to/Spark-X2.5-1.7B \
  --output /path/to/spark-w4a32.mllm --quantized
```

KAI quantizes QKV, attention output, FFN and output-head matrices. Norms,
lookup embedding and the small 8-output gate remain FP32. The output head is
derived from `model.embedding.weight`; lookup and projection have different
storage formats, so tied checkpoint weights do not imply one runtime buffer.
W4A32 uses FP32 interfaces and dynamically quantizes activations to INT8 inside
KleidiAI. Its artifact must be paired with `config_1.7B_w4a32_kai.json`.

## Build and run

```bash
cmake -S . -B build -DMLLM_ENABLE_EXAMPLE=ON
cmake --build build --target mllm-spark25-runner
OMP_NUM_THREADS=1 build/bin/mllm-spark25-runner \
  --model_path /path/to/spark-fp32.mllm \
  --config_path examples/spark2_5/config_1.7B_fp32.json \
  --tokenizer_path /path/to/Spark-X2.5-1.7B/tokenizer.json \
  --prompt "Explain how a sliding-window attention cache works." \
  --enable_thinking false --max_new_tokens 128 --print_token_ids
```

For Android, use the repository's Android NDK build settings and build the
same runner target with `MLLM_BUILD_ARM_BACKEND=ON`. Select an ARM ISA supported
by the target phone, including the instructions required by the chosen KAI
implementation.

The runner defaults to one CPU operation thread. For OpenMP builds, also set
`OMP_NUM_THREADS=1`, as shown above. Override operation threading with
`--engine_cpu_op_thread N`; multi-thread performance is not qualified by this
example. The CLI renders the official single-user-turn template, with an optional
system message. Thinking defaults to enabled; disabling it appends `</think>`
in the generation prompt. Decoding is greedy and stops at EOS id 1 or the
requested generation limit. UTF-8 bytes are buffered across token boundaries.
The runner does not provide tool execution or a conversation-history API.

## Tests and numerical contract

`GELUFocused`, `SigmoidFocused`, `GroupedQueryAttentionFocused`, and `Spark25Focused` cover the
public operations, window/chunk semantics, synthetic model/reset, configuration
and tokenizer. Official tokenizer parity fixtures may be supplied through
`MLLM_SPARK_TOKENIZER_FIXTURES` (tokenizer.json and tokenizer_cases.json).
Spark requests erf GELU and accurate logistic Sigmoid; existing callers retain
their default approximations. FP32 runtime calculations are compared with an FP32 official-model oracle;
this is distinct from bitwise reproduction of the checkpoint's BF16 arithmetic.
Quantized quality and device execution require their own validation.

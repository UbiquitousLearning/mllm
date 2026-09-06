# MiniCPM5-1B on ARM CPU

This example supports the text-only [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) checkpoint.
The runtime contract is batch 1, a 2048-token mobile cache, 16 query heads, 2 native KV heads, and explicit
`head_dim=128`. The implementation keeps KV history at KV-head count and uses a correctness-first eager GQA path;
it does not materialize a persistent 16-head KV cache.

The runner implements the official no-tool chat-template branch, including optional system text and
`enable_thinking=true|false`. Tool schemas, tool calls, and multi-turn history are outside this first product surface.

### Chat template backend

`config.json` selects how the runner renders the prompt through
`chat_template_backend`: `legacy` (default) keeps the byte-stable formatter
above; `jinja_required` renders the checkpoint's own `chat_template.jinja`,
discovered next to `tokenizer.json`, with the vendored `jinja.cpp` engine and
requires a build with `-DMLLM_ENABLE_JINJA_CHAT_TEMPLATE=ON`. A
`jinja_required` model fails at load time instead of falling back when Jinja
support or the template is missing. See
`mllm/preprocessor/chat_template/README.md` for the pipeline and the
Transformers parity gate.

## Convert

Keep embeddings and norms in FP32; the supplied configuration packs transformer Linear weights and the independent
`lm_head` for the KAI W4A32 runtime.

```bash
python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/MiniCPM5-1B \
  --output_path /path/to/minicpm5-1b-w4a32-kai.mllm \
  --model_name MiniCPM5-1B \
  --cfg_path examples/minicpm5/quant_cfg_1B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --format v2 \
  --verbose
```

## Run

```bash
mllm-minicpm5-runner \
  --model_path /path/to/minicpm5-1b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/MiniCPM5-1B/tokenizer.json \
  --config_path examples/minicpm5/config_1B_w4a32_kai.json \
  --prompt "用一句话介绍你自己。" \
  --max_new_tokens 32
```

Omit `--prompt` for the interactive loop. Each prompt is an independent conversation and clears all logical cache
slots before inference. Add `--enable_thinking` to open the official thinking prefix; the default emits the official
closed empty thinking block.

## Reproducible 200-token demo

`demo_prompt_200.txt` is exactly 200 input tokens after the official no-tool chat template is applied with no system
message and `enable_thinking=false`. Benchmark mode checks that count before inference, resets model state before every
request, forces a fixed output length, and writes one JSON object per request rather than relying on console timing.

```bash
mllm-minicpm5-runner \
  --model_path /path/to/minicpm5-1b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/MiniCPM5-1B/tokenizer.json \
  --config_path examples/minicpm5/config_1B_w4a32_kai.json \
  --prompt_file examples/minicpm5/demo_prompt_200.txt \
  --expected_prompt_tokens 200 \
  --max_new_tokens 32 \
  --benchmark_warmup 1 \
  --benchmark_samples 5 \
  --benchmark_jsonl /path/to/fresh-results.jsonl \
  --benchmark_variant minicpm5-1b-w4a32-kai \
  --benchmark_source_sha SOURCE_COMMIT_SHA \
  --engine_cpu_op_thread 4 \
  --engine_dispatcher_thread 4
```

The runner uses greedy decoding with `min_new_tokens=max_new_tokens`, so every valid record contains 32 generated
tokens and 31 decode steps. Compute prefill throughput from `prefill_tokens / prefill_duration_us`, and decode
throughput from `decode_steps / decode_duration_us`; the first generated token belongs to TTFT and is not counted as a
decode step. The JSONL also retains wall time, token IDs, process affinity, and visible CPU/thermal telemetry.

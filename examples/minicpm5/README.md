# MiniCPM5 on ARM CPU

This example supports the text-only [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) and
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B/tree/0e9c66dce9fedde5ba8663bbcdd54b6810bb929a) checkpoints.
The runtime contract is batch 1, a 2048-token mobile cache, 16 query heads, 2 native KV heads, and explicit
`head_dim=128`. The implementation keeps KV history at KV-head count and uses a correctness-first eager GQA path;
it does not materialize a persistent 16-head KV cache.

The runner implements the official no-tool chat-template branch, including optional system text and
`enable_thinking=true|false`. Tool schemas, tool calls, and multi-turn history are outside this first product surface.

| Variant | Layers | Hidden / FFN width | Runtime config | Quantization config |
| --- | ---: | --- | --- | --- |
| 1B | 24 | 1536 / 4608 | `config_1B_w4a32_kai.json` | `quant_cfg_1B_w4a32_kai.json` |
| 2B | 42 | 2048 / 6144 | `config_2B_w4a32_kai.json` | `quant_cfg_2B_w4a32_kai.json` |

Both variants reuse the same model graph, native KV cache and GQA kernels. At the supplied 2048-token
capacity, FP32 K/V payloads occupy 96 MiB for 1B and 168 MiB for 2B. The official 131072-position
RoPE limit does not enlarge the runner's supported mobile cache.
Use the matching checkpoint, runtime config and quantization config together; mixed 1B/2B geometry is rejected.
The 2B tokenizer vocabulary is unchanged. Its updated history template does not change the supported
single-user branch; multi-turn history and tool calling remain unsupported.

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

For 2B, use `MiniCPM5-2B` and `minicpm5-2b-w4a32-kai.mllm` in these commands and select
`quant_cfg_2B_w4a32_kai.json` during conversion and `config_2B_w4a32_kai.json` during inference.
The same `demo_prompt_200.txt` is used for both sizes.

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

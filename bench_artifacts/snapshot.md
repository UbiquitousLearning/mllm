# TinyLlama x86 Context Sweep Snapshot

8 threads, CL = 256 ~ 4096, CPU only.

Decode per-token latency is stable (~0.67–0.88 ms/tok) across all context lengths.
Prefill latency grows roughly quadratically — this is the bottleneck at long contexts.

`perf` hotspot: tinyBLAS small-tile GEMM.

Shape log shows decode is M=1 GEMMs, prefill hits M=254.
Peak RSS tracks linearly with KV cache size as expected.
These shape/memory numbers can inform future AOT padding choices.

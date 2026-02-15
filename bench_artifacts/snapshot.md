# MLLM x86 TinyLlama: Context Sweep & Profiling Snapshot

## 1. Summary
On the x86_64 CPU baseline (8 threads), TinyLlama's decode per-token latency remains highly stable (~0.67–0.88 ms/tok) across context lengths from CL=512 to 4096. However, TTFT (Time-to-First-Token) / prefill latency scales significantly and becomes the primary compute bottleneck for long contexts.

Performance profiling (`perf`) indicates the hotspot is heavily concentrated in `tinyBLAS` small-tile GEMM templates. 

## 2. Shape Statistics (via MLLM_MATMUL_SHAPE_LOG)
- **Decode Phase:** Compute is dominated by small-M GEMMs (M=1). 
- **Prefill Phase:** Exhibits large matrix shapes, commonly hitting M=254.

## 3. Implications for Static Graph & AOT
1. **Shape Bucketing:** The heavily reused GEMM shapes captured here provide the exact target dimensions required for AOT static-graph nearest-padding.
2. **Memory Planning:** Peak RSS scales predictably with context length. The formula-based KV-cache estimation strictly aligns with the real-time footprint, establishing reliable VRAM lower bounds for AOT memory pre-allocation.

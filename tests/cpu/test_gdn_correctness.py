import torch
import time
from torch.utils.cpp_extension import load_inline
import os

# ==========================================
# ⚠️ 请务必替换成你机器上 GDNKernel.hpp 的真实绝对路径！
# ==========================================
KERNEL_HEADER_PATH = "/home/cjt/mllm/mllm-main/mllm/backends/cpu/kernels/GDNKernel.hpp"

# ==========================================
# 1. 动态编译并加载 C++ Kernel (和之前保持一致)
# ==========================================
print("⚙️ 正在动态编译 GDN Kernel...")
cpp_source = f"""
#include <torch/extension.h>
#include "{KERNEL_HEADER_PATH}"

void gdn_forward_wrapper(
    torch::Tensor s_prev, torch::Tensor k, torch::Tensor v,
    torch::Tensor gate, torch::Tensor beta, torch::Tensor s_new) {{
    mllm::cpu::GDNKernel::forward<float>(
        s_prev.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        gate.data_ptr<float>(), beta.data_ptr<float>(),
        s_new.data_ptr<float>(), nullptr, nullptr, s_prev.size(0)
    );
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &gdn_forward_wrapper, "GDN Forward Kernel");
}}
"""

gdn_jit = load_inline(
    name="gdn_jit_benchmark",
    cpp_sources=cpp_source,
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False
)
print("✅ GDN Kernel 编译加载成功！\n")

# ==========================================
# 2. PyTorch 标准答案 (用于正确性对比)
# ==========================================
def gdn_reference_py(s_prev, k, v, gate, beta):
    alpha = torch.exp(gate).unsqueeze(-1).unsqueeze(-1) 
    beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)           
    s_new = alpha * s_prev
    proj = torch.matmul(s_new, k.unsqueeze(-1)).squeeze(-1)
    erase = beta_expanded * (proj.unsqueeze(-1) * k.unsqueeze(-2))
    s_new = s_new - erase
    write = beta_expanded * (v.unsqueeze(-1) * k.unsqueeze(-2))
    s_new = s_new + write
    return s_new

# ==========================================
# 3. 性能 Benchmark 核心逻辑
# ==========================================
def run_benchmark(B, H, D, warmup=50, repeats=500):
    print(f"📊 开始 Benchmark 测试: Batch={B}, Heads={H}, Dim={D}")
    
    # 准备数据
    s_prev = torch.randn(B, H, D, D, dtype=torch.float32)
    k = torch.randn(B, H, D, dtype=torch.float32)
    v = torch.randn(B, H, D, dtype=torch.float32)
    gate = torch.randn(B, H, dtype=torch.float32)
    beta = torch.randn(B, H, dtype=torch.float32)
    s_new = torch.zeros_like(s_prev)

    # --- 步骤 A: 正确性验证 ---
    with torch.no_grad():
        expected = gdn_reference_py(s_prev.clone(), k.clone(), v.clone(), gate.clone(), beta.clone())
    gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
    max_diff = torch.max(torch.abs(s_new - expected)).item()
    is_correct = torch.allclose(s_new, expected, atol=1e-4, rtol=1e-5)
    
    print(f"   [正确性检查] 最大绝对误差: {max_diff:.6e} | 结果: {'✅ 通过' if is_correct else '❌ 失败'}")
    if not is_correct:
        print("   ⚠️ 警告：正确性未通过，Benchmark 数据可能无效！")

    # --- 步骤 B: 预热 (Warmup) ---
    # 预热是为了让 CPU 频率稳定，并完成指令和数据缓存的预热
    print(f"   [预热阶段] 正在空跑 {warmup} 次以稳定 CPU 频率和缓存...")
    for _ in range(warmup):
        gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
    
    # 确保预热时的计算完成（针对 CPU 主要是防止编译器过度优化）
    torch.cpu.synchronize()

    # --- 步骤 C: 正式计时 (Benchmark) ---
    print(f"   [正式测试] 正在循环运行 {repeats} 次并记录耗时...")
    start_time = time.perf_counter() # 使用高精度计时器
    for _ in range(repeats):
        gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
    torch.cpu.synchronize() # 确保所有计算完成
    end_time = time.perf_counter()

    # --- 步骤 D: 统计结果 ---
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / repeats
    
    # 计算吞吐量 (Tokens/s)。这里假设每个 batch 处理 1 个 token
    # 实际吞吐量取决于你的业务定义，这里以 Batch Size 作为每秒处理的 token 数参考
    tokens_per_second = (B * repeats) / (end_time - start_time)

    print(f"\n📈 Benchmark 结果汇总:")
    print(f"   总耗时: {total_time_ms:.2f} ms (运行 {repeats} 次)")
    print(f"   平均单次耗时 (Latency): {avg_time_ms:.4f} ms")
    print(f"   预估吞吐量 (Throughput): {tokens_per_second:.2f} Tokens/s")
    print("-" * 50)

if __name__ == "__main__":
    # 你可以修改这里的维度，测试不同规模下的性能表现
    # 比如测试 Qwen3.5-0.8B 的标准配置
    run_benchmark(B=1, H=16, D=128, warmup=50, repeats=1000)
    run_benchmark(B=4, H=16, D=128, warmup=50, repeats=500)

import torch
import time
import os
from torch.utils.cpp_extension import load_inline

# 尝试导入表格美化库
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# ==========================================
# 请替换为你真实的 GDNKernel.hpp 绝对路径
# ==========================================
KERNEL_HEADER_PATH = "/home/cjt/mllm/mllm-main/mllm/backends/cpu/kernels/GDNKernel.hpp"

# ==========================================
# 1. 动态编译与加载 Kernel
# ==========================================
print("Compiling GDN Kernel...")
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
    name="gdn_jit_report",
    cpp_sources=cpp_source,
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False
)
print("Kernel compiled and loaded successfully.\n")

# ==========================================
# 2. PyTorch 标准参考实现
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
# 3. 核心测试与 Benchmark 逻辑
# ==========================================
def run_comprehensive_test(configs):
    results = []
    
    # 打印测试环境信息
    print("=" * 90)
    print("GDN Kernel Comprehensive Test Report")
    print("=" * 90)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Kernel Header Path: {KERNEL_HEADER_PATH}")
    print("=" * 90 + "\n")

    for B, H, D, repeats in configs:
        print(f"Testing Configuration: Batch={B}, Heads={H}, Dim={D} ...")
        
        # 准备数据
        s_prev = torch.randn(B, H, D, D, dtype=torch.float32)
        k = torch.randn(B, H, D, dtype=torch.float32)
        v = torch.randn(B, H, D, dtype=torch.float32)
        gate = torch.randn(B, H, dtype=torch.float32)
        beta = torch.randn(B, H, dtype=torch.float32)
        s_new = torch.zeros_like(s_prev)

        # --- A. 正确性验证 ---
        with torch.no_grad():
            expected = gdn_reference_py(s_prev.clone(), k.clone(), v.clone(), gate.clone(), beta.clone())
        
        gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
        max_diff = torch.max(torch.abs(s_new - expected)).item()
        is_correct = torch.allclose(s_new, expected, atol=1e-4, rtol=1e-5)
        status = "PASS" if is_correct else "FAIL"

        # --- B. 性能 Benchmark ---
        # 预热
        for _ in range(20):
            gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
        torch.cpu.synchronize()

        # 正式计时
        start_time = time.perf_counter()
        for _ in range(repeats):
            gdn_jit.forward(s_prev, k, v, gate, beta, s_new)
        torch.cpu.synchronize()
        end_time = time.perf_counter()

        # 计算指标
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / repeats
        tokens_per_second = (B * repeats) / (end_time - start_time)

        # 收集结果
        results.append({
            "Batch": B,
            "Heads": H,
            "Dim": D,
            "Status": status,
            "Max Error": f"{max_diff:.2e}",
            "Avg Latency (ms)": f"{avg_time_ms:.4f}",
            "Throughput (Tokens/s)": f"{tokens_per_second:.2f}"
        })
        print(f"   Completed. Avg Latency: {avg_time_ms:.4f} ms | Throughput: {tokens_per_second:.2f} Tokens/s\n")

    # ==========================================
    # 4. 格式化输出最终报告
    # ==========================================
    print("\n" + "=" * 110)
    print("Final Performance and Correctness Report")
    print("=" * 110)
    
    if tabulate:
        print(tabulate(results, headers="keys", tablefmt="grid", stralign="center", numalign="center"))
    else:
        # 简易打印格式
        print(f"{'Batch':<8} {'Heads':<8} {'Dim':<8} {'Status':<10} {'Max Error':<15} {'Avg Latency (ms)':<20} {'Throughput (Tokens/s)':<20}")
        print("-" * 110)
        for r in results:
            print(f"{r['Batch']:<8} {r['Heads']:<8} {r['Dim']:<8} {r['Status']:<10} {r['Max Error']:<15} {r['Avg Latency (ms)']:<20} {r['Throughput (Tokens/s)']:<20}")
    
    print("=" * 110)
    print("Test Finished.")

if __name__ == "__main__":
    # 在这里定义你想测试的配置组合 [Batch, Heads, Dim, 循环次数]
    test_configs = [
        [1, 16, 128, 1000],
        [4, 16, 128, 500],
        [8, 16, 128, 300],
    ]
    run_comprehensive_test(test_configs)

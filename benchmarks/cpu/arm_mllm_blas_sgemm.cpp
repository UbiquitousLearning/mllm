// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cassert>
#include <cstdint>

#include <benchmark/benchmark.h>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"

bool is_aligned_16(const void* addr) { return (reinterpret_cast<uintptr_t>(addr) & 0xF) == 0; }

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
static void mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(benchmark::State& state) {
  int D = state.range(0);
  int S = state.range(1);

  auto A = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
  auto B = mllm::Tensor::random({S, D}, mllm::kFloat32, mllm::kCPU);
  auto C = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
  auto DST = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);

  auto a_ptr = A.ptr<float>();
  auto b_ptr = B.ptr<float>();
  auto c_ptr = C.ptr<float>();
  auto dst_ptr = DST.ptr<float>();

  for (auto _ : state) {
    mllm::cpu::arm::__mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(1, D, S, dst_ptr, a_ptr, b_ptr, c_ptr, false,
                                                                                 true, 1);
  }
}

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
static void mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_neon(benchmark::State& state) {
  int D = state.range(0);
  int S = state.range(1);

  auto A = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
  auto B = mllm::Tensor::random({S, D}, mllm::kFloat32, mllm::kCPU);
  auto C = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
  auto DST = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);

  auto a_ptr = A.ptr<float>();
  auto b_ptr = B.ptr<float>();
  auto c_ptr = C.ptr<float>();
  auto dst_ptr = DST.ptr<float>();

  assert(is_aligned_16(a_ptr));
  assert(is_aligned_16(b_ptr));
  assert(is_aligned_16(c_ptr));
  assert(is_aligned_16(dst_ptr));

  for (auto _ : state) {
    mllm::cpu::arm::__mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(1, D, S, dst_ptr, a_ptr, b_ptr, c_ptr, false, true, 1);
  }
}

static void mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk(benchmark::State& state) {
  int D = state.range(0);
  int S = state.range(1);
  int H = 28;

  auto A = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);
  auto B = mllm::Tensor::random({H, S, D}, mllm::kFloat32, mllm::kCPU);
  auto C = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);
  auto DST = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);

  auto a_ptr = A.ptr<float>();
  auto b_ptr = B.ptr<float>();
  auto c_ptr = C.ptr<float>();
  auto dst_ptr = DST.ptr<float>();

  assert(is_aligned_16(a_ptr));
  assert(is_aligned_16(b_ptr));
  assert(is_aligned_16(c_ptr));
  assert(is_aligned_16(dst_ptr));

  for (auto _ : state) {
    mllm::cpu::arm::__mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk(H, 1, D, S, D, D, S * D, D, dst_ptr, a_ptr, b_ptr,
                                                                              c_ptr, false, true, 1);
  }
}

static void mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk_4threads(benchmark::State& state) {
  int D = state.range(0);
  int S = state.range(1);
  int H = 28;

  auto A = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);
  auto B = mllm::Tensor::random({H, S, D}, mllm::kFloat32, mllm::kCPU);
  auto C = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);
  auto DST = mllm::Tensor::random({H, 1, D}, mllm::kFloat32, mllm::kCPU);

  auto a_ptr = A.ptr<float>();
  auto b_ptr = B.ptr<float>();
  auto c_ptr = C.ptr<float>();
  auto dst_ptr = DST.ptr<float>();

  assert(is_aligned_16(a_ptr));
  assert(is_aligned_16(b_ptr));
  assert(is_aligned_16(c_ptr));
  assert(is_aligned_16(dst_ptr));

  for (auto _ : state) {
    mllm::cpu::arm::__mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk<true>(H, 1, D, S, D, D, S * D, D, dst_ptr, a_ptr,
                                                                                    b_ptr, c_ptr, false, true, 4);
  }
}

BENCHMARK(mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline)
    ->ArgsProduct({
        {64, 96, 128}, {1, 4, 32, 128, 256, 1024},  // S

    })
    ->ArgNames({"D", "S"});

BENCHMARK(mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_neon)
    ->ArgsProduct({
        {64, 96, 128},               // D
        {1, 4, 32, 128, 256, 1024},  // S
    })
    ->ArgNames({"D", "S"});

BENCHMARK(mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk)
    ->ArgsProduct({
        {64, 96, 128},               // D
        {1, 4, 32, 128, 256, 1024},  // S
    })
    ->ArgNames({"D", "S"});

BENCHMARK(mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk_4threads)
    ->ArgsProduct({
        {64, 96, 128},               // D
        {1, 4, 32, 128, 256, 1024},  // S
    })
    ->ArgNames({"D", "S"});

int main(int argc, char** argv) {
  mllm::initializeContext();
  benchmark ::MaybeReenterWithoutASLR(argc, argv);
  char arg0_default[] = "benchmark";
  char* args_default = reinterpret_cast<char*>(arg0_default);
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  ::benchmark ::Initialize(&argc, argv);
  if (::benchmark ::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark ::RunSpecifiedBenchmarks();
  ::benchmark ::Shutdown();
  mllm::shutdownContext();
  return 0;
}

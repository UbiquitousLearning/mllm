// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace mllm::cpu {

// CPU Architecture flags
enum class CPUArch : int32_t {
  UNKNOWN_ARCH = 0,
  X86_ARCH = 1,
  X86_64_ARCH = 2,
  ARM_ARCH = 3,
  ARM64_ARCH = 4,
  MIPS_ARCH = 5,
  MIPS64_ARCH = 6,
  LOONGARCH_ARCH = 7,
  RISCV_ARCH = 8,
  RISCV64_ARCH = 9
};

// Architecture detection using preprocessor macros
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
constexpr CPUArch CURRENT_ARCH = CPUArch::X86_64_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "x86_64";
#define MLLM_HOST_ARCH_X86_64 1
#elif defined(__i386__) || defined(_M_IX86)
constexpr CPUArch CURRENT_ARCH = CPUArch::X86_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "x86";
#define MLLM_HOST_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
constexpr CPUArch CURRENT_ARCH = CPUArch::ARM64_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "arm64";
#define MLLM_HOST_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
constexpr CPUArch CURRENT_ARCH = CPUArch::ARM_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "arm";
#define MLLM_HOST_ARCH_ARM 1
#elif defined(__mips__) || defined(__mips) || defined(__MIPS__)
#if defined(__mips64) || defined(__mips64__)
constexpr CPUArch CURRENT_ARCH = CPUArch::MIPS64_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "mips64";
#define MLLM_HOST_ARCH_MIPS64 1
#else
constexpr CPUArch CURRENT_ARCH = CPUArch::MIPS_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "mips";
#define MLLM_HOST_ARCH_MIPS 1
#endif
#elif defined(__loongarch__)
constexpr CPUArch CURRENT_ARCH = CPUArch::LOONGARCH_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "loongarch";
#define MLLM_HOST_ARCH_LOONGARCH 1
#elif defined(__riscv)
#if __riscv_xlen == 64
constexpr CPUArch CURRENT_ARCH = CPUArch::RISCV64_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "riscv64";
#define MLLM_HOST_ARCH_RISCV64 1
#else
constexpr CPUArch CURRENT_ARCH = CPUArch::RISCV_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "riscv";
#define MLLM_HOST_ARCH_RISCV 1
#endif
#else
constexpr CPUArch CURRENT_ARCH = CPUArch::UNKNOWN_ARCH;
constexpr const char* CURRENT_ARCH_STRING = "unknown";
#define MLLM_HOST_ARCH_UNKNOWN 1
#endif

// Helper functions to check specific architectures
constexpr bool isX86() { return CURRENT_ARCH == CPUArch::X86_ARCH; }

constexpr bool isX86_64() { return CURRENT_ARCH == CPUArch::X86_64_ARCH; }

constexpr bool isARM() { return CURRENT_ARCH == CPUArch::ARM_ARCH; }

constexpr bool isARM64() { return CURRENT_ARCH == CPUArch::ARM64_ARCH; }

constexpr bool isMIPS() { return CURRENT_ARCH == CPUArch::MIPS_ARCH; }

constexpr bool isMIPS64() { return CURRENT_ARCH == CPUArch::MIPS64_ARCH; }

constexpr bool isLoongArch() { return CURRENT_ARCH == CPUArch::LOONGARCH_ARCH; }

constexpr bool isRISCV() { return CURRENT_ARCH == CPUArch::RISCV_ARCH; }

constexpr bool isRISCV64() { return CURRENT_ARCH == CPUArch::RISCV64_ARCH; }

constexpr bool isUnknownArch() { return CURRENT_ARCH == CPUArch::UNKNOWN_ARCH; }

// CPU Feature detection for ARM
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
// ARM features detection
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_ARGS)
#define MLLM_HOST_FEATURE_FP16 1
#endif

#if defined(__ARM_FEATURE_BF16)
#define MLLM_HOST_FEATURE_BF16 1
#endif

#if defined(__ARM_FEATURE_SVE)
#define MLLM_HOST_FEATURE_SVE 1
#endif

#if defined(__ARM_FEATURE_SME)
#define MLLM_HOST_FEATURE_SME 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define MLLM_HOST_FEATURE_NEON 1
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#define MLLM_HOST_FEATURE_DOTPROD 1
#endif

#endif  // ARM architectures

// CPU Feature detection for x86/x86_64
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// x86 features detection - SSE family
#if defined(__SSE__)
#define MLLM_HOST_FEATURE_SSE 1
#endif

#if defined(__SSE2__)
#define MLLM_HOST_FEATURE_SSE2 1
#endif

#if defined(__SSE3__)
#define MLLM_HOST_FEATURE_SSE3 1
#endif

#if defined(__SSSE3__)
#define MLLM_HOST_FEATURE_SSSE3 1
#endif

#if defined(__SSE4_1__)
#define MLLM_HOST_FEATURE_SSE4_1 1
#endif

#if defined(__SSE4_2__)
#define MLLM_HOST_FEATURE_SSE4_2 1
#endif

// AVX family
#if defined(__AVX__)
#define MLLM_HOST_FEATURE_AVX 1
#endif

#if defined(__AVX2__)
#define MLLM_HOST_FEATURE_AVX2 1
#endif

#if defined(__AVX512F__)
#define MLLM_HOST_FEATURE_AVX512F 1
#endif

#if defined(__AVX512BW__)
#define MLLM_HOST_FEATURE_AVX512BW 1
#endif

#if defined(__AVX512CD__)
#define MLLM_HOST_FEATURE_AVX512CD 1
#endif

#if defined(__AVX512DQ__)
#define MLLM_HOST_FEATURE_AVX512DQ 1
#endif

#if defined(__AVX512VL__)
#define MLLM_HOST_FEATURE_AVX512VL 1
#endif

#if defined(__FMA__)
#define MLLM_HOST_FEATURE_FMA 1
#endif

#endif  // x86 architectures

// Helper functions to check specific features
// ARM features
#ifdef MLLM_HOST_FEATURE_FP16
constexpr bool hasFP16() { return true; }
#else
constexpr bool hasFP16() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_BF16
constexpr bool hasBF16() { return true; }
#else
constexpr bool hasBF16() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SVE
constexpr bool hasSVE() { return true; }
#else
constexpr bool hasSVE() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SME
constexpr bool hasSME() { return true; }
#else
constexpr bool hasSME() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_NEON
constexpr bool hasNEON() { return true; }
#else
constexpr bool hasNEON() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_DOTPROD
constexpr bool hasDotProd() { return true; }
#else
constexpr bool hasDotProd() { return false; }
#endif

// x86 SSE family features
#ifdef MLLM_HOST_FEATURE_SSE
constexpr bool hasSSE() { return true; }
#else
constexpr bool hasSSE() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SSE2
constexpr bool hasSSE2() { return true; }
#else
constexpr bool hasSSE2() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SSE3
constexpr bool hasSSE3() { return true; }
#else
constexpr bool hasSSE3() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SSSE3
constexpr bool hasSSSE3() { return true; }
#else
constexpr bool hasSSSE3() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SSE4_1
constexpr bool hasSSE4_1() { return true; }
#else
constexpr bool hasSSE4_1() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_SSE4_2
constexpr bool hasSSE4_2() { return true; }
#else
constexpr bool hasSSE4_2() { return false; }
#endif

// x86 AVX family features
#ifdef MLLM_HOST_FEATURE_AVX
constexpr bool hasAVX() { return true; }
#else
constexpr bool hasAVX() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX2
constexpr bool hasAVX2() { return true; }
#else
constexpr bool hasAVX2() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX512F
constexpr bool hasAVX512F() { return true; }
#else
constexpr bool hasAVX512F() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX512BW
constexpr bool hasAVX512BW() { return true; }
#else
constexpr bool hasAVX512BW() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX512CD
constexpr bool hasAVX512CD() { return true; }
#else
constexpr bool hasAVX512CD() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX512DQ
constexpr bool hasAVX512DQ() { return true; }
#else
constexpr bool hasAVX512DQ() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_AVX512VL
constexpr bool hasAVX512VL() { return true; }
#else
constexpr bool hasAVX512VL() { return false; }
#endif

#ifdef MLLM_HOST_FEATURE_FMA
constexpr bool hasFMA() { return true; }
#else
constexpr bool hasFMA() { return false; }
#endif

}  // namespace mllm::cpu
// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#if defined(__ANDROID__) && defined(__aarch64__) && defined(__clang__) && __clang_major__ >= 19 \
    && __has_include(<arm_sme.h>)
#define MLLM_ANDROID_SME_SPARSE_PV 1

#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace mllm::cpu::shadownpu {

inline constexpr int kSparsePvRows = 16;
inline constexpr int kSparsePvChannels = 128;
inline constexpr int kSparsePvSharedRows = 2 * kSparsePvRows;
inline constexpr uint32_t kSparsePvSharedKeyMask = 0xffff;
inline constexpr uint32_t kSparsePvSharedFirst = 1U << 16;
inline constexpr uint32_t kSparsePvSharedSecond = 1U << 17;

inline bool sparsePvOption(const char* name) noexcept {
  const auto* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool useSparsePvSme() noexcept {
  constexpr unsigned long required = HWCAP2_SME | HWCAP2_SME_F16F32;
  static const bool available = sparsePvOption("MLLM_HMX_SME_PV") && (getauxval(AT_HWCAP2) & required) == required;
  if (!available) return false;
  const int length = prctl(PR_SME_GET_VL, 0L, 0L, 0L, 0L);
  return length >= 0 && (length & PR_SME_VL_LEN_MASK) == 64;
}

inline bool traceSparsePvSme() noexcept {
  static const bool trace = sparsePvOption("MLLM_HMX_SME_PV_TRACE");
  return trace;
}

inline bool useSparsePvBitmapPanel() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_BITMAP_PANEL");
  return enabled;
}

inline bool useSparsePvRetiringPanel() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_RETIRE_PANEL");
  return enabled;
}

inline bool useSparsePvOrderedKeys() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_ORDERED_KEYS");
  return enabled;
}

inline bool useSparsePvAlignedAssist() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_ALIGN_ASSIST");
  return enabled;
}

inline bool useSparsePvSharedValues() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_SHARED_VALUES");
  return enabled;
}

inline bool useSparsePvBitmapSharedValues() noexcept {
  static const bool enabled = sparsePvOption("MLLM_HMX_SME_PV_BITMAP_SHARED_VALUES");
  return enabled;
}

// The panel is key-major, with sixteen FP16 row weights per key.
void sparsePvRowsSme(const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count,
                     float* output, bool ordered_keys = false) noexcept;
void sparsePvRowsSmeRetirePanel(const __fp16* values, const int32_t* keys, __fp16* panel, int32_t key_count,
                                float* output, bool ordered_keys = false) noexcept;

// Each key carries two query-block membership bits in the upper halfword.
void sparsePvSharedRowsSme(const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count,
                           float* output) noexcept;

}  // namespace mllm::cpu::shadownpu
#endif

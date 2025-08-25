// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/cast_types.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void int8_2_fp16(const mllm_int8_t* src, mllm_fp16_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 8, thread_count)
    int remain = len - i;
    if (remain >= 8) {
      int8x8_t v8_src = vld1_s8(src + i);
      int16x8_t v16_src = vmovl_s8(v8_src);
      float32x4_t vf32_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_src)));
      float32x4_t vf32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_src)));
      vst1_f16(dst + i, vcvt_f16_f32(vf32_low));
      vst1_f16(dst + i + 4, vcvt_f16_f32(vf32_high));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 8) {
      int remain = len - i;
      if (remain >= 8) {
        int8x8_t v8_src = vld1_s8(src + i);
        int16x8_t v16_src = vmovl_s8(v8_src);
        float32x4_t vf32_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_src)));
        float32x4_t vf32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_src)));
        vst1_f16(dst + i, vcvt_f16_f32(vf32_low));
        vst1_f16(dst + i + 4, vcvt_f16_f32(vf32_high));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
      }
    }
  }
}

void int32_2_fp16(const mllm_int32_t* src, mllm_fp16_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 4, thread_count)
    int remain = len - i;
    if (remain >= 4) {
      int32x4_t v32_src = vld1q_s32(src + i);
      float32x4_t vf32 = vcvtq_f32_s32(v32_src);
      vst1_f16(dst + i, vcvt_f16_f32(vf32));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 4) {
      int remain = len - i;
      if (remain >= 4) {
        int32x4_t v32_src = vld1q_s32(src + i);
        float32x4_t vf32 = vcvtq_f32_s32(v32_src);
        vst1_f16(dst + i, vcvt_f16_f32(vf32));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
      }
    }
  }
}

void int8_2_fp32(const mllm_int8_t* src, mllm_fp32_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 16, thread_count)
    int remain = len - i;
    if (remain >= 16) {
      int8x16_t v8_src = vld1q_s8(src + i);

      int16x8_t v16_low = vmovl_s8(vget_low_s8(v8_src));
      int16x8_t v16_high = vmovl_s8(vget_high_s8(v8_src));

      int32x4_t v32_ll = vmovl_s16(vget_low_s16(v16_low));
      int32x4_t v32_lh = vmovl_s16(vget_high_s16(v16_low));
      int32x4_t v32_hl = vmovl_s16(vget_low_s16(v16_high));
      int32x4_t v32_hh = vmovl_s16(vget_high_s16(v16_high));

      vst1q_f32(dst + i, vcvtq_f32_s32(v32_ll));
      vst1q_f32(dst + i + 4, vcvtq_f32_s32(v32_lh));
      vst1q_f32(dst + i + 8, vcvtq_f32_s32(v32_hl));
      vst1q_f32(dst + i + 12, vcvtq_f32_s32(v32_hh));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 16) {
      int remain = len - i;
      if (remain >= 16) {
        int8x16_t v8_src = vld1q_s8(src + i);

        int16x8_t v16_low = vmovl_s8(vget_low_s8(v8_src));
        int16x8_t v16_high = vmovl_s8(vget_high_s8(v8_src));

        int32x4_t v32_ll = vmovl_s16(vget_low_s16(v16_low));
        int32x4_t v32_lh = vmovl_s16(vget_high_s16(v16_low));
        int32x4_t v32_hl = vmovl_s16(vget_low_s16(v16_high));
        int32x4_t v32_hh = vmovl_s16(vget_high_s16(v16_high));

        vst1q_f32(dst + i, vcvtq_f32_s32(v32_ll));
        vst1q_f32(dst + i + 4, vcvtq_f32_s32(v32_lh));
        vst1q_f32(dst + i + 8, vcvtq_f32_s32(v32_hl));
        vst1q_f32(dst + i + 12, vcvtq_f32_s32(v32_hh));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
      }
    }
  }
}

void int32_2_fp32(const mllm_int32_t* src, mllm_fp32_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 4, thread_count)
    int remain = len - i;
    if (remain >= 4) {
      int32x4_t v32_src = vld1q_s32(src + i);
      vst1q_f32(dst + i, vcvtq_f32_s32(v32_src));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 4) {
      int remain = len - i;
      if (remain >= 4) {
        int32x4_t v32_src = vld1q_s32(src + i);
        vst1q_f32(dst + i, vcvtq_f32_s32(v32_src));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
      }
    }
  }
}

void fp32_2_fp16(const mllm_fp32_t* src, mllm_fp16_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 8, thread_count)
    int remain = len - i;
    if (remain >= 8) {
      float32x4_t vf32_0 = vld1q_f32(src + i);
      float32x4_t vf32_1 = vld1q_f32(src + i + 4);
      vst1_f16(dst + i, vcvt_f16_f32(vf32_0));
      vst1_f16(dst + i + 4, vcvt_f16_f32(vf32_1));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 8) {
      int remain = len - i;
      if (remain >= 8) {
        float32x4_t vf32_0 = vld1q_f32(src + i);
        float32x4_t vf32_1 = vld1q_f32(src + i + 4);
        vst1_f16(dst + i, vcvt_f16_f32(vf32_0));
        vst1_f16(dst + i + 4, vcvt_f16_f32(vf32_1));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp16_t)src[j]; }
      }
    }
  }
}

void fp16_2_fp32(const mllm_fp16_t* src, mllm_fp32_t* dst, int len, int thread_count) {
  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, len, 8, thread_count)
    int remain = len - i;
    if (remain >= 8) {
      float16x8_t vf16 = vld1q_f16(src + i);
      vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(vf16)));
      vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(vf16)));
    } else {
      for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT();
  } else {
    for (int i = 0; i < len; i += 8) {
      int remain = len - i;
      if (remain >= 8) {
        float16x8_t vf16 = vld1q_f16(src + i);
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(vf16)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(vf16)));
      } else {
        for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
      }
    }
  }
}
}  // namespace mllm::cpu::arm

#endif

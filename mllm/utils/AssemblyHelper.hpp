// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// clang-format off

#ifdef _MSC_VER

// windows
#define MLLM_CPU_ASM_CODE_FRAGMENT AREA |.text|, CODE, READONLY, ALIGN=4
#define MLLM_CPU_ASM_LABEL(__mllm_label) |__mllm_label|
#define MLLM_CPU_ASM_EXPORT(__mllm_label) global __mllm_label
#define MLLM_CPU_ASM_FOOTER end
#define MLLM_CPU_ASM_HARD_CODE_INST(__mllm_num) dcd __mllm_num
#define MLLM_CPU_ASM_TARGET(__mllm_label, __mllm_direction) |__mllm_label|
#define MLLM_CPU_FUNCTION(__mllm_label) |__mllm_label|

#else  /// _MSC_VER

// linux
#define MLLM_CPU_ASM_CODE_FRAGMENT .text
#define MLLM_CPU_ASM_LABEL(__mllm_label) __mllm_label:
#define MLLM_CPU_ASM_TARGET(__mllm_label, __mllm_direction) __mllm_label##__mllm_direction
#define MLLM_CPU_ASM_FUNCTION(__mllm_label) __mllm_label:
#define MLLM_CPU_ASM_EXPORT(__mllm_label)  \
    .global __mllm_label;                  \
    .type __mllm_label, %function
#define MLLM_CPU_ASM_FOOTER
#define MLLM_CPU_ASM_HARD_CODE_INST(__mllm_num) .inst __mllm_num

#endif  ///_MSC_VER

// GPU Assembly Helper.
// CUDA / HIP

// clang-format on
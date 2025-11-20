// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

// Supports:
//  matmul_clamp_f32_qsi8d32p_qai4c32p
//  matmul_clamp_f32_qsi8d32p_qsi4c32p

// TERM is same with kleidiai:
// see https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/README.md
// f32 -  Floating-point 32-bit
// q : Quantized
// s :  Symmetric
// a : Asymmetric
// i : Signed integer
// u : Unsigned integer
// 4 : 4-bit Quantized
// 8 : 8-bit Quantized
// dx : Per dimension quantization
// cx : Per channel quantization
// c32 : Per block quantization, with block length multiple of 32 scale
// f16 : Scale factors are stores as floating-point 16-bit
// p : Matrix is packed
//
// e.g.:
// qsi4cxp :
//      qs - Quantized symmetric
//      i4 - Signed Integer 4-bit
//      cx - Per channel quantized
//      p - packed
// Some other examples :
//      s16s0 - Packing order of data is interleaved
//      s1s0 - Packing order of data is sequential
//      fp16 - Floating-point 16-bit data type

#include <arm_neon.h>
#include <cstdint>
#include <unordered_map>

// for matmul_clamp_f32_qsi8d32p_qai4c32p
// #include "kai_matmul_clamp_fp32"

namespace mllm::cpu::arm {}

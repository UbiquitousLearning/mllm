# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import torch
import numpy as np

"""
//===----------------------------------------------------------------------===//
// MLLM Types Enum
//
// The ID of each datatype is highly correlated with the mllm model file.
//
// If you want to add new datatype, it's restrict to add at the tail of this enum.
//===----------------------------------------------------------------------===//
enum DataTypes : int32_t {
  // MLLM V1 Datatypes.
  kFloat32 = 0,
  kFloat16 = 1,
  kGGUF_Q4_0 = 2,
  kGGUF_Q4_1 = 3,
  kGGUF_Q8_0 = 8,
  kGGUF_Q8_1 = 9,
  kGGUF_Q8_Pertensor = 10,
  kGGUF_Q4_K = 12,
  kGGUF_Q6_K = 14,
  kGGUF_Q8_K = 15,
  kInt8 = 16,
  kInt16 = 17,
  kInt32 = 18,
  kGGUF_Q4_0_4_4 = 19,
  kGGUF_Q4_0_4_8 = 20,
  kGGUF_Q4_0_8_8 = 21,
  kGGUF_Q8_0_4_4 = 22,
  kGGUF_Q3_K = 23,
  kGGUF_Q2_K = 24,
  kGGUF_Q1_K = 25,
  kGGUF_IQ2_XXS = 26,
  kGGUF_IQ2_XS = 27,
  kGGUF_IQ1_S = 28,
  kGGUF_IQ1_M = 29,
  kGGUF_IQ2_S = 30,

  // 31-127 is left for other GGUF Datatypes.

  // MLLM V2 Datatypes.
  kBFloat16 = 128,
  kUInt8 = 129,
  kUInt16 = 130,
  kUInt32 = 131,
  kInt64 = 132,
  kUInt64 = 133,

  // Byte is used to store general data. Such as
  // 1. Special packed kleidiai weight and bias
  // 2. Your customized quantization method.
  kByte = 134,
  kMXFP4 = 135,

  // complex dtypes for STFT and other ops
  kComplexFloat32 = 201,
  kComplexFloat64 = 202,
};
"""

MLLM_TYPE_MAPPING = {
    # PyTorch data type mappings
    torch.float32: 0,          # kFloat32
    torch.float16: 1,          # kFloat16
    torch.bfloat16: 128,       # kBFloat16
    torch.int8: 16,            # kInt8
    torch.int16: 17,           # kInt16
    torch.int32: 18,           # kInt32
    torch.int64: 132,          # kInt64
    torch.uint8: 129,          # kUInt8
    torch.bool: 129,           # kUInt8 (Boolean type in PyTorch is usually represented as uint8)
    # Quantized type mappings
    torch.qint8: 16,           # kInt8
    torch.quint8: 129,         # kUInt8
    torch.qint32: 18,          # kInt32
    
    # NumPy data type mappings
    np.float32: 0,             # kFloat32
    np.float16: 1,             # kFloat16
    np.int8: 16,               # kInt8
    np.int16: 17,              # kInt16
    np.int32: 18,              # kInt32
    np.int64: 132,             # kInt64
    np.uint8: 129,             # kUInt8
    np.bool_: 129,             # kUInt8 (Boolean type in NumPy)
    np.complex64: 201,         # kComplexFloat32
    np.complex128: 202,        # kComplexFloat64
}

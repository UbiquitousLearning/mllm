// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

size_t lanesOfType(DataTypes dtype) {
#define CASE(T)                          \
  case T: {                              \
    return MllmDataTypeInfo<T>::lanes(); \
  }
  switch (dtype) {
    CASE(kFloat32)
    CASE(kFloat16)
    CASE(kGGUF_Q4_0)
    // CASE(kGGUF_Q4_1) There is no such type right now.
    CASE(kGGUF_Q8_0)
    // CASE(kGGUF_Q8_1) There is no such type right now.
    CASE(kGGUF_Q8_Pertensor)
    CASE(kGGUF_Q4_K)
    CASE(kGGUF_Q6_K)
    CASE(kGGUF_Q8_K)
    CASE(kInt8)
    CASE(kInt16)
    CASE(kInt32)
    CASE(kGGUF_Q4_0_4_4)
    CASE(kGGUF_Q4_0_4_8)
    //   CASE(kGGUF_Q4_0_8_8) There is no such type right now.
    CASE(kGGUF_Q8_0_4_4)
    CASE(kGGUF_Q3_K)
    CASE(kGGUF_Q2_K)
    // CASE(kGGUF_Q1_K) There is no such type right now.
    CASE(kGGUF_IQ2_XXS)
    // CASE(kGGUF_IQ2_XS) There is no such type right now.
    // CASE(kGGUF_IQ1_S) There is no such type right now.
    // CASE(kGGUF_IQ1_M) There is no such type right now.
    // CASE(kGGUF_IQ2_S) There is no such type right now.
    // CASE(kBFloat16)  There is no such type right now.
    CASE(kUInt8)
    CASE(kUInt16)
    CASE(kUInt32)
    CASE(kInt64)
    CASE(kUInt64)
    CASE(kMXFP4)
    CASE(kComplexFloat32)
    CASE(kComplexFloat64)
    case kByte: return MllmDataTypeInfo<kUInt8>::lanes();
    default: NYI("Unknown data type");
  }
  return 1;
#undef CASE
}

size_t bytesOfType(DataTypes dtype) {
#define CASE(T)                          \
  case T: {                              \
    return MllmDataTypeInfo<T>::bytes(); \
  }
  switch (dtype) {
    CASE(kFloat32)
    CASE(kFloat16)
    CASE(kGGUF_Q4_0)
    // CASE(kGGUF_Q4_1) There is no such type right now.
    CASE(kGGUF_Q8_0)
    // CASE(kGGUF_Q8_1) There is no such type right now.
    CASE(kGGUF_Q8_Pertensor)
    CASE(kGGUF_Q4_K)
    CASE(kGGUF_Q6_K)
    CASE(kGGUF_Q8_K)
    CASE(kInt8)
    CASE(kInt16)
    CASE(kInt32)
    CASE(kGGUF_Q4_0_4_4)
    CASE(kGGUF_Q4_0_4_8)
    //   CASE(kGGUF_Q4_0_8_8) There is no such type right now.
    CASE(kGGUF_Q8_0_4_4)
    CASE(kGGUF_Q3_K)
    CASE(kGGUF_Q2_K)
    // CASE(kGGUF_Q1_K) There is no such type right now.
    CASE(kGGUF_IQ2_XXS)
    // CASE(kGGUF_IQ2_XS) There is no such type right now.
    // CASE(kGGUF_IQ1_S) There is no such type right now.
    // CASE(kGGUF_IQ1_M) There is no such type right now.
    // CASE(kGGUF_IQ2_S) There is no such type right now.
    // CASE(kBFloat16)  There is no such type right now.
    CASE(kUInt8)
    CASE(kUInt16)
    CASE(kUInt32)
    CASE(kInt64)
    CASE(kUInt64)
    CASE(kMXFP4)
    CASE(kComplexFloat32)
    CASE(kComplexFloat64)
    case kByte: return MllmDataTypeInfo<kUInt8>::bytes();
    default: NYI("Unknown data type");
  }
  return 1;
#undef CASE
}

std::string nameOfType(DataTypes dtype) {
#define CASE(T)                         \
  case T: {                             \
    return MllmDataTypeInfo<T>::name(); \
  }
  switch (dtype) {
    CASE(kFloat32)
    CASE(kFloat16)
    CASE(kGGUF_Q4_0)
    // CASE(kGGUF_Q4_1) There is no such type right now.
    CASE(kGGUF_Q8_0)
    // CASE(kGGUF_Q8_1) There is no such type right now.
    CASE(kGGUF_Q8_Pertensor)
    CASE(kGGUF_Q4_K)
    CASE(kGGUF_Q6_K)
    CASE(kGGUF_Q8_K)
    CASE(kInt8)
    CASE(kInt16)
    CASE(kInt32)
    CASE(kGGUF_Q4_0_4_4)
    CASE(kGGUF_Q4_0_4_8)
    //   CASE(kGGUF_Q4_0_8_8) There is no such type right now.
    CASE(kGGUF_Q8_0_4_4)
    CASE(kGGUF_Q3_K)
    CASE(kGGUF_Q2_K)
    // CASE(kGGUF_Q1_K) There is no such type right now.
    CASE(kGGUF_IQ2_XXS)
    // CASE(kGGUF_IQ2_XS) There is no such type right now.
    // CASE(kGGUF_IQ1_S) There is no such type right now.
    // CASE(kGGUF_IQ1_M) There is no such type right now.
    // CASE(kGGUF_IQ2_S) There is no such type right now.
    // CASE(kBFloat16)  There is no such type right now.
    CASE(kUInt8)
    CASE(kUInt16)
    CASE(kUInt32)
    CASE(kInt64)
    CASE(kUInt64)
    CASE(kMXFP4)
    CASE(kComplexFloat32)
    CASE(kComplexFloat64)
    case kByte: return MllmDataTypeInfo<kUInt8>::name();
    default: NYI("Unknown data type");
  }
  return "Unknow";
#undef CASE
}

}  // namespace mllm

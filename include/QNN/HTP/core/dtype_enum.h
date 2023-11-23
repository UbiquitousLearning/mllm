//==============================================================================
//
// Copyright (c) 2020, 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DTYPE_ENUM_H
#define DTYPE_ENUM_H 1

#include <stdint.h>
#include "weak_linkage.h"

typedef int NN_INT32_T;

typedef unsigned int NN_UINT32_T;

#ifdef __cplusplus
enum class DType : uint32_t {
#else
enum DType {
#endif
    UNKNOWN = 0,
    QUInt8 = 1,
    QUInt16 = 2,
    QInt16 = 3,
    Float32 = 4,
    Int32 = 5,
    QInt32 = 6,
    QInt8 = 7,
    Float16 = 8,
    ZZ_LAST_DTYPE,
    None = 254, //  for output of OpDef representing null output. Not for use by external API.
    Multi = 255 //  for output of OpDef representing multiple outputs. Not for use by external API.
};

#define DTYPE_NAMETABLE_INIT                                                                                           \
    {                                                                                                                  \
        "UNKNOWN", "QUInt8", "QUInt16", "QInt16", "Float32", "Int32", "QInt32", "QInt8", "Float16"                     \
    }

#ifdef __cplusplus
namespace hnnx {
extern "C" {
#endif
API_FUNC_EXPORT char const *DType_name(enum DType);
#ifdef __cplusplus
} // extern C
// this is intended to be only referenced once (inside DType_name, in graph.cc)
// and is placed here for easy maintenance
inline char const *DType_name_inline(DType d)
{
    switch (d) {
    default:
        return "Bad_DType";
    case DType::UNKNOWN:
        return "UNKNOWN";
    case DType::QUInt8:
        return "QUint8";
    case DType::QUInt16:
        return "QUInt16";
    case DType::QInt16:
        return "QInt16";
    case DType::Float16:
        return "Float16";
    case DType::Float32:
        return "Float32";
    case DType::Int32:
        return "Int32";
    case DType::QInt32:
        return "QInt32";
    case DType::QInt8:
        return "QInt8";
    case DType::Multi:
        return "Multi";
    }
}

} // namespace hnnx
#endif

#endif

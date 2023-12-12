//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DTYPE_H
#define DTYPE_H 1

#include <cstdint>
#include <type_traits>
#include "dtype_enum.h"
#include "float16.h"
#include "macros_attribute.h"

template <DType DT> struct dtype_traits {
};

template <> struct dtype_traits<DType::QUInt8> {
    typedef uint8_t element_type;
    typedef uint8_t storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = true;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 0;
};

template <> struct dtype_traits<DType::QUInt16> {
    typedef uint16_t element_type;
    typedef uint16_t storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = true;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 0;
};

template <> struct dtype_traits<DType::QInt16> {
    typedef int16_t element_type;
    typedef uint16_t storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = true;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 0x8000;
};
template <> struct dtype_traits<DType::Float16> {
    typedef Float16 element_type;
    typedef uint16_t storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = false;
    static const bool is_float = true;
    // -inf pattern (but, if hvx flt16 are used, maybe
    // it should be 0xFFFF?
    static const storage_type minus_inf_code = 0xFC00;
};
template <> struct dtype_traits<DType::Float32> {
    typedef float element_type;
    typedef NN_UINT32_T storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = false;
    static const bool is_float = true;
    // -inf pattern (but, if hvx flt16 are used, maybe
    // it should be 0xFFFFFFFF?
    static const storage_type minus_inf_code = 0xFF800000;
};
template <> struct dtype_traits<DType::Int32> {
    typedef NN_INT32_T element_type;
    typedef NN_UINT32_T storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = false;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 1u << 31;
};
template <> struct dtype_traits<DType::QInt32> {
    typedef NN_INT32_T element_type;
    typedef NN_UINT32_T storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = true;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 1u << 31;
};
template <> struct dtype_traits<DType::QInt8> {
    typedef int8_t element_type;
    typedef uint8_t storage_type;
    static const int element_size = sizeof(element_type);
    static const bool is_quant = true;
    static const bool is_float = false;
    static const storage_type minus_inf_code = 128;
};

// 'runtime' attributes
// E.g. Dtype_info(d).elbytes gives the element size.
struct dtype_info {
    unsigned elbytes : 8;
    unsigned is_quant : 1;
    unsigned is_float : 1;
    unsigned is_signed : 1;
};

API_EXPORT dtype_info DType_info(DType d); // in graph.cc

namespace hnnx {
namespace dtype_private {
template <DType DT> dtype_info constexpr inline dtype_info_for()
{
    typedef dtype_traits<DT> traits;
    return dtype_info{
            sizeof(typename traits::element_type), //elbytes
            traits::is_quant, //is_quant
            traits::is_float, //is_float
            (std::is_signed<typename traits::element_type>::value ? 1 : 0) //is_signed
    };
}
template <> dtype_info constexpr inline dtype_info_for<DType::UNKNOWN>()
{
    return dtype_info{
            0, //elbytes
            0, //is_quant
            0, //is_float
            0 //is_signed
    };
}
// this is intended to be only referenced once (inside DType_info, in graph.cc)
// and is placed here for easy maintenance

inline constexpr dtype_info DType_info_inline(DType d)
{
    switch (d) {
    case DType::QUInt8:
        return dtype_info_for<DType::QUInt8>();
    case DType::QUInt16:
        return dtype_info_for<DType::QUInt16>();
    case DType::QInt16:
        return dtype_info_for<DType::QInt16>();
    case DType::Float16:
        return dtype_info_for<DType::Float16>();
    case DType::Float32:
        return dtype_info_for<DType::Float32>();
    case DType::Int32:
        return dtype_info_for<DType::Int32>();
    case DType::QInt32:
        return dtype_info_for<DType::QInt32>();
    case DType::QInt8:
        return dtype_info_for<DType::QInt8>();
    default:
        return dtype_info_for<DType::UNKNOWN>();
    }
}
} //namespace dtype_private
} // namespace hnnx

/* Maybe instead of functions these should be template constexpr variables? */

template <typename TINTERFACE> constexpr DType dtype_of_type()
{
    return DType::UNKNOWN;
}

#endif

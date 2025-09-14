// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// We borrow ideas from TVM C API and its ffi implementation.
//
// Ref: https://github.com/apache/tvm
//
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

// C API of MLLM FFI.

#include <stdint.h>  // NOLINT

// dlpack for torch and numpy, etc compatible
#include <dlpack/dlpack.h>

// mllm c++
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Macros to do weak linking
#ifdef _MSC_VER
#define MLLM_FFI_WEAK __declspec(selectany)
#else
#define MLLM_FFI_WEAK __attribute__((weak))
#endif

// Defines two macros
// MLLM_FFI_DLL: marks the function as a DLL export/import
//              depending on whether MLLM_FFI_EXPORTS is defined
// MLLM_FFI_DLL_EXPORT: always marks the function as a DLL export
#if !defined(MLLM_FFI_DLL) && defined(__EMSCRIPTEN__)
// For Web assembly.
#include <emscripten/emscripten.h>
#define MLLM_FFI_DLL EMSCRIPTEN_KEEPALIVE
#define MLLM_FFI_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(MLLM_FFI_DLL) && defined(_MSC_VER)
#ifdef MLLM_FFI_EXPORTS
#define MLLM_FFI_DLL __declspec(dllexport)
#else
#define MLLM_FFI_DLL __declspec(dllimport)
#endif
#define MLLM_FFI_DLL_EXPORT __declspec(dllexport)
#endif
#ifndef MLLM_FFI_DLL
#define MLLM_FFI_DLL __attribute__((visibility("default")))
#define MLLM_FFI_DLL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
enum MllmFFITypeID : int32_t {
#else
typedef enum {
#endif
  ///< To Capture any type that returned to MLLM
  kMllmFFIAny = -1,
  ///< None in py, nullptr in c++ and NULL in c, etc
  kMllmFFINone = 0,
  ///< POD Int value
  kMllmFFIInt = 1,
  ///< POD Float value
  kMllmFFIFloat = 2,
  ///< POD Bool value
  kMllmFFIBool = 3,
  ///< Opaque pointer object
  kMllmFFIOpaquePtr = 4,
  ///< MLLM Datatype
  kMllmFFIDatatype = 5,
  ///< MLLM Device
  kMllmFFIDevice = 6,
  ///< DLDataType
  kMllmFFIDLDataType = 7,
  ///< DLDevice
  kMllmFFIDLDevice = 8,
  ///< Tensor pointer
  kMllmFFITensorPtr = 9,
  ///< DLTensor pointer
  kMllmFFIDLTensorPtr = 10,
  ///< const char*
  kMllmFFIRawStr = 11,
  ///< Byte array
  kMllmFFIByteArrayPtr = 12,
  ///< R-value reference to ObjectRef
  kMllmFFIObjectRValueRef = 13,
  ///< Small string on stack
  kMllmFFISmallStr = 14,
  ///< Small bytes on stack
  kMllmFFISmallBytes = 15,
  ///< Start of statically defined objects.
  kMllmFFIStaticObject_Begin = 64,
  ///< Object, all objects starts with MllmFFIObject as its header. Will also add other fields
  kMllmFFIObject = 64,
  ///< String object, layout = { MllmFFIObject, MllmFFIByteArray, ... }
  kMllmFFIStr = 65,
  ///< Bytes object, layout = { MllmFFIObject, MllmFFIByteArray, ... }
  kMllmFFIBytes = 66,
  ///< Error object.
  kMllmFFIError = 67,
  ///< Function object.
  kMllmFFIFunction = 68,
  ///< Shape object, layout = { MllmFFIObject, { const int32_t*, size_t }, ... }
  kMllmFFIShape = 69,
  ///< Tensor object, layout = { MllmFFIObject, Tensor, ... }
  kMllmFFITensor = 70,
  ///< DLTensor object, layout = { MllmFFIObject, DLTensor, ... }
  kMllmFFIDLTensor = 71,
  ///< Array object.
  kMllmFFIArray = 72,

  ///< Container below

  ///< Map
  kMllmFFIMap = 73,

  ///< Runtime dynamic loaded module object.
  kMllmFFIModule = 74,
  /*!
   * \brief Opaque python object.
   *
   * This is a special type index to indicate we are storing an opaque PyObject.
   * Such object may interact with callback functions that are registered to support
   * python-related operations.
   *
   * We only translate the objects that we do not recognize into this type index.
   *
   * \sa MllmFFIObjectCreateOpaque
   */
  kMllmFFIOpaquePyObject = 75,
  kMllmFFIOpaqueGolangObject = 76,

  kMllmFFIStaticObject_End,

  ///< Dynamic Boxed: [kMllmFFIDynObjectBegin, +oo) below:

  ///< Start of type indices that are allocated at runtime.
  kMllmFFIDynObject_Begin = 128
#ifdef __cplusplus
};
#else
} MllmFFITypeID;
#endif

typedef struct {              // NOLINT
  int32_t type_index;         ///< Identify what this object type is.
  uint32_t weak_ref_count;    ///< Weak reference counter of the object
  uint64_t strong_ref_count;  ///< Strong reference counter of the object
  union {
    /*!
     * \brief Deleter to be invoked when strong reference counter goes to zero.
     * \param self The self object handle.
     * \param flags The flags to indicate deletion behavior.
     */
    void (*deleter)(void* self, int flags);
    /*!
     * \brief auxilary field to MllmFFIObject is always 8 bytes aligned.
     * \note This helps us to ensure cross platform compatibility.
     */
    int64_t __ensure_align;
  };
} MllmFFIObject;

typedef struct {            // NOLINT
  int32_t type_index;       ///< 4B For runtime type index
  union {                   // 4B
    uint32_t zero_padding;  //  padding, must set to zero for values other than small string.
    /*!
     * \brief Length of small string, with a max value of 7.
     *
     * We keep small str to start at next 4 bytes to ensure alignment
     * when accessing the small str content.
     */
    uint32_t small_str_len;
  };
  union {                             // 8B
    int64_t v_int64;                  // int64
    double v_float64;                 // double
    void* v_ptr;                      // typeless pointer
    const char* v_c_str;              // Raw C-String
    MllmFFIObject* v_obj;             // Obj
    mllm::DataTypes v_mllm_dtype;     // mllm info
    mllm::DeviceTypes v_mllm_device;  // mllm info
    DLDataType v_dlpack_dtype;        // dlpack info
    DLDevice v_dlpack_device;         // dlpack info
    char v_bytes[8];                  // Small strings
    char32_t v_char32[2];             // small UCS4 string and Unicode
    uint64_t v_uint64;                // uint64 repr mainly used for hashing
  };
} MllmFFIAny;
static_assert(sizeof(MllmFFIAny) == 16, "MllmFFIAny size must be 16 bytes");

#ifdef __cplusplus
}  // extern "C"
#endif

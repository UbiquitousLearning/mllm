// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// We borrow ideas from Mllm C API and its ffi implementation.

#pragma once

// C API of MLLM FFI.

// #include <dlpack/dlpack.h>
#include <stdint.h>  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

// TODO

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

// MllmFFIAny

#ifdef __cplusplus
}  // extern "C"
#endif

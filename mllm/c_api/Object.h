// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#ifndef MLLM_C_API_OBJECT_H_
#define MLLM_C_API_OBJECT_H_

#include <stdint.h>  // NOLINT
#include <stddef.h>  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
enum MllmCType : int32_t {
#else
typedef enum {
#endif

  // POD Values
  kPod_Start = 0,
  kInt = 1,
  kFloat = 2,
  kBool = 3,
  kPod_End = 64,

  // Mllm related types
  kMllm_Start = 65,
  kTensor = 66,
  kModule = 67,
  kRetCode = 68,
  kMllm_End = 256,

  // Builtin Containers
  kBuiltinContainer_Start = 257,
  kBuiltinContainerString = 258,
  kBuiltinContainerMap = 259,
  kBuiltinContainerList = 260,
  kBuiltinContainer_End = 512,

#ifdef __cplusplus
};
#else
} MllmCType;
#endif

typedef struct {           // NOLINT
  uint32_t type_id;        // 4B
  int64_t strong_ref_ptr;  // 8B
  uint32_t weak_ref_ptr;   // 4B
} MllmCObject;

typedef struct {  // NOLINT
  char* data;
  size_t size;
} MllmCByteArrayObject;

typedef struct {     // NOLINT
  uint32_t type_id;  // 4B
  union {            // 8B
    int64_t v_int64;
    double v_fp64;
    int32_t v_bool;
    int32_t v_return_code;
    MllmCObject* v_object;
    void* v_bare_ptr;
  };
} MllmCAny;

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  //! MLLM_C_API_OBJECT_H_

//==============================================================================
//
//  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"

namespace qnn_wrapper_api {

// macro utils

// Enables FILE[LINE]: FMT for VALIDATE macro
#ifdef QNN_ENABLE_DEBUG

#define PRINTF(fmt, ...)                    \
  do {                                      \
    printf("%s[%d]: ", __FILE__, __LINE__); \
    printf((fmt), ##__VA_ARGS__);           \
  } while (0)

#else

#define PRINTF(fmt, ...)          \
  do {                            \
    printf((fmt), ##__VA_ARGS__); \
  } while (0)

#endif

#ifdef QNN_ENABLE_DEBUG
#define PRINT_DEBUG(fmt, ...)     \
  do {                            \
    printf("[ DEBUG ] ");         \
    PRINTF((fmt), ##__VA_ARGS__); \
  } while (0)
#else
#define PRINT_DEBUG(fmt, ...)
#endif

// Enables ERROR tag for errors
#define PRINT_ERROR(fmt, ...)     \
  do {                            \
    printf("[ ERROR ] ");         \
    PRINTF((fmt), ##__VA_ARGS__); \
  } while (0)

// Enables WARNING tag for errors
#define PRINT_WARNING(fmt, ...)   \
  do {                            \
    printf("[ WARNING ] ");       \
    PRINTF((fmt), ##__VA_ARGS__); \
  } while (0)

// Enables INFO tag for errors
#define PRINT_INFO(fmt, ...)      \
  do {                            \
    printf("[ INFO ] ");          \
    PRINTF((fmt), ##__VA_ARGS__); \
  } while (0)

#define STRINGFY(str)      str
#define STRINGFYVALUE(str) STRINGFY(str)

// Ensures ModelError_t returning functions return MODEL_NO_ERROR
// retStatus should be set to MODEL_NO_ERROR before passing to macro
#define VALIDATE(value, retStatus)                                                               \
  do {                                                                                           \
    retStatus = value;                                                                           \
    if (retStatus != qnn_wrapper_api::MODEL_NO_ERROR) {                                          \
      PRINT_ERROR(                                                                               \
          "%s expected MODEL_NO_ERROR, got %s\n", #value, getModelErrorName(retStatus).c_str()); \
      return retStatus;                                                                          \
    }                                                                                            \
  } while (0)

// macros for retrieving binary data
#define BINVARSTART(NAME)                                         \
  ({                                                              \
    extern const uint8_t _binary_obj_binary_##NAME##_raw_start[]; \
    (void *)_binary_obj_binary_##NAME##_raw_start;                \
  })
#define BINVAREND(NAME)                                         \
  ({                                                            \
    extern const uint8_t _binary_obj_binary_##NAME##_raw_end[]; \
    (void *)_binary_obj_binary_##NAME##_raw_end;                \
  })
#define BINLEN(NAME)                                                                             \
  ({                                                                                             \
    extern const uint8_t _binary_obj_binary_##NAME##_raw_start[];                                \
    extern const uint8_t _binary_obj_binary_##NAME##_raw_end[];                                  \
    (uint32_t)((_binary_obj_binary_##NAME##_raw_end) - (_binary_obj_binary_##NAME##_raw_start)); \
  })

typedef enum ModelError {
  MODEL_NO_ERROR               = 0,
  MODEL_TENSOR_ERROR           = 1,
  MODEL_PARAMS_ERROR           = 2,
  MODEL_NODES_ERROR            = 3,
  MODEL_GRAPH_ERROR            = 4,
  MODEL_CONTEXT_ERROR          = 5,
  MODEL_GENERATION_ERROR       = 6,
  MODEL_SETUP_ERROR            = 7,
  MODEL_INVALID_ARGUMENT_ERROR = 8,
  MODEL_FILE_ERROR             = 9,
  MODEL_MEMORY_ALLOCATE_ERROR  = 10,
  // Value selected to ensure 32 bits.
  MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
} ModelError_t;

typedef struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
} GraphInfo_t;
typedef GraphInfo_t *GraphInfoPtr_t;

typedef struct GraphConfigInfo {
  char *graphName;
  const QnnGraph_Config_t **graphConfigs;
} GraphConfigInfo_t;

/**
 * @brief Frees all memory allocated tensor attributes.
 *
 * @param[in] tensor Qnn_Tensor_t object to free
 *
 * @return Error code
 */
ModelError_t freeQnnTensor(Qnn_Tensor_t &tensor);

/**
 * @brief Loops through and frees all memory allocated tensor attributes for each tensor
 * object.
 *
 * @param[in] tensors array of tensor objects to free
 *
 * @param[in] numTensors length of the above tensors array
 *
 * @return Error code
 */
ModelError_t freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors);

/**
 * @brief A helper function to free memory malloced for communicating the Graph for a model(s)
 *
 * @param[in] graphsInfo Pointer pointing to location of graph objects
 *
 * @param[in] numGraphs The number of graph objects the above pointer is pointing to
 *
 * @return Error code
 *
 */
ModelError_t freeGraphsInfo(GraphInfoPtr_t **graphsInfo, uint32_t numGraphs);

}  // namespace qnn_wrapper_api

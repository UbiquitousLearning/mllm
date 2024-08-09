//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN CPU component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for CPU backend
 */

#ifndef QNN_CPU_GRAPH_H
#define QNN_CPU_GRAPH_H

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief This enum provides different CPU graph configuration
 *         options associated with QnnGraph
 */
typedef enum {
  QNN_CPU_GRAPH_CONFIG_OPTION_OP_DEBUG_CALLBACK = 1,
  QNN_CPU_GRAPH_CONFIG_OPTION_UNDEFINED         = 0x7fffffff
} QnnCpuGraph_ConfigOption_t;

/* @brief CallBack function pointer to be filled by user.
 *        This callback will be called after each op execution.
 *        Only outputTensor id and data buffer is valid, consumable.
 *        Memory is owned by BE which is valid throughout the callback.
 *        Client should not update any parameter and argument of opConfig.
 *        NULL tensor/buffer indicate invalid data buffer.
 */
typedef Qnn_ErrorHandle_t (*QnnCpuGraph_OpDebugCallback_t)(Qnn_OpConfig_t* opConfig,
                                                           void* callBackParam);

/* @brief Structure to be filled by user.
 *        This structure will have callback function and callback reference data.
 *        Memory is owned by BE which is valid throughout the callback.
 *        Client should not update any parameter and argument of opConfig.
 *        NULL callback function indicate no debug option.
 */
typedef struct {
  void* callBackParam;
  QnnCpuGraph_OpDebugCallback_t cpuGraphOpDebugCallback;
} QnnCpuGraph_OpDebug_t;

// clang-format off
/// QnnCpuGraph_OpDebug_t initializer macro
#define QNN_CPU_GRAPH_OP_DEBUG_INIT       \
  {                                       \
    NULL,    /*callBackParam*/            \
    NULL     /*cpuGraphOpDebugCallback*/  \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

/**
 * @brief        Structure describing the set of configurations supported by graph.
 *               Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *               Below is the map between QnnCpuGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+------------------------------------------+------------------------------------+
 *               | #  | Config Option                            | Configuration Struct/value      |
 *               +====+==========================================+====================================+
 *               | 1  | QNN_CPU_GRAPH_CONFIG_DEBUG_CALLBACK      | QnnCpuGraph_OpDebug_t           |
 *               +----+------------------------------------------+------------------------------------+
 *               \endverbatim
 */
typedef struct {
  QnnCpuGraph_ConfigOption_t option;
  union UNNAMED {
    QnnCpuGraph_OpDebug_t cpuGraphOpDebug;
  };
} QnnCpuGraph_CustomConfig_t;

/// QnnCpuGraph_CustomConfig_t initializer macro
#define QNN_CPU_GRAPH_CUSTOM_CONFIG_INIT                      \
  {                                                           \
    QNN_CPU_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/           \
    {                                                         \
      QNN_CPU_GRAPH_OP_DEBUG_INIT /*cpuGraphOpDebugCallback*/ \
    }                                                         \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

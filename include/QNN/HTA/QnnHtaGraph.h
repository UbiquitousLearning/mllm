//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTA component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for HTA backend
 */

#ifndef QNN_HTA_GRAPH_H
#define QNN_HTA_GRAPH_H

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
 * @brief This enum provides different HTA graph optimization
 *         options that can be used to finalize the graph
 *         for optimum performance
 */
typedef enum QnnHtaGraph_OptimizationType {
  QNN_HTA_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD = 1,
  QNN_HTA_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES   = 2,
  QNN_HTA_GRAPH_OPTIMIZATION_TYPE_UNKNOWN            = 0x7fffffff
} QnnHtaGraph_OptimizationType_t;

/* @brief Struct describing the set of optimization type
 *        and the value associated with the optimization
 */
typedef struct QnnHtaGraph_OptimizationOption {
  QnnHtaGraph_OptimizationType_t type;
  float floatValue;
} QnnHtaGraph_OptimizationOption_t;

// clang-format off
/// QnnHtaGraph_OptimizationOption_t initializer macro
#define QNN_HTA_GRAPH_OPTIMIZATION_OPTION_INIT              \
  {                                                         \
    QNN_HTA_GRAPH_OPTIMIZATION_TYPE_UNKNOWN, /*type*/       \
    0.0f                                     /*floatValue*/ \
  }
// clang-format on

/**
 * @brief This enum provides different HTA graph configuration
 *         options associated with QnnGraph
 */
typedef enum QnnHtaGraph_ConfigOption {
  QNN_HTA_GRAPH_CONFIG_OPTION_OPTIMIZATION = 1,
  QNN_HTA_GRAPH_CONFIG_OPTION_PRIORITY     = 2,
  QNN_HTA_GRAPH_CONFIG_OPTION_UNKNOWN      = 0x7fffffff
} QnnHtaGraph_ConfigOption_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by graph.
 *               Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config
 *               Below is the Map between QnnHtaGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+------------------------------------------+------------------------------------+
 *               | #  | Config Option                            | Configuration Struct/value         |
 *               +====+==========================================+====================================+
 *               | 1  | QNN_HTA_GRAPH_CONFIG_OPTION_OPTIMIZATION | QnnHtaGraph_OptimizationOption_t   |
 *               +----+------------------------------------------+------------------------------------+
 *               | 2  | QNN_HTA_GRAPH_CONFIG_OPTION_PRIORITY     | Qnn_Priority_t                     |
 *               +----+------------------------------------------+------------------------------------+
 *               \endverbatim
 */
typedef struct {
  QnnHtaGraph_ConfigOption_t option;
  union {
    QnnHtaGraph_OptimizationOption_t optimizationOption;
    Qnn_Priority_t priority;
  };
} QnnHtaGraph_CustomConfig_t ;


/// QnnHtaGraph_CustomConfig_t initalizer macro
#define QNN_HTA_GRAPH_CUSTOM_CONFIG_INIT                            \
  {                                                                 \
    QNN_HTA_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/                 \
    {                                                               \
      QNN_HTA_GRAPH_OPTIMIZATION_OPTION_INIT /*optimizationOption*/ \
    }                                                               \
  }

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif

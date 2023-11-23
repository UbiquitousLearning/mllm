//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN DSP component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for DSP backend
 */

#ifndef QNN_DSP_GRAPH_H
#define QNN_DSP_GRAPH_H

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
 * @brief This enum provides different DSP graph optimization
 *        options that can be used to finalize the graph
 *        for optimum performance.
 */
typedef enum {
  QNN_DSP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         = 1,
  QNN_DSP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           = 2,
  QNN_DSP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG = 3,
  QNN_DSP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                = 4,
  QNN_DSP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN                    = 0x7fffffff
} QnnDspGraph_OptimizationType_t;

// clang-format off

/**
 * @brief Struct describing the set of optimization types
 *        and the values associated with each optimization type.
 *
 *        Below is the Map between QnnDspGraph_OptimizationType_t and allowable values:
 *
 *        \verbatim embed:rst:leading-asterisk
 *        +----+------------------------------------------------------------+-----------------------------------------------------------+
 *        | #  | OptimizationType option                                    | Allowable values                                          |
 *        +====+============================================================+===========================================================+
 *        | 1  | QNN_DSP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         | Reserved                                                  |
 *        +----+------------------------------------------------------------+-----------------------------------------------------------+
 *        | 2  | QNN_DSP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           | Reserved                                                  |
 *        +----+------------------------------------------------------------+-----------------------------------------------------------+
 *        | 3  | QNN_DSP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG | Defines the optimization strategy used by the HTP backend |
 *        |    |                                                            |                                                           |
 *        |    |                                                            |   1 = Faster preparation time, less optimal graph         |
 *        |    |                                                            |                                                           |
 *        |    |                                                            |   2 = More optimal graph but may take longer to prepare   |
 *        +----+------------------------------------------------------------+-----------------------------------------------------------+
 *        | 4  | QNN_DSP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                | Reserved                                                  |
 *        +----+------------------------------------------------------------+-----------------------------------------------------------+
 *        \endverbatim
 */
typedef struct {
  QnnDspGraph_OptimizationType_t type;
  float floatValue;
} QnnDspGraph_OptimizationOption_t;

/// QnnDspGraph_OptimizationOption_t initializer macro
#define QNN_DSP_GRAPH_OPTIMIZATION_OPTION_INIT              \
  {                                                         \
    QNN_DSP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN, /*type*/       \
    0.0f                                     /*floatValue*/ \
  }
// clang-format on

/**
 * @brief This enum provides different DSP graph configuration
 *        options associated with QnnGraph
 */
typedef enum {
  QNN_DSP_GRAPH_CONFIG_OPTION_OPTIMIZATION = 1,
  QNN_DSP_GRAPH_CONFIG_OPTION_ENCODING     = 2,
  QNN_DSP_GRAPH_CONFIG_OPTION_PRIORITY     = 3,
  QNN_DSP_GRAPH_CONFIG_OPTION_PRECISION    = 4,
  QNN_DSP_GRAPH_CONFIG_OPTION_UNKNOWN      = 0x7fffffff
} QnnDspGraph_ConfigOption_t;

typedef enum {
  QNN_DSP_GRAPH_ENCODING_DYNAMIC = 1,
  /** @deprecated
   */
  QNN_DSP_GRAPH_ENCOING_DYNAMIC = QNN_DSP_GRAPH_ENCODING_DYNAMIC,
  QNN_DSP_GRAPH_ENCODING_STATIC = 2,
  /** @deprecated
   */
  QNN_DSP_GRAPH_ENCOING_STATIC   = QNN_DSP_GRAPH_ENCODING_STATIC,
  QNN_DSP_GRAPH_ENCODING_UNKNOWN = 0x7fffffff,
  /** @deprecated
   */
  QNN_DSP_GRAPH_ENCOING_UNKNOW = QNN_DSP_GRAPH_ENCODING_UNKNOWN
} QnnDspGraph_Encoding_t;

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
 *               config.
 *
 *               Below is the Map between QnnDspGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+------------------------------------------+------------------------------------+
 *               | #  | Config Option                            | Configuration Struct/value         |
 *               +====+==========================================+====================================+
 *               | 1  | QNN_DSP_GRAPH_CONFIG_OPTION_OPTIMIZATION | QnnDspGraph_OptimizationOption_t   |
 *               +----+------------------------------------------+------------------------------------+
 *               | 2  | QNN_DSP_GRAPH_CONFIG_OPTION_ENCODING     | QnnDspGraph_Encoding_t             |
 *               +----+------------------------------------------+------------------------------------+
 *               | 3  | QNN_DSP_GRAPH_CONFIG_OPTION_PRECISION    | Qnn_Precision_t                    |
 *               +----+------------------------------------------+------------------------------------+
 *               | 4  | QNN_DSP_GRAPH_CONFIG_OPTION_PRIORITY     | Qnn_Priority_t                     |
 *               +----+------------------------------------------+------------------------------------+
 *               \endverbatim
 */
typedef struct {
  QnnDspGraph_ConfigOption_t option;
  union {
    QnnDspGraph_OptimizationOption_t optimizationOption;
    QnnDspGraph_Encoding_t encoding;
    Qnn_Priority_t priority;
    Qnn_Precision_t precision;
  };
} QnnDspGraph_CustomConfig_t;

// clang-format on
/// QnnDspGraph_CustomConfig_t initializer macro
#define QNN_DSP_GRAPH_CUSTOM_CONFIG_INIT                            \
  {                                                                 \
    QNN_DSP_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/                 \
    {                                                               \
      QNN_DSP_GRAPH_OPTIMIZATION_OPTION_INIT /*optimizationOption*/ \
    }                                                               \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

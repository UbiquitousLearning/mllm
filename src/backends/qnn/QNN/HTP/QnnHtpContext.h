//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN HTP component Context API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnContext.h for HTP backend
 */

#ifndef QNN_HTP_CONTEXT_H
#define QNN_HTP_CONTEXT_H

#include "QnnContext.h"

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
 * @brief This enum provides different HTP context configuration
 *        options associated with QnnContext
 */
typedef enum {
  QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED  = 1,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS = 2,
  QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET = 3,
  QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN                 = 0x7fffffff
} QnnHtpContext_ConfigOption_t;

typedef struct {
  // Handle referring to the first context associated to a group. When a new
  // group is to be registered, the following value must be 0.
  Qnn_ContextHandle_t firstGroupHandle;
  // Max spill-fill buffer to be allocated for the group of context in bytes.
  // The value that is passed during the registration of the first context to
  // a group is taken. Subsequent configuration of this value is disregarded.
  uint64_t maxSpillFillBuffer;
} QnnHtpContext_GroupRegistration_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by context.
 *               Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpContext_CustomConfig_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | #  | Config Option                                                       | Configuration Struct/value            |
 *               +====+=====================================================================+=======================================+
 *               | 1  | QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED                | bool                                  |
 *               +====+=====================================================================+=======================================+
 *               | 2  | QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS               | QnnHtpContext_GroupRegistration_t     |
 *               +====+=====================================================================+=======================================+
 *               | 3  | QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET               | uint64_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               \endverbatim
 */
typedef struct QnnHtpContext_CustomConfig {
  QnnHtpContext_ConfigOption_t option;
  union UNNAMED {
    // This field sets the weight sharing which is by default false
    bool weightSharingEnabled;
    QnnHtpContext_GroupRegistration_t groupRegistration;
    // - Init time may be impacted depending the value set below
    // - Value should be grather than 0 and less than or equal to the file size
    //    - If set to 0, the feature is not utilized
    //    - If set to greater than file size, min(fileSize, fileReadMemoryBudgetInMb) is used
    // - As an example, if value 2 is passed, it would translate to (2 * 1024 * 1024) bytes
    uint64_t fileReadMemoryBudgetInMb;
  };
} QnnHtpContext_CustomConfig_t;

/// QnnHtpContext_CustomConfig_t initializer macro
#define QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT            \
  {                                                   \
    QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN  /*option*/ \
    {                                                 \
      false                          /*weightsharing*/\
    }                                                 \
  }

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif

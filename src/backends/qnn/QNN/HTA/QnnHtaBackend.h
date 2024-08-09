//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTA component Backend API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnBackend.h for HTA backend
 */

#ifndef QNN_HTA_BACKEND_H
#define QNN_HTA_BACKEND_H

#include "QnnBackend.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/* @brief Enum describing the set of features supported by HTA backend.
          This is used as a bitmask, so assign unique bits to each entries.
*/
typedef enum {
  ///  The accelerator will always attempt to fold relu activation
  ///  into the immediate preceding convolution operation. This optimization
  ///  is correct when quantization ranges for convolution are equal or
  ///  subset of the Relu operation. For graphs, where this cannot be
  ///  guranteed, the client should set this flag
  QNN_HTA_FOLD_RELU_ACTIVATION_INTO_CONV_OFF = 1 << 0,
  /// UNKNOWN enum event that must not be used
  QNN_HTA_BACKEND_FEATURES_UNKNOWN = 0x7fffffff
} QnnHtaBackend_Features_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief Structure describing the set of configurations supported by the backend.
 *        Objects of this type are to be referenced through QnnBackend_CustomConfig_t.
 */
typedef struct {
    /// field to save the features that are passed
    /// via QnnHtaBackend_Features_t
    uint32_t bitmaskFeatures;
} QnnHtaBackend_CustomConfig_t ;

/// QnnHtaBackend_CustomConfig_t initializer macro
#define QNN_HTA_BACKEND_CUSTOM_CONFIG_INIT \
  { 0 /*bitmaskFeatures*/ }

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif

//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN DSP component Backend API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnBackend.h for DSP backend
 */

#ifndef QNN_DSP_BACKEND_H
#define QNN_DSP_BACKEND_H

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

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/* @brief Enum describing the set of custom configs supported by DSP backend.
*/
typedef enum {
  ///  The accelerator will always attempt to fold relu activation
  ///  into the immediate preceding convolution operation. This optimization
  ///  is correct when quantization ranges for convolution are equal or
  ///  subset of the Relu operation. For graphs, where this cannot be
  ///  guaranteed, the client should set this option to true
  QNN_DSP_BACKEND_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF = 0,
  ///  The accelerator will always attempt to all Convolution
  ///  operation using HMX instructions. Convolution that have
  ///  short depth and/or weights that are not symmetric could
  ///  exhibit inaccurate results. In such cases, clients must
  ///  set this option to true to guarantee correctness of the operation
  QNN_DSP_BACKEND_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF = 1,
  ///  Every APP side user process that uses a DSP via FastRPC
  ///  has a corresponding dynamic user process domain on the DSP side.
  ///  QNN by default opens RPC session as unsigned PD,
  ///  in case this option is set to true,
  ///  RPC session will be opened as signed PD (requires signed .so).
  QNN_DSP_BACKEND_CONFIG_OPTION_USE_SIGNED_PROCESS_DOMAIN = 2,
  /// set QnnDspBackend_DspArch_t for offline prepare mode
  QNN_DSP_BACKEND_CONFIG_OPTION_ARCH = 3,
  /// UNKNOWN enum option that must not be used
  QNN_DSP_BACKEND_CONFIG_OPTION_UNKNOWN = 0x7fffffff
} QnnDspBackend_ConfigOption_t;

typedef enum {
  QNN_DSP_BACKEND_DSP_ARCH_NONE = 0,
  QNN_DSP_BACKEND_DSP_ARCH_V65 = 65,
  QNN_DSP_BACKEND_DSP_ARCH_V66 = 66,
  QNN_DSP_BACKEND_DSP_ARCH_V68 = 68,
  QNN_DSP_BACKEND_DSP_ARCH_V69 = 69,
  QNN_DSP_BACKEND_DSP_ARCH_V73 = 73,
  QNN_DSP_BACKEND_DSP_ARCH_UNKNOWN = 0x7fffffff
} QnnDspBackend_DspArch_t;

/**
 * @brief Structure describing the set of configurations supported by the backend.
 *        Objects of this type are to be referenced through QnnBackend_CustomConfig_t.
 */
typedef struct QnnDspBackend_CustomConfig {
  QnnDspBackend_ConfigOption_t option;
  union UNNAMED {
    bool foldReluActivationIntoConvOff;
    bool shortDepthConvOnHmxOff;
    bool useSignedProcessDomain;
    QnnDspBackend_DspArch_t arch;
  };
} QnnDspBackend_CustomConfig_t ;

/// QnnDspBackend_CustomConfig_t initializer macro
#define QNN_DSP_BACKEND_CUSTOM_CONFIG_INIT \
  {                                                   \
    QNN_DSP_BACKEND_CONFIG_OPTION_UNKNOWN, /*option*/ \
    {                                                 \
      false /*foldReluActivationIntoConvOff*/         \
    }                                                 \
  }

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif

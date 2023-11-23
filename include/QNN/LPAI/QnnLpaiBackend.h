//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN LPAI component Backend API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnBackend.h for LPAI backend
 */

#ifndef QNN_LPAI_BACKEND_H
#define QNN_LPAI_BACKEND_H

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

/**
 * @brief An enum which defines the different targets supported by LPAI compilation.
 */
typedef enum {
    /// LPAI model will be compiled for x86
    QNN_LPAI_BACKEND_TARGET_X86       = 0,
    /// LPAI model will be compiled for ARM
    QNN_LPAI_BACKEND_TARGET_ARM       = 1,
    /// LPAI model will be compiled for ADSP
    QNN_LPAI_BACKEND_TARGET_ADSP      = 2,
    /// LPAI model will be compiled for TENSILICA
    QNN_LPAI_BACKEND_TARGET_TENSILICA = 3,
    /// UNKNOWN enum event that must not be used
    QNN_LPAI_BACKEND_TARGET_UNKNOWN = 0x7fffffff,
} QnnLpaiBackend_Target_t;

/**
 * @brief An enum which defines the version of LPAI Hardware.
 */
typedef enum {
    /// No LPAI HW will be used
    QNN_LPAI_BACKEND_HW_VERSION_NA   = 0,
    /// LPAI HW version v1
    QNN_LPAI_BACKEND_HW_VERSION_V1   = 1,
    /// LPAI HW version v2
    QNN_LPAI_BACKEND_HW_VERSION_V2   = 2,
    /// LPAI HW version v3
    QNN_LPAI_BACKEND_HW_VERSION_V3   = 3,
    /// LPAI HW version v4
    QNN_LPAI_BACKEND_HW_VERSION_V4   = 4,
    /// UNKNOWN enum event that must not be used
    QNN_LPAI_BACKEND_HW_VERSION_UNKNOWN = 0x7fffffff,
} QnnLpaiBackend_HwVersion_t;

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
    QnnLpaiBackend_Target_t      lpaiTarget;
    QnnLpaiBackend_HwVersion_t   hwVersion;
    uint32_t                     enableLayerFusion;
    uint32_t                     enableBatchnormFold;
    uint32_t                     enableChannelAlign;
    uint32_t                     enablePadSplit;
    uint32_t                     excludeIo;
} QnnLpaiBackend_CustomConfig_t ;

// clang-format off
/// QnnLpaiBackend_CustomConfig_t initializer macro
#define QNN_LPAI_BACKEND_CUSTOM_CONFIG_INIT                   \
  {                                                           \
    QNN_LPAI_BACKEND_TARGET_ADSP,     /*lpaiTarget*/          \
    QNN_LPAI_BACKEND_HW_VERSION_V4,   /*hwVersion*/           \
    1u,                               /*enableLayerFusion*/   \
    1u,                               /*enableBatchnormFold*/ \
    0u,                               /*enableChannelAlign*/  \
    0u,                               /*enablePadSplit*/      \
    0u                                /*excludeIo*/           \
  }

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif

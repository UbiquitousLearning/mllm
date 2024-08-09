//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN DSP Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for DSP backend
 */

#ifndef QNN_DSP_COMMON_H
#define QNN_DSP_COMMON_H

#include "QnnCommon.h"

/// DSP Backend identifier
#define QNN_BACKEND_ID_DSP 5

/// DSP interface provider
#define QNN_DSP_INTERFACE_PROVIDER_NAME "DSP_QTI_AISW"

// DSP API Version values
#define QNN_DSP_API_VERSION_MAJOR 5
#define QNN_DSP_API_VERSION_MINOR 0
#define QNN_DSP_API_VERSION_PATCH 1

// clang-format off

/// Macro to set Qnn_ApiVersion_t for DSP backend
#define QNN_DSP_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_DSP_API_VERSION_MAJOR, /*backendApiVersion.major*/     \
      QNN_DSP_API_VERSION_MINOR, /*backendApiVersion.minor*/     \
      QNN_DSP_API_VERSION_PATCH  /*backendApiVersion.patch*/     \
    }                                                            \
  }

// clang-format on

// DSP Binary Version values
#define QNN_DSP_BINARY_VERSION_MAJOR 1
#define QNN_DSP_BINARY_VERSION_MINOR 0
#define QNN_DSP_BINARY_VERSION_PATCH 0

// DSP Context blob Version values
#define QNN_DSP_CONTEXT_BLOB_VERSION_MAJOR 1
#define QNN_DSP_CONTEXT_BLOB_VERSION_MINOR 0
#define QNN_DSP_CONTEXT_BLOB_VERSION_PATCH 0

#endif  // QNN_DSP_COMMON_H

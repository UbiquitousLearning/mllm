//=============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN CPU Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for CPU backend
 */

#ifndef QNN_CPU_COMMON_H
#define QNN_CPU_COMMON_H

#include "QnnCommon.h"

/// CPU Backend identifier
#define QNN_BACKEND_ID_CPU 3

/// CPU interface provider
#define QNN_CPU_INTERFACE_PROVIDER_NAME "CPU_QTI_AISW"

// CPU API Version values
#define QNN_CPU_API_VERSION_MAJOR 1
#define QNN_CPU_API_VERSION_MINOR 1
#define QNN_CPU_API_VERSION_PATCH 0

// clang-format off
/// Macro to set Qnn_ApiVersion_t for CPU backend
#define QNN_CPU_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_CPU_API_VERSION_MAJOR, /*backendApiVersion.major*/     \
      QNN_CPU_API_VERSION_MINOR, /*backendApiVersion.minor*/     \
      QNN_CPU_API_VERSION_PATCH  /*backendApiVersion.patch*/     \
    }                                                            \
  }

// clang-format on

#endif  // QNN_CPU_COMMON_H
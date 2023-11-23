//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTA Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for HTA backend
 */

#ifndef QNN_HTA_COMMON_H
#define QNN_HTA_COMMON_H

#include "QnnCommon.h"

/// HTA Backend identifier
#define QNN_BACKEND_ID_HTA 7

/// HTA interface provider
#define QNN_HTA_INTERFACE_PROVIDER_NAME "HTA_QTI_AISW"

// HTA API Version values

#define QNN_HTA_API_VERSION_MAJOR 2
#define QNN_HTA_API_VERSION_MINOR 0
#define QNN_HTA_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for HTA backend
#define QNN_HTA_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_HTA_API_VERSION_MAJOR, /*backendApiVersion.major*/     \
      QNN_HTA_API_VERSION_MINOR, /*backendApiVersion.minor*/     \
      QNN_HTA_API_VERSION_PATCH  /*backendApiVersion.patch*/     \
    }                                                            \
  }

// clang-format on

// HTA Binary Version values
#define QNN_HTA_BINARY_VERSION_MAJOR 2
#define QNN_HTA_BINARY_VERSION_MINOR 0
#define QNN_HTA_BINARY_VERSION_PATCH 0

// HTA Context blob Version values
#define QNN_HTA_CONTEXT_BLOB_VERSION_MAJOR 1
#define QNN_HTA_CONTEXT_BLOB_VERSION_MINOR 1
#define QNN_HTA_CONTEXT_BLOB_VERSION_PATCH 0

#endif  // QNN_HTA_COMMON_H

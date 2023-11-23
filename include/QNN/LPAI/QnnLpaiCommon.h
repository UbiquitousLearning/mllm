//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN LPAI Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for LPAI backend
 */

#ifndef QNN_LPAI_COMMON_H
#define QNN_LPAI_COMMON_H

#include "QnnCommon.h"

/// QNN LPAI Backend identifier
#define QNN_BACKEND_ID_LPAI 12

/// QNN LPAI interface provider
#define QNN_LPAI_INTERFACE_PROVIDER_NAME "LPAI_QTI_AISW"

/// QNN LPAI API Version values
#define QNN_LPAI_API_VERSION_MAJOR 2
#define QNN_LPAI_API_VERSION_MINOR 5
#define QNN_LPAI_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for LPAI backend
#define QNN_LPAI_API_VERSION_INIT                                \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_LPAI_API_VERSION_MAJOR, /*backendApiVersion.major*/    \
      QNN_LPAI_API_VERSION_MINOR, /*backendApiVersion.minor*/    \
      QNN_LPAI_API_VERSION_PATCH  /*backendApiVersion.patch*/    \
    }                                                            \
  }

// clang-format on

/// QNN LPAI Binary Version values
#define QNN_LPAI_BINARY_VERSION_MAJOR 1
#define QNN_LPAI_BINARY_VERSION_MINOR 0
#define QNN_LPAI_BINARY_VERSION_PATCH 0

/// QNN LPAI Context blob Version values
#define QNN_LPAI_CONTEXT_BLOB_VERSION_MAJOR 1
#define QNN_LPAI_CONTEXT_BLOB_VERSION_MINOR 0
#define QNN_LPAI_CONTEXT_BLOB_VERSION_PATCH 0

#endif  // QNN_LPAI_COMMON_H

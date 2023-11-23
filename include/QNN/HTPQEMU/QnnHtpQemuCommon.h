//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP QEMU Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for HTP QEMU backend
 */

#ifndef QNN_HTP_QEMU_COMMON_H
#define QNN_HTP_QEMU_COMMON_H

#include "QnnCommon.h"

/// HTP QEMU Backend identifier
#define QNN_BACKEND_ID_HTP_QEMU 13

/// HTP QEMU interface provider
#define QNN_HTP_QEMU_INTERFACE_PROVIDER_NAME "HTP_QEMU_QTI_AISW"

// HTP QEMU API Version values
#define QNN_HTP_QEMU_API_VERSION_MAJOR 1
#define QNN_HTP_QEMU_API_VERSION_MINOR 0
#define QNN_HTP_QEMU_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for HTP QEMU backend
#define QNN_HTP_QEMU_API_VERSION_INIT                                 \
  {                                                                   \
    {                                                                 \
        QNN_API_VERSION_MAJOR,        /*coreApiVersion.major*/        \
        QNN_API_VERSION_MINOR,        /*coreApiVersion.major*/        \
        QNN_API_VERSION_PATCH         /*coreApiVersion.major*/        \
    },                                                                \
    {                                                                 \
      QNN_HTP_QEMU_API_VERSION_MAJOR,     /*backendApiVersion.major*/ \
      QNN_HTP_QEMU_API_VERSION_MINOR,     /*backendApiVersion.minor*/ \
      QNN_HTP_QEMU_API_VERSION_PATCH      /*backendApiVersion.patch*/ \
    }                                                                 \
  }

// clang-format on

// DSP Context blob Version values
#define QNN_HTP_QEMU_CONTEXT_BLOB_VERSION_MAJOR 3
#define QNN_HTP_QEMU_CONTEXT_BLOB_VERSION_MINOR 1
#define QNN_HTP_QEMU_CONTEXT_BLOB_VERSION_PATCH 0

#endif  // QNN_HTP_QEMU_COMMON_H

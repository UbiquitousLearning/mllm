//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTA component Device API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnDevice.h for HTA backend
 */
#ifndef QNN_HTA_DEVICE_H
#define QNN_HTA_DEVICE_H

#include "QnnDevice.h"
#include "QnnHtaPerfInfrastructure.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _QnnDevice_Infrastructure_t {
  QnnHtaPerfInfrastructure_SetPowerConfigFn_t setPowerConfig;
} QnnHtaDevice_Infrastructure_t;

// clang-format off
/// QnnHtaDevice_Infrastructure_t initializer macro
#define QNN_HTA_DEVICE_INFRASTRUCTURE_INIT \
  {                                        \
    NULL,     /*setPowerConfig*/           \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
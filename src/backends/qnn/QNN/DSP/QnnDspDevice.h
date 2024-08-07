//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN DSP component Device API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnDevice.h for DSP backend
 */
#ifndef QNN_DSP_DEVICE_H
#define QNN_DSP_DEVICE_H

#include "QnnDevice.h"
#include "QnnDspPerfInfrastructure.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _QnnDevice_Infrastructure_t {
  QnnDspPerfInfrastructure_CreatePowerConfigIdFn_t createPowerConfigId;
  QnnDspPerfInfrastructure_DestroyPowerConfigIdFn_t destroyPowerConfigId;
  QnnDspPerfInfrastructure_SetPowerConfigFn_t setPowerConfig;
  QnnDspPerfInfrastructure_SetMemoryConfigFn_t setMemoryConfig;
  QnnDspPerfInfrastructure_SetThreadConfigFn_t setThreadConfig;
} QnnDspDevice_Infrastructure_t;

#define QNN_DSP_DEVICE_INFRASTRUCTURE_INIT \
  {                                        \
    NULL,     /*createPowerConfigId*/      \
        NULL, /*destroyPowerConfigId*/     \
        NULL, /*setPowerConfig*/           \
        NULL, /*setMemoryConfig*/          \
        NULL  /*setThreadConfig*/          \
  }

#ifdef __cplusplus
}
#endif

#endif
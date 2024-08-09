//==============================================================================
//
// Copyright (c) 2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** @file
 *  @brief QNN HTA component Performance Infrastructure API
 *
 *         Provides interface to the client to control performance and system
 *         settings of the QNN HTA Accelerator
 */

#ifndef QNN_HTA_PERF_INFRASTRUCTURE_H
#define QNN_HTA_PERF_INFRASTRUCTURE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN HTA PerfInfrastructure API result / error codes.
 *
 */
typedef enum {
  QNN_HTA_PERF_INFRASTRUCTURE_MIN_ERROR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE,
  ////////////////////////////////////////////////////////////////////////

  QNN_HTA_PERF_INFRASTRUCTURE_NO_ERROR                 = QNN_SUCCESS,
  QNN_HTA_PERF_INFRASTRUCTURE_ERROR_INVALID_HANDLE_PTR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 0,
  QNN_HTA_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT      = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 1,
  QNN_HTA_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED_CONFIG = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 2,
  QNN_HTA_PERF_INFRASTRUCTURE_ERROR_TRANSPORT          = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 3,

  ////////////////////////////////////////////////////////////////////////
  QNN_HTA_PERF_INFRASTRUCTURE_MAX_ERROR = QNN_MAX_ERROR_PERF_INFRASTRUCTURE
} QnnHtaPerfInfrastructure_Error_t;

/**
 * @brief This enum defines all the possible performance
 *        options in Hta Performance Infrastructure that
 *        relate to setting up of power levels
 */
typedef enum {
  /// config enum implies the usage of powerModeConfig struct. If not provided
  /// will be used as type identificator
  QNN_HTA_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_POWER_MODE = 1,
  /// UNKNOWN config option which must not be used
  QNN_HTA_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnHtaPerfInfrastructure_PowerConfigOption_t;

/**
 * @brief This enum defines all the possible power mode
 *        that a client can set
 */
typedef enum {
  /// default mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_DEFAULT = 0,
  /// low power saver mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_LOW_POWER_SAVER = 1,
  /// power saver mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER = 2,
  /// high power saver mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_HIGH_POWER_SAVER = 3,
  /// balanced mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_BALANCED = 4,
  /// high performance mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_HIGH_PERFORMANCE = 5,
  /// burst mode
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_BURST = 6,
  /// UNKNOWN value that must not be used by client
  QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN = 0x7fffffff
} QnnHtaPerfInfrastructure_PowerMode_t;

/**
 * @brief This struct provides performance infrastructure configuration
 *         associated with setting up of power levels
 */
typedef struct {
  QnnHtaPerfInfrastructure_PowerConfigOption_t config;
  // Organize as union for future expand flexibility defined by PowerConfigOption_t
  union {
    QnnHtaPerfInfrastructure_PowerMode_t powerModeConfig;
  };
} QnnHtaPerfInfrastructure_PowerConfig_t;

/// QnnHtaPerfInfrastructure_PowerConfig_t initializer macro
#define QNN_HTA_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT                   \
  {                                                                     \
    QNN_HTA_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN, /*config*/  \
    {                                                                   \
      QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN /*powerModeConfig*/ \
    }                                                                   \
  }

//=============================================================================
// API Methods
//=============================================================================

/**
 * @brief This API allows client to set up system power configuration that
 *        will enable different performance modes.
 *
 * @param[in] clientId A power client id to associate calls to system
 *            power settings. A value of 0 implies NULL power client id
 *            and can override every other setting the user process. To
 *            enable power settings for multiple clients in the same
 *            process, use a non-zero power client id.
 *
 *
 * @param[in] config Pointer to a NULL terminated array
 *            of config option for performance configuration.
 *            NULL is allowed and indicates no config options are provided.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 */
typedef Qnn_ErrorHandle_t (*QnnHtaPerfInfrastructure_SetPowerConfigFn_t)(
    uint32_t clientId, const QnnHtaPerfInfrastructure_PowerConfig_t** config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_HTA_PERF_INFRASTRUCTURE_H

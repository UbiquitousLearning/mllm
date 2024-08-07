//==============================================================================
//
// Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** @file
 *  @brief QNN DSP component Performance Infrastructure API
 *
 *         Provides interface to the client to control performance and system
 *         settings of the QNN DSP Accelerator
 */

#ifndef QNN_DSP_PERF_INFRASTRUCTURE_H
#define QNN_DSP_PERF_INFRASTRUCTURE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// max rpc polling time allowed - 9999 us
#define QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME 9999

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN DSP PerfInfrastructure API result / error codes.
 *
 */
typedef enum {
  QNN_DSP_PERF_INFRASTRUCTURE_MIN_ERROR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE,
  ////////////////////////////////////////////////////////////////////////

  QNN_DSP_PERF_INFRASTRUCTURE_NO_ERROR                 = QNN_SUCCESS,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_HANDLE_PTR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 0,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT      = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 1,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED_CONFIG = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 2,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_TRANSPORT          = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 3,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED        = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 4,
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_FAILED             = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 5,

  ////////////////////////////////////////////////////////////////////////
  QNN_DSP_PERF_INFRASTRUCTURE_MAX_ERROR = QNN_MAX_ERROR_PERF_INFRASTRUCTURE,
  /// UNDEFINED value that must not be used by client
  QNN_DSP_PERF_INFRASTRUCTURE_ERROR_UNDEFINED = 0x7fffffff
} QnnDspPerfInfrastructure_Error_t;

/**
 * @brief Used to allow client start (non-zero value) or stop participating
 * (zero value) in DCVS
 *
 */
typedef uint32_t QnnDspPerfInfrastructure_DcvsEnable_t;

/**
 * @brief Allows client to set up the sleep latency in microseconds
 *
 */
typedef uint32_t QnnDspPerfInfrastructure_SleepLatency_t;

/**
 * @brief Allows client to disable sleep or low power modes.
 * Pass a non-zero value to disable sleep in DSP
 *
 */
typedef uint32_t QnnDspPerfInfrastructure_SleepDisable_t;

/**
 * @brief sets the minimum size by which user heap should grow
 * when heap is exhausted. This API is expected to be
 * called only once per backend and has a process wide impact
 *
 * Grow size provided in bytes and defaults to 16MB
 */
typedef uint32_t QnnDspPerfInfrastructure_MemGrowSize_t;

/**
 * @brief sets the vtcm size to use for graphs that
 * are prepared offline. This API should be set up
 * before users can finalize a graph offline. It allows
 * the QNN DSP backend to configure the serialized
 * context for the available vtcm on target
 *
 * VTCM size provided in MB and does not have a default
 */
typedef uint32_t QnnDspPerfInfrastructure_VtcmSize_t;

/**
 * @brief sets the number of HVX threads for QNN DSP
 */
typedef uint32_t QnnDspPerfInfrastructure_HvxThreadNumber_t;

/**
 * @brief These are the different voltage corners that can
 * be requested by the client to influence the voting scheme
 * for DCVS
 *
 */
typedef enum {
  /// Maps to HAP_DCVS_VCORNER_DISABLE.
  /// Disable setting up voltage corner
  DCVS_VOLTAGE_CORNER_DISABLE = 0x10,
  /// Maps to HAP_DCVS_VCORNER_SVS2.
  /// Set voltage corner to minimum value supported on platform
  DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER = 0x20,
  /// Maps to HAP_DCVS_VCORNER_SVS2.
  /// Set voltage corner to SVS2 value for the platform
  DCVS_VOLTAGE_VCORNER_SVS2 = 0x30,
  /// Maps to HAP_DCVS_VCORNER_SVS.
  /// Set voltage corner to SVS value for the platform
  DCVS_VOLTAGE_VCORNER_SVS = 0x40,
  /// Maps to HAP_DCVS_VCORNER_SVS_PLUS.
  /// Set voltage corner to SVS_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_SVS_PLUS = 0x50,
  /// Maps to HAP_DCVS_VCORNER_NOM.
  /// Set voltage corner to NOMINAL value for the platform
  DCVS_VOLTAGE_VCORNER_NOM = 0x60,
  /// Maps to HAP_DCVS_VCORNER_NOM_PLUS.
  /// Set voltage corner to NOMINAL_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_NOM_PLUS = 0x70,
  /// Maps to HAP_DCVS_VCORNER_TURBO.
  /// Set voltage corner to TURBO value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO = 0x80,
  /// Maps to HAP_DCVS_VCORNER_TURBO_PLUS.
  /// Set voltage corner to TURBO_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO_PLUS = 0x90,
  /// Maps to HAP_DCVS_VCORNER_MAX.
  /// Set voltage corner to maximum value supported on the platform
  DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER = 0xA0,
  /// UNKNOWN value that must not be used by client
  DCVS_VOLTAGE_VCORNER_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_VoltageCorner_t;

/**
 * @brief This enum defines all the possible power mode
 *        that a client can set to influence DCVS mode
 */
typedef enum {
  /// Maps to HAP_DCVS_V2_ADJUST_UP_DOWN.
  /// Allows for DCVS to adjust up and down
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN = 0x1,
  /// Maps to HAP_DCVS_V2_ADJUST_ONLY_UP.
  /// Allows for DCVS to adjust up only
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_ONLY_UP = 0x2,
  /// Maps to HAP_DCVS_V2_POWER_SAVER_MODE.
  /// Higher thresholds for power efficiency
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE = 0x4,
  /// Maps to HAP_DCVS_V2_POWER_SAVER_AGGRESSIVE_MODE.
  /// Higher thresholds for power efficiency with faster ramp down
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_AGGRESSIVE_MODE = 0x8,
  /// Maps to HAP_DCVS_V2_PERFORMANCE_MODE.
  /// Lower thresholds for maximum performance
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE = 0x10,
  /// Maps to HAP_DCVS_V2_DUTY_CYCLE_MODE.
  /// The below value applies only for HVX clients:
  ///  - For streaming class clients:
  ///   - detects periodicity based on HVX usage
  ///   - lowers clocks in the no HVX activity region of each period.
  ///  - For compute class clients:
  ///   - Lowers clocks on no HVX activity detects and brings clocks up on detecting HVX activity
  ///   again.
  ///   - Latency involved in bringing up the clock will be at max 1 to 2 ms.
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE = 0x20,
  /// UNKNOWN value that must not be used by client
  QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_PowerMode_t;

/**
 * @brief This enum defines all the possible performance
 *        options in Dsp Performance Infrastructure that
 *        relate to setting up of power levels
 */
typedef enum {
  /// config enum implies the usage of dcvsEnableConfig struct. For dcvs v2, if not provided, will
  /// set to false
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE = 1,
  /// config enum implies the usage of sleepLatencyConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY = 2,
  /// config enum implies the usage of sleepDisableConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_DISABLE = 3,
  /// config enum implies the usage of dcvsPowerModeConfig struct. If not provided, power save mode
  /// will be used
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE = 4,
  /// config enum implies the usage of dcvsVoltageCornerConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER = 5,
  /// config enum implies the usage of busVoltageCornerConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER = 6,
  /// config enum implies the usage of coreVoltageCornerConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER = 7,
  /// config enum implies the usage of rpcControlLatencyConfig struct
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY = 9,
  /// config enum implies the usage of rpcPollingTimeConfig struct
  /// this config is only supported on V69 and later
  /// if enabled, this config is applied to entire process
  /// max allowed is QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME us
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME = 10,
  /// config HMX timeout interval in us. The HMX is turned off after the set interval
  /// time if no interaction with it after an inference is finished.
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_TIMEOUT_INTERVAL_US = 11,
  /// UNKNOWN config option which must not be used
  QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_PowerConfigOption_t;

/**
 * @brief Allows client to set up the RPC control latency in microseconds
 *
 */
typedef uint32_t QnnDspPerfInfrastructure_RpcControlLatency_t;

/**
 * @brief Allows client to set up the RPC polling time in microseconds
 */
typedef uint32_t QnnDspPerfInfrastructure_RpcPollingTime_t;

/**
 * @brief Allows client to set up the HMX timeout interval in microseconds
 */
typedef uint32_t QnnDspPerfInfrastructure_HmxTimeoutIntervalUs_t;

/**
 * @brief This struct provides performance infrastructure configuration
 *         associated with setting up of power levels
 */
typedef struct {
  QnnDspPerfInfrastructure_PowerConfigOption_t config;
  union {
    QnnDspPerfInfrastructure_DcvsEnable_t dcvsEnableConfig;
    QnnDspPerfInfrastructure_SleepLatency_t sleepLatencyConfig;
    QnnDspPerfInfrastructure_SleepDisable_t sleepDisableConfig;
    QnnDspPerfInfrastructure_PowerMode_t dcvsPowerModeConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t dcvsVoltageCornerMinConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t dcvsVoltageCornerTargetConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t dcvsVoltageCornerMaxConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t busVoltageCornerMinConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t busVoltageCornerTargetConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t busVoltageCornerMaxConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t coreVoltageCornerMinConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t coreVoltageCornerTargetConfig;
    QnnDspPerfInfrastructure_VoltageCorner_t coreVoltageCornerMaxConfig;
    QnnDspPerfInfrastructure_RpcControlLatency_t rpcControlLatencyConfig;
    QnnDspPerfInfrastructure_RpcPollingTime_t rpcPollingTimeConfig;
    QnnDspPerfInfrastructure_HmxTimeoutIntervalUs_t hmxTimeoutIntervalUsConfig;
  };
} QnnDspPerfInfrastructure_PowerConfig_t;

/// QnnDspPerfInfrastructure_PowerConfig_t initializer macro
#define QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT                  \
  {                                                                    \
    QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN, /*config*/ \
    {                                                                  \
      0 /*dcvsEnableConfig*/                                           \
    }                                                                  \
  }

/**
 * @brief This enum defines all the possible performance
 *        options in Dsp Performance Infrastructure that
 *        relate to system memory settings
 */
typedef enum {
  /// sets memory grow size
  QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE = 1,
  /// set the size of VTCM configuration (in MB) to use
  /// This setting is applicable only for off target usage.
  /// For on-target usage, refer QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_VTCM_USAGE_FACTOR
  QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_VTCM_SIZE = 2,
  /// set the vtcm usage factor on-target
  QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_VTCM_USAGE_FACTOR = 3,
  /// UNKNOWN config option that must not be used
  QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_MemoryConfigOption_t;

/**
 * @brief This enum defines all the possible performance
 *        options in Dsp Performance Infrastructure that
 *        relate to thread settings
 */
typedef enum {
  /// sets number of HVX threads
  QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIGOPTION_NUMBER_OF_HVX_THREADS = 1,
  /// UNKNOWN config option that must not be used
  QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_ThreadConfigOption_t;

/**
 * @brief This enum defines all the possible vtcm
 *        usage configuration. These settings apply only
 *        for on-target libraries
 *
 */
typedef enum {
  /// use all the vtcm available on target
  QNN_DSP_PERF_INFRASTRUCTURE_VTCM_USE_FULL = 1,
  /// use bare minimal vtcm available on target. This is
  /// not supported in the current release.
  QNN_DSP_PERF_INFRASTRUCTURE_VTCM_USE_MIN     = 2,
  QNN_DSP_PERF_INFRASTRUCTURE_VTCM_USE_UNKNOWN = 0x7fffffff
} QnnDspPerfInfrastructure_VtcmUsageFactor_t;

/**
 * @brief Provides performance infrastructure configuration
 *        options that are memory specific
 */
typedef struct {
  QnnDspPerfInfrastructure_MemoryConfigOption_t config;
  union {
    QnnDspPerfInfrastructure_MemGrowSize_t memGrowSizeConfig;
    QnnDspPerfInfrastructure_VtcmSize_t vtcmSizeInMB;
    QnnDspPerfInfrastructure_VtcmUsageFactor_t vtcmUsageConfig;
  };
} QnnDspPerfInfrastructure_MemoryConfig_t;

/// QnnDspPerfInfrastructure_MemoryConfig_t initializer macro
#define QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIG_INIT                  \
  {                                                                     \
    QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN, /*config*/ \
    {                                                                   \
      0 /*memGrowSizeConfig*/                                           \
    }                                                                   \
  }

/**
 * @brief Provides performance infrastructure configuration
 *        options that are thread specific
 */
typedef struct {
  QnnDspPerfInfrastructure_ThreadConfigOption_t config;
  union {
    QnnDspPerfInfrastructure_HvxThreadNumber_t numHvxThreads;
  };
} QnnDspPerfInfrastructure_ThreadConfig_t;

/// QnnDspPerfInfrastructure_ThreadConfig_t initializer macro
#define QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIG_INIT                  \
  {                                                                     \
    QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIGOPTION_UNKNOWN, /*config*/ \
    {                                                                   \
      0 /*numHvxThreads*/                                               \
    }                                                                   \
  }

//=============================================================================
// API Methods
//=============================================================================

/**
 * @brief This API allows client to create power configuration id that
 *        has to be used to set different performance modes.
 *        Power configuration id has to be destroyed by client when not needed.
 *
 * @param[out] powerConfigId Pointer to power configuration id to be created.
 *
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if power configuration
 *            id is NULL
 */
typedef Qnn_ErrorHandle_t (*QnnDspPerfInfrastructure_CreatePowerConfigIdFn_t)(
    uint32_t* powerConfigId);

/**
 * @brief This API allows client to destroy power configuration id.
 *
 * @param[in] powerConfigId A power configuration id to be destroyed.
 *
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if power configuration
 *            id does not exist
 */
typedef Qnn_ErrorHandle_t (*QnnDspPerfInfrastructure_DestroyPowerConfigIdFn_t)(
    uint32_t powerConfigId);

/**
 * @brief This API allows client to set up system power configuration that
 *        will enable different performance modes. This API uses
 *        HAP_power_dcvs_v3_payload struct to config HAP power parameters.
 *        Detailed HAP power parameters description please refer to Hexagon
 *        SDK HAP_power_dcvs_v3_payload documentation.
 *
 * @param[in] powerConfigId A power client id to associate calls to system
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
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if power configuration
 *            does not exist
 */
typedef Qnn_ErrorHandle_t (*QnnDspPerfInfrastructure_SetPowerConfigFn_t)(
    uint32_t powerConfigId, const QnnDspPerfInfrastructure_PowerConfig_t** config);

/**
 * @brief This API allows clients to set up configuration associated with
 *        system memory
 *
 * @param[in] config Pointer to a NULL terminated array
 *            of config option for system memory configuration.
 *            NULL is allowed and indicates no config options are provided.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 */
typedef Qnn_ErrorHandle_t (*QnnDspPerfInfrastructure_SetMemoryConfigFn_t)(
    const QnnDspPerfInfrastructure_MemoryConfig_t** config);

/**
 * @brief This API allows clients to set up configuration for threads
 *
 * @param[in] config Pointer to a NULL terminated array
 *            of config option for thread configuration.
 *            NULL is allowed and indicates no config options are provided.
 *
 * @note This function should be called after QnnBackend_initialize and
 *       before Context and Graph calls
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED_CONFIG if invalid
 *            config or value passed
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if config is NULL
 *         \n QNN_DSP_PERF_INFRASTRUCTURE_ERROR_TRANSPORT if unable to set the
 *            settings in DSP
 */
typedef Qnn_ErrorHandle_t (*QnnDspPerfInfrastructure_SetThreadConfigFn_t)(
    const QnnDspPerfInfrastructure_ThreadConfig_t** config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_DSP_PERF_INFRASTRUCTURE_H

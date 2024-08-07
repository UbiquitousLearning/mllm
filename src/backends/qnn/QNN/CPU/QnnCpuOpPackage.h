//==============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** @file
 *  @brief CPU Operation Package component API
 *
 *         Provides interface to interact with OpPackage libraries registered
 *         with the CPU backend.
 */

#ifndef QNN_CPU_OP_PACKAGE_H
#define QNN_CPU_OP_PACKAGE_H

#include "CPU/QnnCpuCommon.h"
#include "QnnGraph.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QNN_CPUOPPACKAGE_TENSOR_DATA_FORMAT_FLAT_BUFFER 0

/**
 * @brief A value representing a tensor data format.
 */
typedef uint32_t QnnCpuOpPackage_TensorDataFormat_t;

/**
 * @brief A value representing a profile data in ms.
 */
typedef double QnnCpuOpPackage_ProfileData_t;

/**
 * @brief An enum to specify a param type.
 */
typedef enum {
  QNN_CPU_PARAMTYPE_SCALAR = 0,
  QNN_CPU_PARAMTYPE_TENSOR = 1,
  QNN_CPU_PARAMTYPE_STRING = 2,
  // Unused, present to ensure 32 bits.
  QNN_CPU_PARAMTYPE_UNDEFINED = 0xFFFFFFFF
} QnnCpuOpPackage_ParamType_t;

/**
 * @brief An enum to specify tensor data type.
 */
typedef enum {
  QNN_CPU_DATATYPE_BOOL_8   = 0x0508,
  QNN_CPU_DATATYPE_INT_8    = 0x0008,
  QNN_CPU_DATATYPE_INT_32   = 0x0032,
  QNN_CPU_DATATYPE_UINT_8   = 0x0108,
  QNN_CPU_DATATYPE_UINT_32  = 0x0132,
  QNN_CPU_DATATYPE_FLOAT_32 = 0x0232,
  // Unused, present to ensure 32 bits.
  QNN_CPU_DATATYPE_UNDEFINED = 0x7FFFFFFF
} QnnCpuOpPackage_DataType_t;

/**
 * @brief An enum to specify logging level.
 */
typedef enum {
  QNN_CPU_MSG_ERROR = 1,
  QNN_CPU_MSG_DEBUG = 2,
  QNN_CPU_MSG_LOW   = 3,
  QNN_CPU_MSG_MED   = 4,
  QNN_CPU_MSG_HIGH  = 5,
  // Unused, present to ensure 32 bits
  QNN_CPU_MSG_UNDEFINED = 0x7FFFFFFF
} QnnCpuOpPackage_MsgType_t;

/**
 * @brief An enum to specify the profiling type.
 */
typedef enum {
  QNN_CPU_PROFILE_BASIC    = 1,
  QNN_CPU_PROFILE_DETAILED = 2,
  // Unused, present to ensure 32 bits
  QNN_CPU_PROFILE_UNDEFINED = 0x7FFFFFFF
} QnnCpuOpPackage_ProfileType_t;

/**
 * @brief A struct which defines the Global infrastructure.
 */
typedef struct _QnnOpPackage_GlobalInfrastructure_t {
  // Message
  void (*reportMessage)(QnnCpuOpPackage_MsgType_t msgType, const char* msg, ...);

  // Profile
  void (*profile)(QnnCpuOpPackage_ProfileType_t profileType,
                  QnnCpuOpPackage_ProfileData_t timeInMsec);
} QnnCpuOpPackage_GlobalInfra_t;

// clang-format off
/// QnnCpuOpPackage_GlobalInfra_t initializer macro
#define QNN_CPU_OP_PACKAGE_GLOBAL_INFRA_INIT \
  {                                          \
    NULL,    /*reportMessage*/               \
    NULL     /*profile*/                     \
  }
// clang-format on

typedef Qnn_ErrorHandle_t (*QnnCpuOpPackage_OpImplFn_t)(void* opPkgNodeData);

/**
 * @brief A struct which defines the OpImpl definition.
 */
typedef struct _QnnOpPackage_OpImpl_t {
  QnnCpuOpPackage_OpImplFn_t opImplFn;
  void* userData;
} QnnCpuOpPackage_OpImpl_t;

// clang-format off
/// QnnCpuOpPackage_OpImpl_t initializer macro
#define QNN_CPU_OP_PACKAGE_OPIMPL_INIT \
  {                                    \
    NULL,    /*kernelFn*/              \
    NULL     /*userData*/              \
  }
// clang-format on

/**
 * @brief A struct which describes the properties of a tensor.
 *
 */
typedef struct {
  QnnCpuOpPackage_TensorDataFormat_t dataFormat;
  QnnCpuOpPackage_DataType_t dataType;
  uint32_t rank;
  uint32_t* maxDimensions;
  uint32_t* currentDimensions;
  void* data;
  Qnn_QuantizeParams_t quantizeParams;
} QnnCpuOpPackage_Tensor_t;

// clang-format off
/// QnnCpuOpPackage_Tensor_t initializer macro
#define QNN_CPU_OP_PACKAGE_TENSOR_INIT                        \
  {                                                           \
    QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/        \
    QNN_CPU_DATATYPE_UNDEFINED,         /*dataType*/          \
    0,                                  /*rank*/              \
    NULL,                               /*maxDimensions*/     \
    NULL,                               /*currentDimensions*/ \
    NULL,                               /*data*/              \
    QNN_QUANTIZE_PARAMS_INIT            /*quantizeParams*/    \
  }
// clang-format on

/**
 * @brief A struct which describes the parameters of a node.
 *
 */
typedef struct {
  QnnCpuOpPackage_ParamType_t type;
  const char* name;
  union {
    double scalarParam;
    const char* string;
    QnnCpuOpPackage_Tensor_t* tensorParam;
  };
} QnnCpuOpPackage_Param_t;

// clang-format off
/// QnnCpuOpPackage_Param_t initializer macro
#define QNN_CPU_OP_PACKAGE_PARAM_INIT     \
  {                                       \
    QNN_CPU_PARAMTYPE_UNDEFINED, /*type*/ \
    NULL,                        /*name*/ \
    {                                     \
      0 /*scalarParam*/                   \
    }                                     \
  }
// clang-format on

/**
 * @brief A struct which describes the node.
 *
 */
typedef struct _QnnOpPackage_Node_t {
  const char* name;
  const char* packageName;
  const char* typeName;
  uint32_t numOfParams;
  QnnCpuOpPackage_Param_t** params;
  uint32_t numOfInputs;
  QnnCpuOpPackage_Tensor_t** inputs;
  uint32_t numOfOutputs;
  QnnCpuOpPackage_Tensor_t** outputs;
} QnnCpuOpPackage_Node_t;

// clang-format off
/// QnnCpuOpPackage_Node_t initializer macro
#define QNN_CPU_OP_PACKAGE_NODE_INIT \
  {                                  \
    NULL,     /*name*/               \
    NULL,     /*packageName*/        \
    NULL,     /*typeName*/           \
    0,        /*numOfParams*/        \
    NULL,     /*params*/             \
    0,        /*numOfInputs*/        \
    NULL,     /*inputs*/             \
    0,        /*numOfOutputs*/       \
    NULL      /*outputs*/            \
  }
// clang-format on

/**
 * @brief Graph infrastructure.
 *
 */
typedef _QnnOpPackage_GraphInfrastructure_t QnnCpuOpPackage_GraphInfrastructure_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_CPU_OP_PACKAGE_H

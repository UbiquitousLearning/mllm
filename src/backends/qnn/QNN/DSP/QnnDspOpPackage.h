//==============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QNN_DSP_OP_PACKAGE_HPP
#define QNN_DSP_OP_PACKAGE_HPP

#include "QnnOpPackage.h"
#include "QnnTypes.h"
#include "Udo/UdoImplDsp.h"

/**
 * @brief A struct which defines the Global infrastructure.
 */
typedef struct _QnnOpPackage_GlobalInfrastructure_t {
  /// include the UdoMalloc, UdoFree and so on
  Udo_DspGlobalInfrastructure_t* dspGlobalInfra;
} QnnDspOpPackage_GlobalInfrastructure_t;

/**
 * @brief A struct which defines the operation info.
 */
typedef struct _QnnOpPackage_OperationInfo_t {
  char* opType;
  uint32_t numOfStaticParams;
  uint32_t numOfInputs;
  uint32_t numOfOutputs;

  Udo_CreateOpFactoryFunction_t createOpFactory;
  Udo_CreateOperationFunction_t createOperation;
  Udo_ExecuteOpFunction_t executeOp;
  Udo_ReleaseOpFunction_t releaseOp;
  Udo_ReleaseOpFactoryFunction_t releaseOpFactory;
  Udo_ValidateOperationFunction_t validateOp;
  Udo_QueryOperationFunction_t queryOp;
} QnnDspOpPackage_OperationInfo_t;

#endif

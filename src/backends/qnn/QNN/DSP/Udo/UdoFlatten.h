//==============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "DSP/Udo/UdoBase.h"

#define HVX_ALIGNMENT 128
#define DSP_STRUCT_ALIGNMENT 8
#define DSP_ALIGN(X, ALIGNMENT) (((X) + ALIGNMENT - 1) & (~((ALIGNMENT)-1)))

typedef struct dspStaticParamsMeta {
        uint32_t size;
        uint32_t numParams;       
} dspStaticParamsMeta_t;

typedef struct tensorParamInfo {
        SnpeUdo_TensorLayout_t layout;
        SnpeUdo_QuantizeParams_t quantizeInfo;
        SnpeUdo_DataType_t dataType;
        uint32_t paddingFor8byteAlignment;
} tensorParamInfo_t;

typedef struct udoString {
        uint32_t sizeStruct;  // aligned
        uint32_t lengthString;  // does not include null character
        // followed by a string
} udoString_t;  // allocate mem for string for 8 byte alignment

typedef struct dims {
        uint32_t size;
        uint32_t rank;
        uint32_t ds;  // rank # of max dimensions followed by rank # of current dimensions for tensors
} dims_t;

typedef struct tensorData {
	uint32_t structSize;
        uint32_t dataSize;
        // followed by actual tensor data
} tensorData_t;

typedef struct dspStaticParamDescriptor {
        uint32_t size;   // including size of descriptor (including dims + data for tensors) (or including string for strings)
        SnpeUdo_ParamType_t paramType;
        union {   // not used for string data
                SnpeUdo_ScalarParam_t scalarInfo;
                tensorParamInfo_t tensorInfo;
        };
        udoString_t name;
        // followed by char*
        // in case of tensor, followed by dim_stride and tensor_data
        // in case of string, followed by udo_string and char*
} dspStaticParamDescriptor_t;

typedef struct paramSizes {
       uint32_t descriptorSize;
       uint32_t nameStructSize;
       uint32_t dimsSize;
       uint32_t dataStructSize;
       uint32_t dataSize;
       uint32_t stringDataStructSize;
} paramSizes_t;

typedef struct dspStaticParams {
        dspStaticParamsMeta_t meta;
        dspStaticParamDescriptor_t paramDesc;
} dspStaticParams_t;


int 
SnpeUdo_flattenStaticParams (SnpeUdo_Param_t** paramList, uint32_t numParams, uint32_t* flattenedSize, void** flattened);

void 
SnpeUdo_freeFlattenedStaticParams (void** flattened);


//==============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPU specialization of the QnnOpPackage.h interface.
 */

#ifndef QNN_GPU_OP_PACKAGE_H
#define QNN_GPU_OP_PACKAGE_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "GPU/QnnGpuCommon.h"
#include "GPU/QnnGpuGraph.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// QnnOpPackage_GlobalInfrastructure_t specialization.
//=============================================================================

/**
 * @brief A struct which is used to communicate device constant properties
 */
typedef struct {
  /// GPU device version string
  char deviceVersion[128];
  /// GPU driver interface version {major, minor}
  uint32_t interfaceVersion[2];
  /// GPU Adreno(TM) tier string
  char tierName[8];
  /// GPU driver version {product, major, minor, patch}
  uint32_t compilerVersion[4];
  /// GPU device max work group size
  size_t maxWorkGroupSize;
  /// GPU device image 2D max width
  size_t image2dMaxWidth;
  /// GPU device image 2D max height
  size_t image2dMaxHeight;
  /// GPU device max memory allocation size
  size_t maxBufferAllocSize;
  /// GPU device addr alignment in bits
  uint32_t baseAddrAlignment;
  /// GPU device image 2D Array max width
  size_t image2dArrayMaxWidth;
  /// GPU device image 2D Array max height
  size_t image2dArrayMaxHeight;
  /// GPU device image 2D Array max depth
  size_t image2dArrayMaxDepth;
} QnnGpu_DeviceProperties_t;

/**
 * @brief A QNN GPU struct specializing QnnOpPackage_GlobalInfrastructure_t
 */
typedef struct _QnnOpPackage_GlobalInfrastructure_t {
  /// GPU backend version (as returned by QnnBackend_getApiVersion())
  const Qnn_ApiVersion_t* sdkApiVersion;
  /// GPU device properties
  const QnnGpu_DeviceProperties_t* deviceProperties;
  /// Null terminated path to the OpenCL driver used by the backend
  const char* driverPath;
} QnnGpuOpPackage_GlobalInfrastructure_t;

//=============================================================================
// QnnOpPackage_PackageInfo_t specialization.
//=============================================================================

/**
 * @brief A struct having op package specific information
 */
typedef struct _QnnOpPackage_PackageInfo_t {
  /// Null terminated hash key string of all kernel sources
  const char* kernelRepoHash;
} QnnGpuOpPackage_PackageInfo_t;

//=============================================================================
// QnnOpPackage_Optimization_t specialization.
//=============================================================================

/**
 * @brief An enum to specify the QNN GPU optimization type
 *
 */
typedef enum {
  /// Undefined option only used for QNN_GPU_OP_PACKAGE_OPTIMIZATION_INIT
  QNN_GPU_OPTIMIZATION_TYPE_UNDEFINED = 0,
  /// Super node optimization
  QNN_GPU_OPTIMIZATION_TYPE_SUPER_NODE = 2,
} QnnGpuOpPackage_OptimizationType_t;

/**
 * @brief A struct representing a super node connection constraint.
 */
typedef struct {
  /// Producer node corresponding to QnnGpuOpPackage_SuperNodeOptimization_t::operations
  uint32_t producer;
  /// Output tensor index corresponding to the producer node
  uint32_t producerOutputIndex;
  /// Consumer node corresponding to QnnGpuOpPackage_SuperNodeOptimization_t::operations
  uint32_t consumer;
  /// Output tensor index corresponding to the consumer node
  uint32_t consumerInputIndex;
} QnnGpuOpPackage_SuperNodeConnectionConstraint_t;

/**
 * @brief An enum to specify the source of a tensor in an op def for a tensor constraint.
 *
 */
typedef enum {
  /// Tensor is an op def output
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_SOURCE_OUTPUT = 1,
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_SOURCE_INPUT  = 2,
} QnnGpuOpPackage_TensorConstraintSource_t;

/**
 * @brief An enum to specify the tensor constraint type.
 *
 */
typedef enum {
  /// Add a Qnn_DataType_t to the whitelist of allowable types.
  /// If no data type constraint is present for a tensor, all data types are allowed.
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_CONSTRAINT_DATA_TYPE = 1,
  /// Tensor must match it's rank
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_CONSTRAINT_RANK = 2,
  /// Tensor must match one of it's dimensions
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_CONSTRAINT_DIMENSION = 3,
  /// Add a Qnn_TensorType_t to the whitelist of allowable tensor types.
  /// If no tensor type constraint is present for a tensor, all types are allowed.
  QNN_GPU_OPTIMIZATION_SUPER_NODE_TENSOR_CONSTRAINT_TENSOR_TYPE = 4,
} QnnGpuOpPackage_TensorConstraintType_t;

/**
 * @brief A struct representing a tensor constraint.
 */
typedef struct {
  /// Operation corresponding to QnnGpuOpPackage_SuperNodeOptimization_t::operations
  uint32_t operationIndex;
  /// Source of the tensor in the Qnn_OpConfig_t
  QnnGpuOpPackage_TensorConstraintSource_t source;
  union {
    /// Tensor index in the Qnn_OpConfig_t, used only for inputs and outputs
    uint32_t index;
    /// Tensor parameter name in the Qnn_OpConfig_t, used only for parameters
    const char* name;
  };
  /// Type of tensor constraint
  QnnGpuOpPackage_TensorConstraintType_t type;
  union {
    /// Tensor data type for Qnn_DataType_t constraints
    Qnn_DataType_t dataType;
    /// Tensor type for Qnn_TensorType_t constraints
    Qnn_TensorType_t tensorType;
    /// Tensor rank for rank constraints
    uint32_t rank;
    struct {
      /// Tensor dimension index for dimension constraints
      uint32_t index;
      /// Tensor dimension size for dimension constraints
      uint32_t size;
    } dimension;
  };
} QnnGpuOpPackage_TensorConstraint_t;

typedef struct {
  /// Null-terminated array of comma separated lists of operations used for matching super node ops.
  /// An asterisk (*) may be used to represent any operation type.
  const char** operations;
  /// Null-terminated array of pointers to super node connection constraints
  QnnGpuOpPackage_SuperNodeConnectionConstraint_t** connectionConstraints;
  /// Null-terminated array of pointers to super node tensor constraints
  QnnGpuOpPackage_TensorConstraint_t** tensorConstraints;
} QnnGpuOpPackage_SuperNodeOptimization_t;

// clang-format off
/// QnnGpuOpPackage_SuperNodeOptimization_t initializer macro
#define QNN_GPU_OP_PACKAGE_SUPER_NODE_OPTIMIZATION_INIT \
  {                                                     \
    NULL, /*operations*/                                \
    NULL, /*connectionConstraints*/                     \
    NULL, /*tensorConstraints*/                         \
  }
// clang-format on

/**
 * @brief A struct representing a QNN GPU optimization.
 */
typedef struct _QnnOpPackage_Optimization_t {
  /// Type of optimization
  QnnGpuOpPackage_OptimizationType_t type;
  /// Op package assigned name of the optimization
  const char* name;
  union {
    /// Super node optimization, used when type is QNN_GPU_OPTIMIZATION_TYPE_SUPER_NODE
    const QnnGpuOpPackage_SuperNodeOptimization_t* superNode;
  };
} QnnGpuOpPackage_Optimization_t;

/// QnnGpuOpPackage_Optimization_t initializer macro
#define QNN_GPU_OP_PACKAGE_OPTIMIZATION_INIT            \
  {                                                     \
    QNN_GPU_OPTIMIZATION_TYPE_UNDEFINED, NULL, { NULL } \
  }

//=============================================================================
// QnnOpPackage_GraphInfrastructure_t specialization.
//=============================================================================

/**
 * @brief A QNN GPU struct specializing QnnOpPackage_GraphInfrastructure_t
 */
typedef struct _QnnOpPackage_GraphInfrastructure_t {
  /// GPU precision mode, user-supplied hint used for optimal kernel selection
  QnnGpu_Precision_t precisionMode;
} QnnGpuOpPackage_GraphInfrastructure_t;

//=============================================================================
// QNN GPU Memory Object
//=============================================================================

/**
 * @brief An enum to specify the QNN GPU memory object type
 *
 */
typedef enum {
  /// Host memory, only used for Qnn_Param_t tensors
  QNN_GPU_MEM_OBJ_TYPE_HOST = 0,
  /// GPU driver buffer memory object
  QNN_GPU_MEM_OBJ_TYPE_BUFFER = 1,
  /// GPU driver image 2D memory object
  QNN_GPU_MEM_OBJ_TYPE_IMAGE2D = 2,
  /// GPU driver image 2D array memory object
  QNN_GPU_MEM_OBJ_TYPE_IMAGE2D_ARRAY = 3,
  /// Aggregation of GPU driver image 2D memory objects
  QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D = 4,
  /// Aggregation of GPU driver image 2D array memory objects
  QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D_ARRAY = 5,
  /// Memory type is unclaimed and can be specified by the op package via the \n
  /// QnnGpu_OutputClaim_t struct
  QNN_GPU_MEM_OBJ_TYPE_UNCLAIMED = 6,
} QnnGpu_MemoryObjectType_t;

/**
 * @brief An enum to specify the QNN GPU memory layout
 *
 */
typedef enum {
  /// HWC layout
  QNN_GPU_MEM_LAYOUT_HWC = 0,
  /// HCW layout
  QNN_GPU_MEM_LAYOUT_HCW = 1,
  /// CHW layout
  QNN_GPU_MEM_LAYOUT_CHW = 2,
  /// Undefined
  QNN_GPU_MEM_LAYOUT_UNDEFINED = 0x7FFFFFFF,
} QnnGpu_MemoryLayout_t;

/**
 * @brief A QNN GPU struct specifying a memory object
 *        This struct is used with the following kernel argument types:
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READ
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READWRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_WRITE
 */
typedef struct {
  /// Type of memory object
  QnnGpu_MemoryObjectType_t type;
  /// Data type of the memory object
  Qnn_DataType_t dataType;
  /// Memory object dimensions                                                                 \n
  ///   Size is numDimensions. Uses the following type dependent format:                       \n
  ///   QNN_GPU_MEM_OBJ_TYPE_BUFFER                   -> {numElements}                         \n
  ///   QNN_GPU_MEM_OBJ_TYPE_IMAGE2D                  -> {height,width}                        \n
  ///   QNN_GPU_MEM_OBJ_TYPE_IMAGE2D_ARRAY            -> {height,width,array_size}             \n
  ///   QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D       -> {num_batches,height,width}            \n
  ///   QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D_ARRAY -> {num_batches,height,width,array_size}
  uint32_t* dimensions;
  /// Memory object offsets                                         \n
  ///   Size is numDimensions.                                      \n
  ///   Indicates where the data store starts in the memory object. \n
  uint32_t* offsets;
  /// Number of dimensions in memory object                           \n
  ///   Size is numDimensions. Has the following type dependent size: \n
  ///   QNN_GPU_MEM_OBJ_TYPE_BUFFER                   -> 1            \n
  ///   QNN_GPU_MEM_OBJ_TYPE_IMAGE2D                  -> 2            \n
  ///   QNN_GPU_MEM_OBJ_TYPE_IMAGE2D_ARRAY            -> 3            \n
  ///   QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D       -> 3            \n
  ///   QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D_ARRAY -> 4
  uint32_t numDimensions;
  /// Memory object layout                           \n
  /// Op package specific layout identifier          \n
  /// Default is QNN_GPU_MEM_LAYOUT_UNDEFINED if not already specified by a prior operation
  QnnGpu_MemoryLayout_t layout;
} QnnGpu_MemoryObject_t;

// clang-format off
/// QnnGpu_MemoryObject_t initializer macro
#define QNN_GPU_MEMORY_OBJECT_INIT                    \
  {                                                   \
    QNN_GPU_MEM_OBJ_TYPE_UNCLAIMED, /*type*/          \
    QNN_DATATYPE_UNDEFINED,         /*dataType*/      \
    NULL,                           /*dimensions*/    \
    NULL,                           /*offsets*/       \
    0u,                             /*numDimensions*/ \
    QNN_GPU_MEM_LAYOUT_UNDEFINED    /*layout*/        \
  }
// clang-format on

//=============================================================================
// QnnOpPackage_Node_t specialization.
//=============================================================================

/**
 * @brief A QNN GPU struct specifying a storage tensor
 */
typedef struct {
  /// Tensor ID
  uint32_t id;
  /// Tensor's associated memory object
  const QnnGpu_MemoryObject_t* memoryObject;
} QnnGpu_TensorStorageType_t;

// clang-format off
/// QnnGpu_TensorStorageType_t initializer macro
#define QNN_GPU_TENSOR_STORAGE_TYPE_INIT \
  {                                      \
    0u,   /*id*/                         \
    NULL  /*memoryObject*/               \
  }
// clang-format on

/**
 * @brief A QNN GPU struct specializing QnnOpPackage_Node_t
 */
typedef struct _QnnOpPackage_Node_t {
  /// Optimization index, see QnnOpPackage_Info_t, ignore when only one op config provided
  uint32_t optimization;
  /// Null-terminated array of operation config pointers
  /// Only one pointer provided when no optimizations performed
  const Qnn_OpConfig_t** configs;
  /// Null-terminated array of tensor storage type pointers called out in the config
  const QnnGpu_TensorStorageType_t** storageTypes;
} QnnGpuOpPackage_Node_t;

//=============================================================================
// QnnOpPackage_OpImpl_t specialization.
//=============================================================================

/**
 * @brief A QNN GPU struct specifying an output tensor claim. Using the principle
 *        of least work, operations must output a memory object type that is most
 *        convenient for itself. Only QNN_TENSOR_TYPE_NATIVE tensor types may
 *        be claimed.
 */
typedef struct {
  /// Index into the Qnn_OpConfig_t provided in QnnGpuOpPackage_Node_t
  uint32_t opConfigIndex;
  /// Index into the operation outputs to identify the tensor
  uint32_t outputIndex;
  /// Specification of the claimed memory object
  const QnnGpu_MemoryObject_t* memoryObject;
} QnnGpu_OutputClaim_t;

// clang-format off
/// QnnGpu_OutputClaim_t initializer macro
#define QNN_GPU_OUTPUT_CLAIM_INIT \
  {                               \
    0u,      /*opConfigIndex*/    \
    0u,      /*outputIndex*/      \
    NULL     /*memoryObject*/     \
  }
// clang-format on

/**
 * @brief An enum to specify the kernel argument type.
 *
 */
typedef enum {
  /// Operation input tensor used as kernel input
  QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ = 0,
  /// Operation input tensor used as kernel output
  QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE = 1,
  /// Operation output tensor used as kernel output
  QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE = 2,
  /// Operation internal tensor used as kernel input
  QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READ = 3,
  /// Operation internal tensor used as kernel input/output
  QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READWRITE = 4,
  /// Operation internal tensor used as kernel output
  QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_WRITE = 5,
  /// Plain old data kernel argument
  QNN_GPU_KERNEL_ARG_TYPE_DATA = 6,
  /// Local memory kernel argument
  QNN_GPU_KERNEL_ARG_TYPE_LOCAL = 7,
  /// Null pointer kernel argument
  QNN_GPU_KERNEL_ARG_TYPE_NULL_PTR = 8,
} QnnGpu_KernelArgType_t;

/**
 * @brief A QNN GPU struct specifying a kernel argument corresponding to a tensor.
 *        This struct is used with the following kernel argument types:
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READ
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_READWRITE
 *          - QNN_GPU_KERNEL_ARG_TYPE_INTERNAL_WRITE
 */
typedef struct {
  /// Index into the Qnn_OpConfig_t provided in QnnGpuOpPackage_Node_t, ignored for INTERNAL types
  uint32_t opConfigIndex;
  /// Index into the operation input ot output list or the internal tensor list
  uint32_t tensorIndex;
  /// Batch element index for aggregated tensor types
  uint32_t element;
} QnnGpu_TensorKernelArg_t;

// clang-format off
/// QnnGpu_TensorKernelArg_t initializer macro
#define QNN_GPU_TENSOR_KERNEL_ARG_INIT \
  {                                    \
    0u,   /*opConfigIndex*/            \
    0u,   /*tensorIndex*/              \
    0u    /*element*/                  \
  }
// clang-format on

/**
 * @brief An enum to specify the kernel data argument type.
 *
 */
typedef enum {
  QNN_GPU_KERNEL_ARG_CL_TYPE_CHAR   = 0,
  QNN_GPU_KERNEL_ARG_CL_TYPE_UCHAR  = 1,
  QNN_GPU_KERNEL_ARG_CL_TYPE_SHORT  = 2,
  QNN_GPU_KERNEL_ARG_CL_TYPE_USHORT = 3,
  QNN_GPU_KERNEL_ARG_CL_TYPE_INT    = 4,
  QNN_GPU_KERNEL_ARG_CL_TYPE_UINT   = 5,
  QNN_GPU_KERNEL_ARG_CL_TYPE_LONG   = 6,
  QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG  = 7,
  QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT  = 8,
  QNN_GPU_KERNEL_ARG_CL_TYPE_DOUBLE = 9,
} QnnGpu_DataKernelArgType_t;

/**
 * @brief A QNN GPU struct specifying a kernel argument corresponding to a plain old data.
 *        This struct is used only with the QNN_GPU_KERNEL_ARG_TYPE_DATA arg type.
 */
typedef struct {
  /// Data type of the data
  QnnGpu_DataKernelArgType_t type;
  union {
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_CHAR
    int8_t qnnChar;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_UCHAR
    uint8_t qnnUChar;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_SHORT
    int16_t qnnShort;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_USHORT
    uint16_t qnnUShort;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_INT
    int32_t qnnInt;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_UINT
    uint32_t qnnUInt;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_LONG
    int64_t qnnLong;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG
    uint64_t qnnULong;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT
    float qnnFloat;
    /// Used with QNN_GPU_KERNEL_ARG_CL_TYPE_DOUBLE
    double qnnDouble;
  };
} QnnGpu_DataKernelArg_t;

/// QnnGpu_DataKernelArg_t initializer macro
#define QNN_GPU_DATA_KERNEL_ARG_INIT          \
  {                                           \
    QNN_GPU_KERNEL_ARG_CL_TYPE_CHAR, /*type*/ \
    {                                         \
      0 /*qnnChar*/                           \
    }                                         \
  }

/**
 * @brief A QNN GPU struct specifying a kernel argument corresponding to a local memory type.
 *        This struct is used only with the QNN_GPU_KERNEL_ARG_TYPE_LOCAL arg type.
 */
typedef struct {
  /// Size of the memory requested in bytes
  uint32_t size;
} QnnGpu_LocalKernelArg_t;

/// QnnGpu_LocalKernelArg_t initializer macro
#define QNN_GPU_LOCAL_KERNEL_ARG_INIT \
  { 0u /*size*/ }

/**
 * @brief A QNN GPU struct specifying a kernel argument.
 *        Note that the QNN_GPU_KERNEL_ARG_TYPE_NULL_PTR type does not have an entry in
 *        the union.
 */
typedef struct {
  /// Type of kernel argument
  QnnGpu_KernelArgType_t type;
  union {
    /// Tensor type argument
    QnnGpu_TensorKernelArg_t tensor;
    /// Plain old data argument
    QnnGpu_DataKernelArg_t data;
    /// Local memory argument
    QnnGpu_LocalKernelArg_t local;
  };
} QnnGpu_KernelArg_t;

/// QnnGpu_KernelArg_t initializer macro
#define QNN_GPU_KERNEL_ARG_INIT                 \
  {                                             \
    QNN_GPU_KERNEL_ARG_TYPE_NULL_PTR, /*type*/  \
    {                                           \
      QNN_GPU_TENSOR_KERNEL_ARG_INIT /*tensor*/ \
    }                                           \
  }

/**
 * @brief An enum to specify the kernel source type.
 *
 */
typedef enum {
  QNN_GPU_KERNEL_SOURCE_TYPE_TEXT   = 0,
  QNN_GPU_KERNEL_SOURCE_TYPE_BINARY = 1,
} QnnGpu_KernelSourceType_t;

/**
 * @brief A QNN GPU struct specifying a kernel.
 */
typedef struct {
  /// Kernel source code or binary
  const void* kernelSource;
  /// Length of kernel source/binary in bytes
  size_t sourceLength;
  /// Type of kernel source
  QnnGpu_KernelSourceType_t sourceType;
  /// Null terminated build options string used for kernel compilation
  const char* buildOptions;
  /// Rank of the globalWorkSizes
  size_t globalWorkDim;
  /// Global work sizes used by enqueuing the kernel
  size_t globalWorkSizes[3];
  /// Rank of the localWorkSizes
  size_t localWorkDim;
  /// Local work sizes used by enqueuing the kernel
  size_t localWorkSizes[3];
  /// Null-terminated array of kernel arguments in the order they appear in the kernel function
  QnnGpu_KernelArg_t** args;
  /// Null terminated name of the kernel
  const char* name;
  /// If non-zero, kernel will be enqueued during execute even if it is static
  uint32_t isDynamic;
  /// Reserved field, must be null
  void* reserved;
} QnnGpu_Kernel_t;

// clang-format off
/// QnnGpu_Kernel_t initializer macro
#define QNN_GPU_KERNEL_INIT                              \
  {                                                      \
    NULL,                            /*kernelSource*/    \
    0u,                              /*sourceLength*/    \
    QNN_GPU_KERNEL_SOURCE_TYPE_TEXT, /*sourceType*/      \
    NULL,                            /*buildOptions*/    \
    0u,                              /*globalWorkDim*/   \
    {0u},                            /*globalWorkSizes*/ \
    0u,                              /*localWorkDim*/    \
    {0u},                            /*localWorkSizes*/  \
    NULL,                            /*args*/            \
    NULL,                            /*name*/            \
    0u,                              /*isDynamic*/       \
    NULL                             /*reserved*/        \
  }
// clang-format on

/**
 * @brief A QNN GPU struct specifying an operation.
 */
typedef struct _QnnOpPackage_OpImpl_t {
  /// Null-terminated array of output claims
  QnnGpu_OutputClaim_t** outputClaims;
  /// Null-terminated array of tensor requests
  QnnGpu_MemoryObject_t** memoryObjects;
  /// Null-terminated array of kernels
  QnnGpu_Kernel_t** kernels;
} QnnGpu_Operation_t;

// clang-format off
/// QnnGpu_Operation_t initializer macro
#define QNN_GPU_OPERATION_INIT     \
  {                                \
    NULL,     /*outputClaims*/     \
    NULL,     /*memoryObjects*/    \
    NULL,     /*kernels*/          \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

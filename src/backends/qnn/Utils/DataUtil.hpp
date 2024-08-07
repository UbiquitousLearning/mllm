//==============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <map>
#include <queue>
#include <vector>

#include "QnnTypes.h"

namespace qnn {
namespace tools {
namespace datautil {
enum class StatusCode {
  SUCCESS,
  DATA_READ_FAIL,
  DATA_WRITE_FAIL,
  FILE_OPEN_FAIL,
  DIRECTORY_CREATE_FAIL,
  INVALID_DIMENSIONS,
  INVALID_DATA_TYPE,
  DATA_SIZE_MISMATCH,
  INVALID_BUFFER,
};

const size_t g_bitsPerByte = 8;

using ReadBatchDataRetType_t = std::tuple<StatusCode, size_t, size_t>;

std::tuple<StatusCode, size_t> getDataTypeSizeInBytes(Qnn_DataType_t dataType);

std::tuple<StatusCode, size_t> calculateLength(std::vector<size_t> dims, Qnn_DataType_t dataType);

size_t calculateElementCount(std::vector<size_t> dims);

std::tuple<StatusCode, size_t> getFileSize(std::string filePath);

StatusCode readDataFromFile(std::string filePath,
                            std::vector<size_t> dims,
                            Qnn_DataType_t dataType,
                            uint8_t* buffer);

/*
 * Read data in batches from Queue and try to matches the model input's
 * batches. If the queue is empty while matching the batch size of model,
 * pad the remaining buffer with zeros
 * @param filePathsQueue image paths queue
 * @param dims model input dimensions
 * @param dataType to create input buffer from file
 * @param buffer to fill the input image data
 *
 * @return ReadBatchDataRetType_t returns numFilesCopied and batchSize along
 * with status
 */
ReadBatchDataRetType_t readBatchDataAndUpdateQueue(std::queue<std::string>& filePaths,
                                                   std::vector<size_t> dims,
                                                   Qnn_DataType_t dataType,
                                                   uint8_t* buffer);

StatusCode readBinaryFromFile(std::string filePath, uint8_t* buffer, size_t bufferSize);

StatusCode writeDataToFile(std::string fileDir,
                           std::string fileName,
                           std::vector<size_t> dims,
                           Qnn_DataType_t dataType,
                           uint8_t* buffer);

StatusCode writeBatchDataToFile(std::vector<std::string> fileDirs,
                                std::string fileName,
                                std::vector<size_t> dims,
                                Qnn_DataType_t dataType,
                                uint8_t* buffer,
                                const size_t batchSize);

StatusCode writeBinaryToFile(std::string fileDir,
                             std::string fileName,
                             uint8_t* buffer,
                             size_t bufferSize);

template <typename T_QuantType>
datautil::StatusCode floatToTfN(
    T_QuantType* out, float* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode tfNToFloat(
    float* out, T_QuantType* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode castToFloat(float* out, T_QuantType* in, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode castFromFloat(T_QuantType* out, float* in, size_t numElements);

const std::map<Qnn_DataType_t, size_t> g_dataTypeToSize = {
    {QNN_DATATYPE_INT_8, 1},
    {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},
    {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},
    {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},
    {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},
    {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_FLOAT_64, 8},
    {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2},
    {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},
    {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
    {QNN_DATATYPE_BOOL_8, 1},
};
}  // namespace datautil
}  // namespace tools
}  // namespace qnn

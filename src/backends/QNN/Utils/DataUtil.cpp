//==============================================================================
//
//  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

using namespace qnn;
using namespace qnn::tools;

std::tuple<datautil::StatusCode, size_t> datautil::getDataTypeSizeInBytes(Qnn_DataType_t dataType) {
  if (g_dataTypeToSize.find(dataType) == g_dataTypeToSize.end()) {
    QNN_ERROR("Invalid qnn data type provided");
    return std::make_tuple(StatusCode::INVALID_DATA_TYPE, 0);
  }
  return std::make_tuple(StatusCode::SUCCESS, g_dataTypeToSize.find(dataType)->second);
}

size_t datautil::calculateElementCount(std::vector<size_t> dims) {
  if (dims.size() == 0) {
    return 0;
  }
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

std::tuple<datautil::StatusCode, size_t> datautil::calculateLength(std::vector<size_t> dims,
                                                                   Qnn_DataType_t dataType) {
  if (dims.size() == 0) {
    QNN_ERROR("dims.size() is zero");
    return std::make_tuple(StatusCode::INVALID_DIMENSIONS, 0);
  }
  StatusCode returnStatus{StatusCode::SUCCESS};
  size_t length{0};
  std::tie(returnStatus, length) = getDataTypeSizeInBytes(dataType);
  if (StatusCode::SUCCESS != returnStatus) {
    return std::make_tuple(returnStatus, 0);
  }
  length *= calculateElementCount(dims);
  return std::make_tuple(StatusCode::SUCCESS, length);
}

datautil::StatusCode datautil::readDataFromFile(std::string filePath,
                                                std::vector<size_t> dims,
                                                Qnn_DataType_t dataType,
                                                uint8_t* buffer) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return StatusCode::FILE_OPEN_FAIL;
  }
  in.seekg(0, in.end);
  const size_t length = in.tellg();
  in.seekg(0, in.beg);
  StatusCode err{StatusCode::SUCCESS};
  size_t l{0};
  std::tie(err, l) = datautil::calculateLength(dims, dataType);
  if (StatusCode::SUCCESS != err) {
    return err;
  }
  if (length != l) {
    QNN_ERROR("Input file %s: file size in bytes (%d), should be equal to: %d",
              filePath.c_str(),
              length,
              l);
    return StatusCode::DATA_SIZE_MISMATCH;
  }

  if (!in.read(reinterpret_cast<char*>(buffer), length)) {
    QNN_ERROR("Failed to read the contents of: %s", filePath.c_str());
    return StatusCode::DATA_READ_FAIL;
  }
  return StatusCode::SUCCESS;
}

datautil::ReadBatchDataRetType_t datautil::readBatchDataAndUpdateQueue(
    std::queue<std::string>& filePaths,
    std::vector<size_t> dims,
    Qnn_DataType_t dataType,
    uint8_t* buffer) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return std::make_tuple(StatusCode::INVALID_BUFFER, 0, 0);
  }
  StatusCode err{StatusCode::SUCCESS};
  size_t l{0};
  std::tie(err, l) = datautil::calculateLength(dims, dataType);
  if (StatusCode::SUCCESS != err) {
    return std::make_tuple(err, 0, 0);
  }
  size_t numInputsCopied = 0;
  size_t numBatchSize    = 0;
  size_t totalLength     = 0;
  do {
    if (filePaths.empty()) {
      numBatchSize += (l - totalLength) / (totalLength / numBatchSize);
      // pad the vector with zeros
      memset(buffer + totalLength, 0, (l - totalLength) * sizeof(char));
      totalLength = l;
    } else {
      std::ifstream in(filePaths.front(), std::ifstream::binary);
      if (!in) {
        QNN_ERROR("Failed to open input file: %s", filePaths.front().c_str());
        return std::make_tuple(StatusCode::FILE_OPEN_FAIL, numInputsCopied, numBatchSize);
      }
      in.seekg(0, in.end);
      const size_t length = in.tellg();
      in.seekg(0, in.beg);
      if ((l % length) != 0 || length > l || length == 0) {
        QNN_ERROR("Input file %s: file size in bytes (%d), should be multiples of: %d",
                  filePaths.front().c_str(),
                  length,
                  l);
        return std::make_tuple(StatusCode::DATA_SIZE_MISMATCH, numInputsCopied, numBatchSize);
      }
      if (!in.read(reinterpret_cast<char*>(buffer + (numInputsCopied * length)), length)) {
        QNN_ERROR("Failed to read the contents of: %s", filePaths.front().c_str());
        return std::make_tuple(StatusCode::DATA_READ_FAIL, numInputsCopied, numBatchSize);
      }
      QNN_VERBOSE("Return from readDataFromFile()");
      totalLength += length;
      numInputsCopied += 1;
      numBatchSize += 1;
      filePaths.pop();
    }
  } while (totalLength < l);
  return std::make_tuple(StatusCode::SUCCESS, numInputsCopied, numBatchSize);
}

std::tuple<datautil::StatusCode, size_t> datautil::getFileSize(std::string filePath) {
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return std::make_tuple(StatusCode::FILE_OPEN_FAIL, 0);
  }
  in.seekg(0, in.end);
  const size_t length = in.tellg();
  in.seekg(0, in.beg);
  return std::make_tuple(StatusCode::SUCCESS, length);
}

datautil::StatusCode datautil::readBinaryFromFile(std::string filePath,
                                                  uint8_t* buffer,
                                                  size_t bufferSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return StatusCode::FILE_OPEN_FAIL;
  }
  if (!in.read(reinterpret_cast<char*>(buffer), bufferSize)) {
    QNN_ERROR("Failed to read the contents of: %s", filePath.c_str());
    return StatusCode::DATA_READ_FAIL;
  }
  return StatusCode::SUCCESS;
}

datautil::StatusCode datautil::writeDataToFile(std::string fileDir,
                                               std::string fileName,
                                               std::vector<size_t> dims,
                                               Qnn_DataType_t dataType,
                                               uint8_t* buffer) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  if (!pal::Directory::makePath(fileDir)) {
    QNN_ERROR("Failed to create output directory: %s", fileDir.c_str());
    return StatusCode::DIRECTORY_CREATE_FAIL;
  }
  const std::string outputPath(fileDir + pal::Path::getSeparator() + fileName);
  std::ofstream os(outputPath, std::ofstream::binary);
  if (!os) {
    QNN_ERROR("Failed to open output file for writing: %s", outputPath.c_str());
    return StatusCode::FILE_OPEN_FAIL;
  }
  StatusCode err{StatusCode::SUCCESS};
  size_t length{0};
  std::tie(err, length) = datautil::calculateLength(dims, dataType);
  if (StatusCode::SUCCESS != err) {
    return err;
  }
  for (size_t l = 0; l < length; l++) {
    os.write(reinterpret_cast<char*>(&(*(buffer + l))), 1);
  }
  return StatusCode::SUCCESS;
}

datautil::StatusCode datautil::writeBatchDataToFile(std::vector<std::string> fileDirs,
                                                    std::string fileName,
                                                    std::vector<size_t> dims,
                                                    Qnn_DataType_t dataType,
                                                    uint8_t* buffer,
                                                    const size_t batchSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  StatusCode err{StatusCode::SUCCESS};
  size_t length{0};
  std::tie(err, length) = datautil::calculateLength(dims, dataType);
  if (StatusCode::SUCCESS != err) {
    return err;
  }
  auto outputSize = (length / batchSize);
  for (size_t batchIndex = 0; batchIndex < fileDirs.size(); batchIndex++) {
    std::string fileDir = fileDirs[batchIndex];
    if (!pal::Directory::makePath(fileDir)) {
      QNN_ERROR("Failed to create output directory: %s", fileDir.c_str());
      return StatusCode::DIRECTORY_CREATE_FAIL;
    }
    const std::string outputPath(fileDir + pal::Path::getSeparator() + fileName);
    std::ofstream os(outputPath, std::ofstream::binary);
    if (!os) {
      QNN_ERROR("Failed to open output file for writing: %s", outputPath.c_str());
      return StatusCode::FILE_OPEN_FAIL;
    }
    for (size_t l = 0; l < outputSize; l++) {
      size_t bufferIndex = l + (batchIndex * outputSize);
      os.write(reinterpret_cast<char*>(&(*(buffer + bufferIndex))), 1);
    }
  }
  return StatusCode::SUCCESS;
}

datautil::StatusCode datautil::writeBinaryToFile(std::string fileDir,
                                                 std::string fileName,
                                                 uint8_t* buffer,
                                                 size_t bufferSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  if (!pal::Directory::makePath(fileDir)) {
    QNN_ERROR("Failed to create output directory: %s", fileDir.c_str());
    return StatusCode::DIRECTORY_CREATE_FAIL;
  }
  const std::string outputPath(fileDir + pal::Path::getSeparator() + fileName);
  std::ofstream os(outputPath, std::ofstream::binary);
  if (!os) {
    QNN_ERROR("Failed to open output file for writing: %s", outputPath.c_str());
    return StatusCode::FILE_OPEN_FAIL;
  }
  os.write(reinterpret_cast<char*>(buffer), bufferSize);
  return StatusCode::SUCCESS;
}

template <typename T_QuantType>
datautil::StatusCode datautil::floatToTfN(
    T_QuantType* out, float* in, int32_t offset, float scale, size_t numElements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "floatToTfN supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    QNN_ERROR("Received a nullptr");
    return StatusCode::INVALID_BUFFER;
  }

  size_t dataTypeSizeInBytes = sizeof(T_QuantType);
  size_t bitWidth            = dataTypeSizeInBytes * g_bitsPerByte;
  double trueBitWidthMax     = pow(2, bitWidth) - 1;
  double encodingMin         = offset * scale;
  double encodingMax         = (trueBitWidthMax + offset) * scale;
  double encodingRange       = encodingMax - encodingMin;

  for (size_t i = 0; i < numElements; ++i) {
    int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);
    if (quantizedValue < 0)
      quantizedValue = 0;
    else if (quantizedValue > (int)trueBitWidthMax)
      quantizedValue = (int)trueBitWidthMax;
    out[i] = static_cast<T_QuantType>(quantizedValue);
  }
  return StatusCode::SUCCESS;
}

template datautil::StatusCode datautil::floatToTfN<uint8_t>(
    uint8_t* out, float* in, int32_t offset, float scale, size_t numElements);

template datautil::StatusCode datautil::floatToTfN<uint16_t>(
    uint16_t* out, float* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode datautil::tfNToFloat(
    float* out, T_QuantType* in, int32_t offset, float scale, size_t numElements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "tfNToFloat supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    QNN_ERROR("Received a nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  for (size_t i = 0; i < numElements; i++) {
    double quantizedValue = static_cast<double>(in[i]);
    double offsetDouble   = static_cast<double>(offset);
    out[i]                = static_cast<double>((quantizedValue + offsetDouble) * scale);
  }
  return StatusCode::SUCCESS;
}

template datautil::StatusCode datautil::tfNToFloat<uint8_t>(
    float* out, uint8_t* in, int32_t offset, float scale, size_t numElements);

template datautil::StatusCode datautil::tfNToFloat<uint16_t>(
    float* out, uint16_t* in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode datautil::castToFloat(float* out, T_QuantType* in, size_t numElements) {
  if (nullptr == out || nullptr == in) {
    QNN_ERROR("Received a nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  for (size_t i = 0; i < numElements; i++) {
    out[i] = static_cast<float>(in[i]);
  }
  return StatusCode::SUCCESS;
}

template datautil::StatusCode datautil::castToFloat<uint8_t>(float* out,
                                                             uint8_t* in,
                                                             size_t numElements);

template datautil::StatusCode datautil::castToFloat<uint16_t>(float* out,
                                                              uint16_t* in,
                                                              size_t numElements);

template datautil::StatusCode datautil::castToFloat<uint32_t>(float* out,
                                                              uint32_t* in,
                                                              size_t numElements);

template datautil::StatusCode datautil::castToFloat<int8_t>(float* out,
                                                            int8_t* in,
                                                            size_t numElements);

template datautil::StatusCode datautil::castToFloat<int16_t>(float* out,
                                                             int16_t* in,
                                                             size_t numElements);

template datautil::StatusCode datautil::castToFloat<int32_t>(float* out,
                                                             int32_t* in,
                                                             size_t numElements);

template <typename T_QuantType>
datautil::StatusCode datautil::castFromFloat(T_QuantType* out, float* in, size_t numElements) {
  if (nullptr == out || nullptr == in) {
    QNN_ERROR("Received a nullptr");
    return StatusCode::INVALID_BUFFER;
  }
  for (size_t i = 0; i < numElements; i++) {
    out[i] = static_cast<T_QuantType>(in[i]);
  }
  return StatusCode::SUCCESS;
}

template datautil::StatusCode datautil::castFromFloat<uint8_t>(uint8_t* out,
                                                               float* in,
                                                               size_t numElements);

template datautil::StatusCode datautil::castFromFloat<uint16_t>(uint16_t* out,
                                                                float* in,
                                                                size_t numElements);

template datautil::StatusCode datautil::castFromFloat<uint32_t>(uint32_t* out,
                                                                float* in,
                                                                size_t numElements);

template datautil::StatusCode datautil::castFromFloat<int8_t>(int8_t* out,
                                                              float* in,
                                                              size_t numElements);

template datautil::StatusCode datautil::castFromFloat<int16_t>(int16_t* out,
                                                               float* in,
                                                               size_t numElements);

template datautil::StatusCode datautil::castFromFloat<int32_t>(int32_t* out,
                                                               float* in,
                                                               size_t numElements);
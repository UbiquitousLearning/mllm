// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

// Ascend ACL error checking macro
#define MLLM_ACL_CHECK(err)                                                                                    \
  if (err != ACL_SUCCESS) {                                                                                    \
    MLLM_ERROR_EXIT(::mllm::ExitCode::kAscendError, "ACL error code {}: {}", int(err), aclGetRecentErrMsg()); \
  }

// Ascend ATB error checking macro
#define MLLM_ATB_CHECK(err)                                                                                    \
  if (err != atb::NO_ERROR) {                                                                                  \
    MLLM_ERROR_EXIT(::mllm::ExitCode::kAscendError, "ATB error code {}", int(err));                           \
  }

namespace mllm::ascend {

// Get global ATB Context (Lazy Initialization: aclrtSetDevice, atb::CreateContext, aclrtCreateStream, SetExecuteStream)
atb::Context* getGlobalAtbContext();

// Get global ATB Stream
aclrtStream getGlobalAtbStream();

// Sync global ATB Stream
void syncGlobalAtbStream();

// Convert MLLM Tensor metadata to ATB TensorDesc
void fillAtbTensorDesc(const Tensor& t, atb::TensorDesc& desc);

// Ascend device information structure
struct AscendDeviceInfo {
  std::string name;
  unsigned int id;
  size_t total_memory;  // bytes
  size_t free_memory;   // bytes
  std::string soc_version;
};

// Ascend device metadata collector (singleton)
class AscendDeviceMetaInfo {
 public:
  AscendDeviceMetaInfo();

  static AscendDeviceMetaInfo& instance() {
    static AscendDeviceMetaInfo instance;
    return instance;
  }

  AscendDeviceMetaInfo(const AscendDeviceMetaInfo&) = delete;
  AscendDeviceMetaInfo& operator=(const AscendDeviceMetaInfo&) = delete;

  std::vector<AscendDeviceInfo> devices;
};

// RAII handle for Ascend tensor with automatic memory block management
struct AscendTensorHandle {
  AscendTensorHandle() = default;
  AscendTensorHandle(Tensor tensor, int block_id);  // Construct with tensor and memory block ID
  ~AscendTensorHandle();  // Auto-release memory block

  AscendTensorHandle(const AscendTensorHandle&) = delete;
  AscendTensorHandle& operator=(const AscendTensorHandle&) = delete;
  AscendTensorHandle(AscendTensorHandle&& other) noexcept;  // Move constructor
  AscendTensorHandle& operator=(AscendTensorHandle&& other) noexcept;  // Move assignment

  void release();  // Manually release memory block and invalidate handle
  bool valid() const { return block_id_ >= 0; }  // Check if handle owns a valid memory block

  Tensor& tensor() { return tensor_; }  // Access tensor
  const Tensor& tensor() const { return tensor_; }  // Access tensor (const)
  int blockId() const { return block_id_; }  // Get memory block ID

 private:
  Tensor tensor_;
  int block_id_{-1};
};

// Prepare Ascend tensor from host float data (converts to FP16, allocates device memory, copies data)
AscendTensorHandle prepareAscendTensor(const std::vector<float>& host_data,
                                       int batch,
                                       int size);

}  // namespace mllm::ascend

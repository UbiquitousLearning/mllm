// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <chrono>

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"  // IWYU pragma: keep

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

// Setup ATB Tensor with correct dataSize calculated by ATB Utils
void fillAtbTensor(const Tensor& t, atb::Tensor& atb_tensor);

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

// Copy Ascend tensor to host as float (currently assumes FP16 tensor data).
std::vector<float> copyAscendTensorToHost(const Tensor& t);

// Verify Ascend tensor against expected values.
bool verifyAscendTensor(const Tensor& t,
                        const std::vector<float>& expected,
                        float atol = 1e-2f,
                        float rtol = 1e-2f,
                        bool verbose = true,
                        std::vector<float>* actual_out = nullptr);

using RefFn = std::function<std::vector<float>()>;
bool verifyAscendTensor(const Tensor& t,
                        const RefFn& ref_fn,
                        float atol = 1e-2f,
                        float rtol = 1e-2f,
                        bool verbose = true,
                        std::vector<float>* actual_out = nullptr);

// RAII timer for measuring scoped durations (optionally syncs the global stream).
class AscendTimer {
 public:
  explicit AscendTimer(const char* tag, bool sync_before = true, bool sync_after = true);
  ~AscendTimer();

 private:
  const char* tag_;
  bool sync_before_;
  bool sync_after_;
  std::chrono::high_resolution_clock::time_point start_;
};

// Convenience macros for scoped timing.
#define ASCEND_TIME_SCOPE(tag) ::mllm::ascend::AscendTimer timer_scope_##__LINE__(tag, true, true)
#define ASCEND_TIME_SCOPE_NOSYNC(tag) ::mllm::ascend::AscendTimer timer_scope_##__LINE__(tag, false, false)

}  // namespace mllm::ascend

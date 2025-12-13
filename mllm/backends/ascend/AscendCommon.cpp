// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendCommon.hpp"

#include <vector>
#include <mutex>
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

namespace mllm::ascend {

namespace {
aclrtStream& globalAtbStream() {
  static aclrtStream stream = nullptr;
  return stream;
}
}  // namespace

AscendTensorHandle::AscendTensorHandle(Tensor tensor, int block_id)
    : tensor_(std::move(tensor)), block_id_(block_id) {}

AscendTensorHandle::~AscendTensorHandle() { release(); }

AscendTensorHandle::AscendTensorHandle(AscendTensorHandle&& other) noexcept
    : tensor_(std::move(other.tensor_)), block_id_(other.block_id_) {
  other.block_id_ = -1;
}

AscendTensorHandle& AscendTensorHandle::operator=(AscendTensorHandle&& other) noexcept {
  if (this != &other) {
    release();
    tensor_ = std::move(other.tensor_);
    block_id_ = other.block_id_;
    other.block_id_ = -1;
  }
  return *this;
}

void AscendTensorHandle::release() {
  if (block_id_ >= 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(block_id_);
    block_id_ = -1;
    tensor_.impl()->storage()->ptr_ = nullptr;
  } else if (tensor_.impl() != nullptr) {
    tensor_.delete_();
  }
}

AscendTensorHandle prepareAscendTensor(const std::vector<float>& host_data,
                                       int batch,
                                       int size) {
  const size_t expected_elements = static_cast<size_t>(batch) * static_cast<size_t>(size);
  MLLM_RT_ASSERT_EQ(host_data.size(), expected_elements);

  std::vector<half_float::half> fp16_data(expected_elements);
  for (size_t i = 0; i < expected_elements; ++i) {
    fp16_data[i] = half_float::half(host_data[i]);
  }

  auto tensor = Tensor::empty({batch, size}, kFloat16, kAscend);
  tensor.alloc();

  void* device_ptr = tensor.ptr<void>();
  const size_t bytes = tensor.bytes();

  auto ret = aclrtMemcpy(
      device_ptr, bytes,
      fp16_data.data(), bytes,
      ACL_MEMCPY_HOST_TO_DEVICE);

  if (ret != ACL_SUCCESS) {
    MLLM_ACL_CHECK(ret);
  }

  return AscendTensorHandle(std::move(tensor), -1);
}

atb::Context* getGlobalAtbContext() {
  static atb::Context* ctx = nullptr;
  static std::once_flag init_flag;

  std::call_once(init_flag, [&] {
    // 1. Set Device
    auto acl_ret = aclrtSetDevice(0);
    MLLM_ACL_CHECK(acl_ret);

    // 2. Create Context
    auto ret = atb::CreateContext(&ctx);
    MLLM_ATB_CHECK(ret);

    // 3. Create Stream
    auto& stream = globalAtbStream();
    acl_ret = aclrtCreateStream(&stream);
    MLLM_ACL_CHECK(acl_ret);

    // 4. Set Stream
    ctx->SetExecuteStream(stream);
  });
  return ctx;
}

aclrtStream getGlobalAtbStream() {
  getGlobalAtbContext(); // Ensure initialized
  return globalAtbStream();
}

void syncGlobalAtbStream() {
  auto stream = globalAtbStream();
  if (stream != nullptr) {
    auto ret = aclrtSynchronizeStream(stream);
    MLLM_ACL_CHECK(ret);
  }
}

void fillAtbTensorDesc(const Tensor& t, atb::TensorDesc& desc) {
  desc.dtype = ACL_FLOAT16; // Currently hardcoded as per demo, can be expanded later
  desc.format = ACL_FORMAT_ND;

  auto shape = t.shape();
  desc.shape.dimNum = static_cast<uint64_t>(shape.size());
  for (uint64_t i = 0; i < desc.shape.dimNum; ++i) {
    desc.shape.dims[i] = static_cast<int64_t>(shape[i]);
  }
}

AscendDeviceMetaInfo::AscendDeviceMetaInfo() {
#ifndef ASCENDC_CPU_DEBUG
  // Initialize ACL to query devices
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    MLLM_ERROR("Failed to initialize ACL for device enumeration: {}", ret);
    return;
  }

  // Get device count
  uint32_t device_count = 0;
  ret = aclrtGetDeviceCount(&device_count);
  if (ret != ACL_SUCCESS) {
    MLLM_ERROR("Failed to get Ascend device count: {}", ret);
    aclFinalize();
    return;
  }

  // Collect info for each device
  for (uint32_t i = 0; i < device_count; ++i) {
    AscendDeviceInfo info;
    info.id = i;
    info.name = "Ascend Device " + std::to_string(i);

    // Set device to query its properties
    ret = aclrtSetDevice(i);
    if (ret == ACL_SUCCESS) {
      // Get memory information
      size_t free_mem = 0, total_mem = 0;
      ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
      if (ret == ACL_SUCCESS) {
        info.total_memory = total_mem;
        info.free_memory = free_mem;
      } else {
        info.total_memory = 0;
        info.free_memory = 0;
      }

      // SOC version - platform specific, set to unknown for now
      info.soc_version = "Unknown";
    } else {
      info.total_memory = 0;
      info.free_memory = 0;
      info.soc_version = "Unknown";
    }

    devices.push_back(info);
  }

  // Finalize ACL after enumeration
  aclFinalize();
#else
  // In CPU debug mode, add a dummy device
  AscendDeviceInfo info;
  info.id = 0;
  info.name = "Ascend CPU Debug Device";
  info.total_memory = 0;
  info.free_memory = 0;
  info.soc_version = "CPU_DEBUG";
  devices.push_back(info);
#endif
}

}  // namespace mllm::ascend

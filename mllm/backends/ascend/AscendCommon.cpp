// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendCommon.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/core/DataTypes.hpp"

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

  return {std::move(tensor), -1};
}

std::vector<float> copyAscendTensorToHost(const Tensor& t) {
  // Current implementation assumes FP16 tensor on Ascend.
  MLLM_RT_ASSERT(t.dtype() == kFloat16);

  // Use generic .to(kCPU) + CPU-side cast instead of raw aclrtMemcpy.
  // This goes through the X2X op we implemented for Ascend, keeping
  // all device transfer logic in one place.
  auto cpu_tensor = const_cast<Tensor&>(t).to(::mllm::kCPU);

  const size_t elem_cnt = cpu_tensor.numel();
  std::vector<float> host(elem_cnt);

  auto* src = cpu_tensor.ptr<mllm_fp16_t>();
  for (size_t i = 0; i < elem_cnt; ++i) {
    host[i] = static_cast<float>(src[i]);
  }
  return host;
}

bool verifyAscendTensor(const Tensor& t,
                        const std::vector<float>& expected,
                        float atol,
                        float rtol,
                        bool verbose,
                        std::vector<float>* actual_out) {
  auto actual = copyAscendTensorToHost(t);
  if (actual_out != nullptr) {
    *actual_out = actual;
  }

  if (actual.size() != expected.size()) {
    if (verbose) {
      std::cout << "[AscendVerify] size mismatch: actual " << actual.size()
                << " vs expected " << expected.size() << "\n";
    }
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::abs(actual[i] - expected[i]);
    const float thr = atol + rtol * std::abs(expected[i]);
    if (diff > thr) {
      ok = false;
      if (verbose) {
        std::cout << "[AscendVerify] idx " << i
                  << " expected " << expected[i]
                  << " got " << actual[i]
                  << " diff " << diff
                  << " thr " << thr << "\n";
      }
    }
  }

  if (verbose) {
    std::cout << (ok ? "[AscendVerify] OK" : "[AscendVerify] FAIL") << "\n";
  }
  return ok;
}

bool verifyAscendTensor(const Tensor& t,
                        const RefFn& ref_fn,
                        float atol,
                        float rtol,
                        bool verbose,
                        std::vector<float>* actual_out) {
  auto expected = ref_fn();
  return verifyAscendTensor(t, expected, atol, rtol, verbose, actual_out);
}

AscendTimer::AscendTimer(const char* tag, bool sync_before, bool sync_after)
    : tag_(tag),
      sync_before_(sync_before),
      sync_after_(sync_after) {
  if (sync_before_) {
    syncGlobalAtbStream();
  }
  start_ = std::chrono::high_resolution_clock::now();
}

AscendTimer::~AscendTimer() {
  if (sync_after_) {
    syncGlobalAtbStream();
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
  std::cout << "[AscendTimer] " << tag_ << " : " << ms << " ms\n";
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

void fillAtbTensor(const Tensor& t, atb::Tensor& atb_tensor) {
  fillAtbTensorDesc(t, atb_tensor.desc);
  atb_tensor.deviceData = reinterpret_cast<uint8_t*>(t.ptr<void>());
  // Use MLLM tensor's actual bytes as dataSize to match allocated memory
  atb_tensor.dataSize = t.bytes();
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

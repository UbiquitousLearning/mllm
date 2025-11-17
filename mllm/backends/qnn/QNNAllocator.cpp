// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include <algorithm>
#include <cstring>
#include <dlfcn.h>

namespace mllm::qnn {

namespace {
constexpr bool kVerboseQnnAllocatorLogs = false;
}  // namespace

#define QNN_ALLOCATOR_VERBOSE(...)                                     \
  do {                                                                 \
    if constexpr (kVerboseQnnAllocatorLogs) { MLLM_INFO(__VA_ARGS__); } \
  } while (0)

// specified in QNN doc
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

QNNAllocator::QNNAllocator() {
  void* libCdspHandle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libCdspHandle) { MLLM_ERROR_EXIT(1, "dlopen libcdsprpc.so failed"); }

  rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
  rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");
}

QNNAllocator::QNNAllocator(QNN_INTERFACE_VER_TYPE qnnInterface, void* context)
    : qnnInterface_(qnnInterface), context_(context) {
  MLLM_RT_ASSERT(context_ != nullptr);

  void* libCdspHandle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libCdspHandle) { MLLM_ERROR_EXIT(1, "dlopen libcdsprpc.so failed"); }

  rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
  rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");
}

QNNAllocator::~QNNAllocator() {
  for (auto iter = ptrToFdAndMemHandleMap_.begin(); iter != ptrToFdAndMemHandleMap_.end();) {
    Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&iter->second.second, 1);
    if (QNN_SUCCESS != deregisterRet) {
      MLLM_WARN("~QNNAllocator: memDeRegister failed during shutdown, status=0x{:x}", deregisterRet);
    }
    qnnMemPtrSet_.erase(iter->first);
    rpcmem_free(iter->first);
    iter = ptrToFdAndMemHandleMap_.erase(iter);
  }

  for (void* ptr : qnnMemPtrSet_) {
    rpcmem_free(ptr);
  }
  qnnMemPtrSet_.clear();
}

bool QNNAllocator::alloc(Storage* storage) {
  size_t request_bytes = allocSize(storage);
  uint8_t* ptr = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, request_bytes);

  MLLM_RT_ASSERT(ptr != nullptr);

  storage->ptr_ = ptr;
  qnnMemPtrSet_.insert(ptr);

  return true;
}

void QNNAllocator::free(Storage* storage) {
  auto ptr = storage->ptr_;
  
  // Early return if ptr is nullptr or not in qnnMemPtrSet_ (already freed or never allocated)
  if (ptr == nullptr) {
    // too noisy during decode; silently ignore nullptr frees
    return;
  }
  
  if (qnnMemPtrSet_.count(ptr) == 0) {
    QNN_ALLOCATOR_VERBOSE("QNNAllocator::free called for ptr={} that is not in qnnMemPtrSet_, ignoring", ptr);
    return;
  }

  void* alternative_ptr = nullptr;  // Another ptr using the same mem_handle, if any
  
  if (ptrToFdAndMemHandleMap_.count(ptr)) {
    auto iter = ptrToFdAndMemHandleMap_.find(ptr);
    auto mem_handle = iter->second.second;
    
    // Check if any other ptr is using the same mem_handle
    for (const auto& [other_ptr, fd_and_handle] : ptrToFdAndMemHandleMap_) {
      if (other_ptr != ptr && fd_and_handle.second == mem_handle) {
        alternative_ptr = other_ptr;
        break;
      }
    }
    
    // Only deRegister if this is the last ptr using this mem_handle
    if (alternative_ptr == nullptr) {
      auto status = qnnInterface_.memDeRegister(&mem_handle, 1);
      if (status != QNN_SUCCESS) {
        MLLM_WARN("QNNAllocator::free memDeRegister failed, status=0x{:x}, ptr={}, fd={}", status, ptr, iter->second.first);
      }
      // Remove from ptrToFdAndMemHandleMap_ and ptrToSizeMap_
      // The actual buffer will be freed later in the function
      ptrToFdAndMemHandleMap_.erase(iter);
      ptrToSizeMap_.erase(ptr);
    } else {
      QNN_ALLOCATOR_VERBOSE("QNNAllocator::free skipping deRegister for ptr={} because other ptrs use the mem_handle", ptr);
      ptrToFdAndMemHandleMap_.erase(iter);
      ptrToSizeMap_.erase(ptr);
    }
  } else {
    // ptr is in qnnMemPtrSet_ but not in ptrToFdAndMemHandleMap_
    // This means it was allocated but never registered (e.g., memRegister failed)
    // Just free the buffer without deRegister
    QNN_ALLOCATOR_VERBOSE("QNNAllocator::free freeing unregistered buffer ptr={}", ptr);
    qnnMemPtrSet_.erase(ptr);
    rpcmem_free(ptr);
    eraseTensorMappingsForPtr(ptr, "free(unregistered buffer)");
    clearLastRegistrationIfMatches(ptr, "free(unregistered buffer)");
    return;
  }
  
  // Update or keep tensor ID and name mappings
  // If mem_handle is still in use (alternative_ptr exists), update mappings to point to alternative_ptr
  // Otherwise, free the buffer and clear mappings
  if (alternative_ptr != nullptr) {
    // Update mappings to point to alternative_ptr instead of deleting them
    for (auto& entry : tensorIdToPtrMap_) {
      if (entry.second == ptr) { entry.second = alternative_ptr; }
    }
    for (auto& entry : tensorNameToPtrMap_) {
      if (entry.second == ptr) { entry.second = alternative_ptr; }
    }
    // Don't free the buffer here since alternative_ptr is still using it
    qnnMemPtrSet_.erase(ptr);
    clearLastRegistrationIfMatches(ptr, "free(ptr) -> redirected to alias");
  } else {
    // Since QNN doesn't support re-registering a deRegistered buffer (fd may be invalidated),
    // we should free the buffer immediately even if there are mappings.
    // The decode phase will allocate a new buffer when needed.
    qnnMemPtrSet_.erase(ptr);
    rpcmem_free(ptr);
    eraseTensorMappingsForPtr(ptr, "free(ptr) -> mem_handle released");
    clearLastRegistrationIfMatches(ptr, "free(ptr) -> mem_handle released");
  }
  storage->ptr_ = nullptr;
}

bool QNNAllocator::registerQnnTensorToSharedBuffer(Storage* storage, Qnn_Tensor_t& qnn_tensor) {
  MLLM_RT_ASSERT(storage != nullptr);
  void* ptr = storage->ptr_;

  // Make sure there has a memory that we can register to.
  MLLM_RT_ASSERT(ptr != nullptr);
  MLLM_RT_ASSERT(qnnMemPtrSet_.count(ptr));

  auto original_mem_type = QNN_TENSOR_GET_MEM_TYPE(qnn_tensor);
  Qnn_MemHandle_t original_mem_handle = QNN_TENSOR_GET_MEM_HANDLE(qnn_tensor);

  uint32_t tensor_id = QNN_TENSOR_GET_ID(qnn_tensor);
  const char* tensor_name_cstr = QNN_TENSOR_GET_NAME(qnn_tensor);
  std::string tensor_name = tensor_name_cstr ? tensor_name_cstr : "unknown";

  uint32_t rank = QNN_TENSOR_GET_RANK(qnn_tensor);
  uint32_t* dims_ptr = QNN_TENSOR_GET_DIMENSIONS(qnn_tensor);
  Qnn_DataType_t data_type = QNN_TENSOR_GET_DATA_TYPE(qnn_tensor);

  size_t element_bytes = 0;
  if (auto it = QNNDataTypeToSize.find(data_type); it != QNNDataTypeToSize.end()) { element_bytes = it->second; }

  size_t element_cnt = 1;
  std::vector<uint32_t> dims;
  dims.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    uint32_t dim = dims_ptr ? dims_ptr[i] : 0;
    dims.push_back(dim);
    element_cnt *= (dim == 0 ? 1 : dim);
  }
  size_t total_bytes = element_cnt * element_bytes;

  std::string shape_str = "[]";
  if (!dims.empty()) {
    shape_str = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
      shape_str += std::to_string(dims[i]);
      if (i + 1 < dims.size()) { shape_str += ", "; }
    }
    shape_str += "]";
  }

  QNN_ALLOCATOR_VERBOSE(
      "registerQnnTensorToSharedBuffer: ptr={}, tensor_id={}, tensor_name={}, tensorIdToPtrMap_.size()={}", ptr, tensor_id,
      tensor_name, tensorIdToPtrMap_.size());

  auto updateMappings = [&](void* mapped_ptr) {
    tensorIdToPtrMap_[tensor_id] = mapped_ptr;
    if (tensor_name != "unknown") { tensorNameToPtrMap_[tensor_name] = mapped_ptr; }
    ptrToSizeMap_[mapped_ptr] = total_bytes;
  };

  auto reuseExistingBuffer = [&](void* existing_ptr) -> bool {
    auto fd_handle_iter = ptrToFdAndMemHandleMap_.find(existing_ptr);
    if (fd_handle_iter == ptrToFdAndMemHandleMap_.end()) { return false; }

    Qnn_MemHandle_t existing_mem_handle = fd_handle_iter->second.second;
    size_t existing_size = ptrToSizeMap_.count(existing_ptr) > 0 ? ptrToSizeMap_[existing_ptr] : 0;

    if (existing_ptr != ptr) {
      size_t bytes_to_copy = total_bytes;
      if (existing_size > 0) { bytes_to_copy = std::min(bytes_to_copy, existing_size); }
      if (bytes_to_copy > 0) { std::memcpy(existing_ptr, ptr, bytes_to_copy); }

      if (qnnMemPtrSet_.count(ptr) > 0) {
        qnnMemPtrSet_.erase(ptr);
        rpcmem_free(ptr);
      }
      storage->ptr_ = existing_ptr;
    }

    QNN_TENSOR_SET_MEM_TYPE(qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(qnn_tensor, existing_mem_handle);
    updateMappings(existing_ptr);
    rememberLastRegistration(tensor_id, tensor_name, existing_ptr, existing_mem_handle, total_bytes);
    return true;
  };

  // if already registered, just set the mem handle
  if (ptrToFdAndMemHandleMap_.count(ptr) > 0) {
    Qnn_MemHandle_t mem_handle = ptrToFdAndMemHandleMap_[ptr].second;
    QNN_TENSOR_SET_MEM_TYPE(qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(qnn_tensor, mem_handle);
    updateMappings(ptr);
    rememberLastRegistration(tensor_id, tensor_name, ptr, mem_handle, total_bytes);
    return true;
  }

  // Check if we can reuse an existing buffer for the same tensor ID
  if (tensorIdToPtrMap_.count(tensor_id) > 0) {
    void* existing_ptr = tensorIdToPtrMap_[tensor_id];
    QNN_ALLOCATOR_VERBOSE("Found existing mapping for tensor_id={}: existing_ptr={}", tensor_id, existing_ptr);

    if (existing_ptr == nullptr) {
      QNN_ALLOCATOR_VERBOSE(
          "Existing mapping for tensor_id={} has nullptr ptr (buffer was freed), will register new buffer", tensor_id);
      tensorIdToPtrMap_.erase(tensor_id);
    } else if (reuseExistingBuffer(existing_ptr)) {
      return true;
    } else {
      MLLM_WARN("Existing ptr {} for tensor_id={} is no longer registered, removing from map", existing_ptr, tensor_id);
      tensorIdToPtrMap_.erase(tensor_id);
    }
  } else {
    QNN_ALLOCATOR_VERBOSE("No existing mapping found for tensor_id={}", tensor_id);
  }

  // Also check by tensor name as fallback (in case ID changed)
  if (tensor_name != "unknown" && tensorNameToPtrMap_.count(tensor_name) > 0) {
    void* existing_ptr = tensorNameToPtrMap_[tensor_name];
    QNN_ALLOCATOR_VERBOSE("Found existing mapping for tensor_name={}: existing_ptr={}", tensor_name, existing_ptr);

    if (existing_ptr == nullptr) {
      QNN_ALLOCATOR_VERBOSE(
          "Existing mapping for tensor_name={} has nullptr ptr (mem_handle was deRegistered), will register new buffer",
          tensor_name);
      tensorNameToPtrMap_.erase(tensor_name);
    } else if (reuseExistingBuffer(existing_ptr)) {
      return true;
    } else {
      MLLM_WARN("Existing ptr {} for tensor_name={} is no longer registered", existing_ptr, tensor_name);
      tensorNameToPtrMap_.erase(tensor_name);
    }
  }

  // Get the file id of this memory space.
  int mem_fd = rpcmem_to_fd(ptr);
  MLLM_RT_ASSERT(mem_fd != -1);

  // Make qnn memory descriptor. Set ION.
  Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
  mem_descriptor.memShape = {
      .numDim = rank,
      .dimSize = dims_ptr,
      .shapeConfig = nullptr,
  };
  mem_descriptor.dataType = data_type;
  mem_descriptor.memType = QNN_MEM_TYPE_ION;
  mem_descriptor.ionInfo.fd = mem_fd;
  QNN_TENSOR_SET_MEM_TYPE(qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);

  // Register to QNN memory
  Qnn_MemHandle_t mem_handle = QNN_TENSOR_GET_MEM_HANDLE(qnn_tensor);
  auto status = qnnInterface_.memRegister(context_, &mem_descriptor, 1u, &mem_handle);

  if (status != QNN_SUCCESS) {
    auto stats = getRegisteredBufferStats();
    MLLM_ERROR("QNNAllocator::registerQnnTensorToSharedBuffer memRegister failed, status=0x{:x}, ptr={}, fd={}, bytes={}, "
               "shape={}, dtype={}, tensor_id={}, tensor_name={}",
               status, ptr, mem_fd, total_bytes, shape_str, static_cast<int>(mem_descriptor.dataType), tensor_id, tensor_name);
    MLLM_ERROR("Current registered buffers: {} buffers, {} MB", stats.count, stats.total_bytes / (1024 * 1024));

    // Try to reuse existing buffer for the same tensor ID or name as fallback
    bool fallback_success = false;
    if (tensorIdToPtrMap_.count(tensor_id) > 0) {
      void* existing_ptr = tensorIdToPtrMap_[tensor_id];
      if (existing_ptr != nullptr) {
        MLLM_WARN("Fallback: Reusing existing buffer by ID for tensor_id={}, tensor_name={}, old_ptr={}, new_ptr={}",
                  tensor_id, tensor_name, existing_ptr, ptr);
        fallback_success = reuseExistingBuffer(existing_ptr);
      }
    }
    if (!fallback_success && tensor_name != "unknown" && tensorNameToPtrMap_.count(tensor_name) > 0) {
      void* existing_ptr = tensorNameToPtrMap_[tensor_name];
      if (existing_ptr != nullptr) {
        MLLM_WARN("Fallback: Reusing existing buffer by name for tensor_id={}, tensor_name={}, old_ptr={}, new_ptr={}",
                  tensor_id, tensor_name, existing_ptr, ptr);
        fallback_success = reuseExistingBuffer(existing_ptr);
      }
    }

    if (!fallback_success && hasLastRegistrationInfo_) {
      bool same_tensor_id = tensor_id != 0 && tensor_id == lastRegistrationInfo_.tensor_id;
      bool same_tensor_name = tensor_name != "unknown" && !tensor_name.empty()
                              && tensor_name == lastRegistrationInfo_.tensor_name;
      bool ptr_still_registered = lastRegistrationInfo_.ptr != nullptr
                                  && ptrToFdAndMemHandleMap_.count(lastRegistrationInfo_.ptr) > 0;
      if ((same_tensor_id || same_tensor_name) && ptr_still_registered) {
        MLLM_WARN("Fallback: Reusing last successful buffer for tensor_id={}, tensor_name={}, old_ptr={}, new_ptr={}",
                  tensor_id, tensor_name, lastRegistrationInfo_.ptr, ptr);
        fallback_success = reuseExistingBuffer(lastRegistrationInfo_.ptr);
      } else {
        MLLM_WARN("Fallback: Last registration info unusable for tensor_id={}, tensor_name={}, "
                  "same_tensor_id={}, same_tensor_name={}, ptr_registered={}",
                  tensor_id, tensor_name, same_tensor_id, same_tensor_name, ptr_still_registered);
      }
    }

    if (!fallback_success) {
      MLLM_ERROR("QNNAllocator::registerQnnTensorToSharedBuffer: memRegister failed and fallback also failed. "
                 "Buffer ptr={} will be freed, tensor registration cannot proceed.", ptr);

      if (qnnMemPtrSet_.count(ptr) > 0) {
        qnnMemPtrSet_.erase(ptr);
        rpcmem_free(ptr);
        storage->ptr_ = nullptr;
        eraseTensorMappingsForPtr(ptr, "register failure -> freed ptr");
        clearLastRegistrationIfMatches(ptr, "register failure -> freed ptr");
        QNN_ALLOCATOR_VERBOSE("QNNAllocator::registerQnnTensorToSharedBuffer: Freed ptr={} ({} bytes) after failure", ptr,
                              total_bytes);
      }

      QNN_TENSOR_SET_MEM_HANDLE(qnn_tensor, original_mem_handle);
      QNN_TENSOR_SET_MEM_TYPE(qnn_tensor, original_mem_type);
      return false;
    }
    return true;
  } else {
    QNN_ALLOCATOR_VERBOSE("Register shared buffer ptr={}, fd={}, bytes={}, shape={}, dtype={}, tensor_id={}, tensor_name={}",
                          ptr, mem_fd, total_bytes, shape_str, static_cast<int>(mem_descriptor.dataType), tensor_id,
                          tensor_name);
  }

  QNN_TENSOR_SET_MEM_HANDLE(qnn_tensor, mem_handle);

  ptrToFdAndMemHandleMap_.insert({ptr, {mem_fd, mem_handle}});
  updateMappings(ptr);
  rememberLastRegistration(tensor_id, tensor_name, ptr, mem_handle, total_bytes);
  return true;
}

void QNNAllocator::deRegisterQnnTensorFromSharedBuffer(void* ptr) {
  auto iter = ptrToFdAndMemHandleMap_.find(ptr);
  if (iter == ptrToFdAndMemHandleMap_.end()) { return; }

  Qnn_ErrorHandle_t status = qnnInterface_.memDeRegister(&(iter->second.second), 1);
  if (status != QNN_SUCCESS) {
    MLLM_WARN("QNNAllocator::deRegisterQnnTensorFromSharedBuffer memDeRegister failed, status=0x{:x}, ptr={}, fd={}", status,
              ptr, iter->second.first);
  }

  ptrToFdAndMemHandleMap_.erase(iter);
  ptrToSizeMap_.erase(ptr);
  eraseTensorMappingsForPtr(ptr, "explicit deRegister");
  clearLastRegistrationIfMatches(ptr, "explicit deRegister");
}

QNNAllocator::BufferStats QNNAllocator::getRegisteredBufferStats() const {
  BufferStats stats{};
  stats.count = ptrToFdAndMemHandleMap_.size();
  stats.total_bytes = 0;
  
  for (const auto& [ptr, size] : ptrToSizeMap_) {
    stats.total_bytes += size;
  }
  
  return stats;
}

bool QNNAllocator::isRegistered(void* ptr) const {
  return ptrToFdAndMemHandleMap_.count(ptr) > 0;
}

size_t QNNAllocator::getRegisteredBufferSize(void* ptr) const {
  auto it = ptrToSizeMap_.find(ptr);
  if (it == ptrToSizeMap_.end()) { return 0; }
  return it->second;
}

void QNNAllocator::eraseTensorMappingsForPtr(void* ptr, std::string_view reason) {
  if (ptr == nullptr) { return; }

  for (auto it = tensorIdToPtrMap_.begin(); it != tensorIdToPtrMap_.end();) {
    if (it->second == ptr) {
      it = tensorIdToPtrMap_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = tensorNameToPtrMap_.begin(); it != tensorNameToPtrMap_.end();) {
    if (it->second == ptr) {
      it = tensorNameToPtrMap_.erase(it);
    } else {
      ++it;
    }
  }
}

void QNNAllocator::rememberLastRegistration(uint32_t tensor_id, const std::string& tensor_name, void* ptr,
                                            Qnn_MemHandle_t mem_handle, size_t total_bytes) {
  if (ptr == nullptr || mem_handle == nullptr) { return; }
  lastRegistrationInfo_.tensor_id = tensor_id;
  lastRegistrationInfo_.tensor_name = tensor_name;
  lastRegistrationInfo_.ptr = ptr;
  lastRegistrationInfo_.mem_handle = mem_handle;
  lastRegistrationInfo_.bytes = total_bytes;
  hasLastRegistrationInfo_ = true;
  // Note: Remembered registration info is used as fallback mechanism, logging removed for performance
}

void QNNAllocator::clearLastRegistrationIfMatches(void* ptr, std::string_view reason) {
  if (!hasLastRegistrationInfo_ || ptr == nullptr) { return; }
  if (lastRegistrationInfo_.ptr == ptr) {
    lastRegistrationInfo_ = {};
    hasLastRegistrationInfo_ = false;
  }
}

#undef QNN_ALLOCATOR_VERBOSE

std::shared_ptr<QNNAllocator> createQNNAllocator() { return std::make_shared<QNNAllocator>(); }

}  // namespace mllm::qnn

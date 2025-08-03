// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm {

void shutdownContext() {
  auto all_threads = Context::instance().refSessionThreads();
  auto this_thread = Context::instance().thisThread();
  for (auto& tcb : all_threads) {
    if (tcb.first != this_thread->system_tid) {
      tcb.second->attached_contexts._ref_raw_data().clear();
      tcb.second->ir_context = nullptr;
      tcb.second->trace_mode = false;
    }
  }
  ::mllm::cleanThisThread();
}

void setRandomSeed(uint64_t seed) { Context::instance().setRandomSeed(seed); }

void setMaximumNumThreads(uint32_t num_threads) {
  // TODO
}

void memoryReport() { Context::instance().memoryManager()->report(); }

bool isOpenCLAvailable() {
  // TODO
  return false;
}

bool isQnnAvailable() {
  // TODO
  return false;
}

void perfStart() { Context::instance().setPerfMode(true); }

void perfEnd() {
  Context::instance().setPerfMode(false);
  Context::instance().getPerfFile()->finalize();
}

void cleanThisThread() {
  Context::instance().thisThread()->attached_contexts._ref_raw_data().clear();
  Context::instance().thisThread()->ir_context = nullptr;
  Context::instance().thisThread()->trace_mode = false;
}

PerfFile::ptr_t getPerfFile() { return Context::instance().getPerfFile(); }

SessionTCB::ptr_t thisThread() { return Context::instance().thisThread(); }

ParameterFile::ptr_t load(const std::string& file_name, ModelFileVersion v, DeviceTypes map_2_device) {
  if (v == ModelFileVersion::kV1 && map_2_device == kCPU) {
    return ParameterFileIOImpl<kCPU, ModelFileVersion::kV1>::read(file_name);
  }

  // return empty if not match all.
  return ParameterFile::create();
}

namespace test {

// |a - b| <= atol + rtol * |b|
AllCloseResult allClose(const Tensor& a, const Tensor& b, float rtol, float atol, bool equal_nan) {
  AllCloseResult result;
  if (a.isNil() || b.isNil()) {
    MLLM_WARN("a or b is nil");
    return result;
  }

  if (a.shape() != b.shape()) {
    MLLM_WARN("shape not match");
    return result;
  }

  if (a.dtype() != b.dtype()) {
    MLLM_WARN("dtype not match, {} != {}", nameOfType(a.dtype()), nameOfType(b.dtype()));
    return result;
  }

  result.total_elements = a.numel();

  switch (a.dtype()) {
    case kFloat32: {
      mllm_fp32_t* a_ptr = a.ptr<mllm_fp32_t>();
      mllm_fp32_t* b_ptr = b.ptr<mllm_fp32_t>();

      for (size_t i = 0; i < a.numel(); ++i) {
        mllm_fp32_t a_val = a_ptr[i];
        mllm_fp32_t b_val = b_ptr[i];

        if (std::isnan(a_val) && std::isnan(b_val)) {
          if (equal_nan) {
            continue;
          } else {
            result.mismatched_elements++;
            continue;
          }
        }

        if (std::isinf(a_val) && std::isinf(b_val)) {
          if (a_val == b_val) {
            continue;
          } else {
            result.mismatched_elements++;
            mllm_fp32_t abs_diff = std::abs(a_val - b_val);
            result.max_absolute_diff = std::max(result.max_absolute_diff, abs_diff);
            continue;
          }
        }

        mllm_fp32_t abs_diff = std::abs(a_val - b_val);
        mllm_fp32_t rel_diff = abs_diff / (std::abs(b_val) + 1e-12f);

        result.max_absolute_diff = std::max(result.max_absolute_diff, abs_diff);
        result.max_relative_diff = std::max(result.max_relative_diff, rel_diff);

        if (abs_diff > (atol + rtol * std::abs(b_val))) { result.mismatched_elements++; }
      }
      break;
    }
    case kFloat16: {
      mllm_fp16_t* a_ptr = a.ptr<mllm_fp16_t>();
      mllm_fp16_t* b_ptr = b.ptr<mllm_fp16_t>();

      for (size_t i = 0; i < a.numel(); ++i) {
        mllm_fp32_t a_val = (mllm_fp32_t)a_ptr[i];
        mllm_fp32_t b_val = (mllm_fp32_t)b_ptr[i];

        if (std::isnan(a_val) && std::isnan(b_val)) {
          if (equal_nan) {
            continue;
          } else {
            result.mismatched_elements++;
            continue;
          }
        }

        if (std::isinf(a_val) && std::isinf(b_val)) {
          if (a_val == b_val) {
            continue;
          } else {
            result.mismatched_elements++;
            mllm_fp32_t abs_diff = std::abs(a_val - b_val);
            result.max_absolute_diff = std::max(result.max_absolute_diff, abs_diff);
            continue;
          }
        }

        mllm_fp32_t abs_diff = std::abs(a_val - b_val);
        mllm_fp32_t rel_diff = abs_diff / (std::abs(b_val) + 1e-12f);

        result.max_absolute_diff = std::max(result.max_absolute_diff, abs_diff);
        result.max_relative_diff = std::max(result.max_relative_diff, rel_diff);

        if (abs_diff > (atol + rtol * std::abs(b_val))) { result.mismatched_elements++; }
      }
      break;
    }
    case kInt8: {
      const mllm_int8_t* a_ptr = a.ptr<mllm_int8_t>();
      const mllm_int8_t* b_ptr = b.ptr<mllm_int8_t>();
      __allCloseProcessIntegerType<mllm_int8_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kInt16: {
      const mllm_int16_t* a_ptr = a.ptr<mllm_int16_t>();
      const mllm_int16_t* b_ptr = b.ptr<mllm_int16_t>();
      __allCloseProcessIntegerType<mllm_int16_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kInt32: {
      const mllm_int32_t* a_ptr = a.ptr<mllm_int32_t>();
      const mllm_int32_t* b_ptr = b.ptr<mllm_int32_t>();
      __allCloseProcessIntegerType<mllm_int32_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kInt64: {
      const mllm_int64_t* a_ptr = a.ptr<mllm_int64_t>();
      const mllm_int64_t* b_ptr = b.ptr<mllm_int64_t>();
      __allCloseProcessIntegerType<mllm_int64_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kUInt8: {
      const mllm_uint8_t* a_ptr = a.ptr<mllm_uint8_t>();
      const mllm_uint8_t* b_ptr = b.ptr<mllm_uint8_t>();
      __allCloseProcessIntegerType<mllm_uint8_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kUInt16: {
      const mllm_uint16_t* a_ptr = a.ptr<mllm_uint16_t>();
      const mllm_uint16_t* b_ptr = b.ptr<mllm_uint16_t>();
      __allCloseProcessIntegerType<mllm_uint16_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kUInt32: {
      const mllm_uint32_t* a_ptr = a.ptr<mllm_uint32_t>();
      const mllm_uint32_t* b_ptr = b.ptr<mllm_uint32_t>();
      __allCloseProcessIntegerType<mllm_uint32_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    case kUInt64: {
      const mllm_uint64_t* a_ptr = a.ptr<mllm_uint64_t>();
      const mllm_uint64_t* b_ptr = b.ptr<mllm_uint64_t>();
      __allCloseProcessIntegerType<mllm_uint64_t>(a_ptr, b_ptr, a.numel(), result, rtol, atol);
      break;
    }
    default:
      MLLM_WARN("allClose unsupported data type, return false as default");
      result.is_close = false;
      return result;
  }
  result.is_close = (result.mismatched_elements == 0);
  return result;
}

}  // namespace test

namespace async {

std::vector<Tensor> wait(std::pair<TaskResult::sender_t, Task::ptr_t>& sender) {
  stdexec::sync_wait(std::move(sender.first));
  return sender.second->outputs;
}

}  // namespace async

}  // namespace mllm

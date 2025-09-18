// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <sstream>
#include <fmt/core.h>
#include <fmt/base.h>
#include <fmt/ostream.h>

#include "mllm/mllm.hpp"
#include "mllm/ffi/Object.hh"

namespace mllm::ffi {
//===----------------------------------------------------------------------===//
// Helper Functions. [Test if FFI works]
//===----------------------------------------------------------------------===//
void echo(const tvm::ffi::String& a) { fmt::print("{}", a.c_str()); }

::mllm::ffi::Tensor empty(const tvm::ffi::Shape& shape, const ::mllm::ffi::DType& dtype, const ::mllm::ffi::Device& device) {
  ::mllm::Tensor::shape_t mllm_shape;
  for (int64_t i : shape) { mllm_shape.push_back(i); }
  auto container = ::mllm::Tensor::empty(mllm_shape, dtype->dtype, device->device);
  container.alloc();
  return ::mllm::ffi::Tensor(container);
}

}  // namespace mllm::ffi

//===----------------------------------------------------------------------===//
// REGISTER: Helper Functions. [Test if FFI works]
//===----------------------------------------------------------------------===//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Runtime related
  refl::GlobalDef().def("mllm.echo", mllm::ffi::echo);
  refl::GlobalDef().def("mllm.initialize_context", mllm::initializeContext);
  refl::GlobalDef().def("mllm.shutdown_context", mllm::shutdownContext);

  // Primitives
  refl::GlobalDef().def("mllm.cpu_", []() -> mllm::ffi::Device { return mllm::ffi::Device(::mllm::DeviceTypes::kCPU); });
  refl::GlobalDef().def("mllm.cuda_", []() -> mllm::ffi::Device { return mllm::ffi::Device(::mllm::DeviceTypes::kCUDA); });
  refl::GlobalDef().def("mllm.qnn_", []() -> mllm::ffi::Device { return mllm::ffi::Device(::mllm::DeviceTypes::kQNN); });
  refl::GlobalDef().def("mllm.float32_", []() -> mllm::ffi::DType { return mllm::ffi::DType(::mllm::DataTypes::kFloat32); });
  refl::GlobalDef().def("mllm.float16_", []() -> mllm::ffi::DType { return mllm::ffi::DType(::mllm::DataTypes::kFloat16); });
  refl::GlobalDef().def("mllm.bfloat16_", []() -> mllm::ffi::DType { return mllm::ffi::DType(::mllm::DataTypes::kBFloat16); });
}

//===----------------------------------------------------------------------===//
// REGISTER: Tensor Object
//===----------------------------------------------------------------------===//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Tensor related
  refl::GlobalDef().def("mllm.empty", mllm::ffi::empty);
  refl::GlobalDef().def("mllm.from_torch", [](const tvm::ffi::Tensor& t) -> mllm::ffi::Tensor {
    auto dl_pack = t.get()->ToDLPack();
    return ::mllm::ffi::Tensor(mllm::ffi::__from_dlpack(dl_pack));
  });

  refl::GlobalDef().def("mllm.from_numpy", [](const tvm::ffi::Tensor& t) -> mllm::ffi::Tensor {
    auto dl_pack = t.get()->ToDLPack();
    return ::mllm::ffi::Tensor(mllm::ffi::__from_dlpack(dl_pack));
  });

  refl::GlobalDef().def("mllm.Device.to_pod",
                        [](const mllm::ffi::Device& obj) -> int32_t { return (int32_t)obj.get()->device; });
  refl::GlobalDef().def("mllm.DType.to_pod", [](const mllm::ffi::DType& obj) -> int32_t { return (int32_t)obj.get()->dtype; });

  refl::ObjectDef<mllm::ffi::TensorObj>().def_static(
      "__create__", []() -> mllm::ffi::Tensor { return ::mllm::ffi::Tensor(mllm::Tensor::nil()); });
  refl::GlobalDef().def("mllm.Tensor.str", [](const mllm::ffi::Tensor& obj) -> std::string {
    std::stringstream ss;
    fmt::print(ss, "{}", obj.get()->mllm_tensor_);
    return ss.str();
  });
  refl::GlobalDef().def("mllm.Tensor.shape", [](const mllm::ffi::Tensor& obj) -> tvm::ffi::Shape {
    const auto& mllm_shape = obj.get()->mllm_tensor_.shape();
    std::vector<int64_t> shape_int64(mllm_shape.begin(), mllm_shape.end());
    tvm::ffi::Shape ret(shape_int64);
    return ret;
  });
  refl::GlobalDef().def("mllm.Tensor.dtype", [](const mllm::ffi::Tensor& obj) -> ::mllm::ffi::DType {
    return ::mllm::ffi::DType(obj.get()->mllm_tensor_.dtype());
  });
  refl::GlobalDef().def("mllm.Tensor.device", [](const mllm::ffi::Tensor& obj) -> ::mllm::ffi::Device {
    return ::mllm::ffi::Device(obj.get()->mllm_tensor_.device());
  });
  refl::GlobalDef().def("mllm.Tensor.tobytes", [](const mllm::ffi::Tensor& obj) -> ::tvm::ffi::Bytes {
    return {obj.get()->mllm_tensor_.ptr<char>(), obj.get()->mllm_tensor_.bytes()};
  });
}

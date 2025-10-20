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
#include "mllm/nn/Functional.hpp"
#include "mllm/ffi/Object.hh"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/engine/service/Service.hpp"

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

  // Tensor operations bindings
  refl::GlobalDef().def("mllm.Tensor.alloc", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto tensor = obj.get()->mllm_tensor_;
    tensor.alloc();
    return mllm::ffi::Tensor(tensor);
  });
  refl::GlobalDef().def("mllm.zeros",
                        [](const tvm::ffi::Shape& shape, const ::mllm::ffi::DType& dtype,
                           const ::mllm::ffi::Device& device) -> mllm::ffi::Tensor {
                          ::mllm::Tensor::shape_t mllm_shape;
                          for (int64_t i : shape) { mllm_shape.push_back(i); }
                          auto tensor = ::mllm::Tensor::zeros(mllm_shape, dtype->dtype, device->device);
                          return ::mllm::ffi::Tensor(tensor);
                        });
  refl::GlobalDef().def("mllm.ones",
                        [](const tvm::ffi::Shape& shape, const ::mllm::ffi::DType& dtype,
                           const ::mllm::ffi::Device& device) -> mllm::ffi::Tensor {
                          ::mllm::Tensor::shape_t mllm_shape;
                          for (int64_t i : shape) { mllm_shape.push_back(i); }
                          auto tensor = ::mllm::Tensor::ones(mllm_shape, dtype->dtype, device->device);
                          return ::mllm::ffi::Tensor(tensor);
                        });
  refl::GlobalDef().def("mllm.arange",
                        [](float start, float end, float step, const ::mllm::ffi::DType& dtype,
                           const ::mllm::ffi::Device& device) -> mllm::ffi::Tensor {
                          auto tensor = ::mllm::Tensor::arange(start, end, step, dtype->dtype, device->device);
                          return ::mllm::ffi::Tensor(tensor);
                        });
  refl::GlobalDef().def("mllm.random",
                        [](const tvm::ffi::Shape& shape, float start, float end, const ::mllm::ffi::DType& dtype,
                           const ::mllm::ffi::Device& device) -> mllm::ffi::Tensor {
                          ::mllm::Tensor::shape_t mllm_shape;
                          for (int64_t i : shape) { mllm_shape.push_back(i); }
                          auto tensor = ::mllm::Tensor::random(mllm_shape, start, end, dtype->dtype, device->device);
                          return ::mllm::ffi::Tensor(tensor);
                        });

  // Binary operations with tensor
  refl::GlobalDef().def("mllm.Tensor.add", [](const mllm::ffi::Tensor& lhs, const mllm::ffi::Tensor& rhs) -> mllm::ffi::Tensor {
    auto lhs_tensor = lhs.get()->mllm_tensor_;
    auto rhs_tensor = rhs.get()->mllm_tensor_;
    return mllm::ffi::Tensor(lhs_tensor + rhs_tensor);
  });
  refl::GlobalDef().def("mllm.Tensor.sub", [](const mllm::ffi::Tensor& lhs, const mllm::ffi::Tensor& rhs) -> mllm::ffi::Tensor {
    auto lhs_tensor = lhs.get()->mllm_tensor_;
    auto rhs_tensor = rhs.get()->mllm_tensor_;
    return mllm::ffi::Tensor(lhs_tensor - rhs_tensor);
  });
  refl::GlobalDef().def("mllm.Tensor.mul", [](const mllm::ffi::Tensor& lhs, const mllm::ffi::Tensor& rhs) -> mllm::ffi::Tensor {
    auto lhs_tensor = lhs.get()->mllm_tensor_;
    auto rhs_tensor = rhs.get()->mllm_tensor_;
    return mllm::ffi::Tensor(lhs_tensor * rhs_tensor);
  });
  refl::GlobalDef().def("mllm.Tensor.div", [](const mllm::ffi::Tensor& lhs, const mllm::ffi::Tensor& rhs) -> mllm::ffi::Tensor {
    auto lhs_tensor = lhs.get()->mllm_tensor_;
    auto rhs_tensor = rhs.get()->mllm_tensor_;
    return mllm::ffi::Tensor(lhs_tensor / rhs_tensor);
  });

  // Binary operations with scalar
  refl::GlobalDef().def("mllm.Tensor.add_scalar", [](const mllm::ffi::Tensor& lhs, float rhs) -> mllm::ffi::Tensor {
    auto result = lhs.get()->mllm_tensor_;
    result = result + rhs;
    return mllm::ffi::Tensor(result);
  });
  refl::GlobalDef().def("mllm.Tensor.sub_scalar", [](const mllm::ffi::Tensor& lhs, float rhs) -> mllm::ffi::Tensor {
    auto result = lhs.get()->mllm_tensor_;
    result = result - rhs;
    return mllm::ffi::Tensor(result);
  });
  refl::GlobalDef().def("mllm.Tensor.mul_scalar", [](const mllm::ffi::Tensor& lhs, float rhs) -> mllm::ffi::Tensor {
    auto result = lhs.get()->mllm_tensor_;
    result = result * rhs;
    return mllm::ffi::Tensor(result);
  });
  refl::GlobalDef().def("mllm.Tensor.div_scalar", [](const mllm::ffi::Tensor& lhs, float rhs) -> mllm::ffi::Tensor {
    auto result = lhs.get()->mllm_tensor_;
    result = result / rhs;
    return mllm::ffi::Tensor(result);
  });

  // Unary operations
  refl::GlobalDef().def("mllm.Tensor.abs", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.abs());
  });
  refl::GlobalDef().def("mllm.Tensor.neg", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(-result);
  });
  refl::GlobalDef().def("mllm.Tensor.clip",
                        [](const mllm::ffi::Tensor& obj, float min_val, float max_val) -> mllm::ffi::Tensor {
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.clip(min_val, max_val));
                        });

  // Reduction operations
  refl::GlobalDef().def("mllm.Tensor.min", [](const mllm::ffi::Tensor& obj, int32_t dim, bool keep_dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.min(dim, keep_dim));
  });

  refl::GlobalDef().def("mllm.Tensor.max", [](const mllm::ffi::Tensor& obj, int32_t dim, bool keep_dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.max(dim, keep_dim));
  });

  refl::GlobalDef().def("mllm.Tensor.sum", [](const mllm::ffi::Tensor& obj, int32_t dim, bool keep_dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.sum(dim, keep_dim));
  });

  refl::GlobalDef().def("mllm.Tensor.mean", [](const mllm::ffi::Tensor& obj, int32_t dim, bool keep_dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.mean(dim, keep_dim));
  });

  // Shape operations
  refl::GlobalDef().def("mllm.Tensor.transpose", [](const mllm::ffi::Tensor& obj, int dim0, int dim1) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.transpose(dim0, dim1));
  });
  refl::GlobalDef().def("mllm.Tensor.T", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.T());
  });

  refl::GlobalDef().def("mllm.Tensor.view",
                        [](const mllm::ffi::Tensor& obj, const tvm::ffi::Shape& indicies) -> mllm::ffi::Tensor {
                          ::mllm::Tensor::shape_t mllm_indicies;
                          for (int64_t i : indicies) { mllm_indicies.push_back(i); }
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.view(mllm_indicies));
                        });
  refl::GlobalDef().def("mllm.Tensor.unsqueeze", [](const mllm::ffi::Tensor& obj, int32_t dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.unsqueeze(dim));
  });
  refl::GlobalDef().def("mllm.Tensor.squeeze", [](const mllm::ffi::Tensor& obj, int32_t dim) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.squeeze(dim));
  });
  refl::GlobalDef().def("mllm.Tensor.permute",
                        [](const mllm::ffi::Tensor& obj, const tvm::ffi::Shape& indices) -> mllm::ffi::Tensor {
                          ::mllm::Tensor::shape_t mllm_indices;
                          for (int64_t i : indices) { mllm_indices.push_back(i); }
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.permute(mllm_indices));
                        });
  refl::GlobalDef().def("mllm.Tensor.contiguous", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.contiguous());
  });
  refl::GlobalDef().def("mllm.Tensor.clone", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.clone());
  });
  refl::GlobalDef().def("mllm.Tensor.repeat",
                        [](const mllm::ffi::Tensor& obj, int32_t multiplier, int32_t dim) -> mllm::ffi::Tensor {
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.repeat(multiplier, dim));
                        });

  // Device and dtype conversion
  refl::GlobalDef().def("mllm.Tensor.to_device",
                        [](const mllm::ffi::Tensor& obj, const ::mllm::ffi::Device& device) -> mllm::ffi::Tensor {
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.to(device->device));
                        });
  refl::GlobalDef().def("mllm.Tensor.to_dtype",
                        [](const mllm::ffi::Tensor& obj, const ::mllm::ffi::DType& dtype) -> mllm::ffi::Tensor {
                          auto result = obj.get()->mllm_tensor_;
                          return mllm::ffi::Tensor(result.to(dtype->dtype));
                        });
  refl::GlobalDef().def("mllm.Tensor.cpu", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.cpu());
  });
  refl::GlobalDef().def("mllm.Tensor.cuda", [](const mllm::ffi::Tensor& obj) -> mllm::ffi::Tensor {
    auto result = obj.get()->mllm_tensor_;
    return mllm::ffi::Tensor(result.cuda());
  });

  // Property accessors
  refl::GlobalDef().def("mllm.Tensor.get_name",
                        [](const mllm::ffi::Tensor& obj) -> std::string { return obj.get()->mllm_tensor_.name(); });
  refl::GlobalDef().def("mllm.Tensor.set_name", [](const mllm::ffi::Tensor& obj, const std::string& name) -> mllm::ffi::Tensor {
    auto tensor = obj.get()->mllm_tensor_;
    tensor.setName(name);
    return mllm::ffi::Tensor(tensor);
  });
  refl::GlobalDef().def("mllm.Tensor.numel",
                        [](const mllm::ffi::Tensor& obj) -> size_t { return obj.get()->mllm_tensor_.numel(); });
  refl::GlobalDef().def("mllm.Tensor.rank",
                        [](const mllm::ffi::Tensor& obj) -> size_t { return obj.get()->mllm_tensor_.rank(); });
  refl::GlobalDef().def("mllm.Tensor.is_contiguous",
                        [](const mllm::ffi::Tensor& obj) -> bool { return obj.get()->mllm_tensor_.isContiguous(); });
}

//===----------------------------------------------------------------------===//
// REGISTER: NN Functions.
//===----------------------------------------------------------------------===//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def("mllm.matmul_impl_default", []() -> int { return (int)(::mllm::aops::MatMulOpType::kDefault); });
  refl::GlobalDef().def("mllm.matmul_impl_gguf", []() -> int { return (int)(::mllm::aops::MatMulOpType::kGGUF); });
  refl::GlobalDef().def("mllm.matmul_impl_blas", []() -> int { return (int)(::mllm::aops::MatMulOpType::kBLAS); });
  refl::GlobalDef().def("mllm.matmul_impl_mllmblas", []() -> int { return (int)(::mllm::aops::MatMulOpType::kMllmBlas); });
  refl::GlobalDef().def("mllm.nn.functional.matmul",
                        [](const mllm::ffi::Tensor& lhs, const mllm::ffi::Tensor& rhs, bool transpose_A, bool transpose_B,
                           int type) -> ::mllm::ffi::Tensor {
                          auto ll = lhs.get()->mllm_tensor_;
                          auto rr = rhs.get()->mllm_tensor_;
                          return ::mllm::ffi::Tensor{::mllm::nn::functional::matmul(ll, rr, transpose_A, transpose_B,
                                                                                    (::mllm::aops::MatMulOpType)type)};
                        });
}

//===----------------------------------------------------------------------===//
// REGISTER: Service Functions.
//===----------------------------------------------------------------------===//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def("mllm.service.startService",
                        [](int work_threads = 1) -> void { ::mllm::service::startService(work_threads); });
  refl::GlobalDef().def("mllm.service.stopService", []() -> void { ::mllm::service::stopService(); });
  refl::GlobalDef().def("mllm.service.sendRequest",
                        [](const std::string& json_str) -> void { ::mllm::service::sendRequest(json_str); });
  refl::GlobalDef().def("mllm.service.getResponse",
                        [](const std::string& id) -> std::string { return ::mllm::service::getResponse(id); });
  refl::GlobalDef().def("mllm.service.insertSession",
                        [](const std::string& session_id, const mllm::ffi::Session& session) -> void {
                          ::mllm::service::insertSession(session_id, session.get()->session_ptr_);
                        });
}

//===----------------------------------------------------------------------===//
// REGISTER: Quantize && Packing Functions.
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "mllm.quantize_pack.KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk",
      [](const std::string& tile_cfg_name, const mllm::ffi::Tensor& ffi_weight,
         const mllm::ffi::Tensor& ffi_bias) -> mllm::ffi::Tensor {
        ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper_;

        auto weight = ffi_weight.get()->mllm_tensor_;
        auto bias = ffi_bias.get()->mllm_tensor_;

        auto weight_shape = weight.shape();
        auto out_channels = weight_shape[0];
        auto in_channels = weight_shape[1];

        mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg;
        tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;

        if (tile_cfg_name == "qai8dxp1x8_qsi4c32p4x8_1x4x32") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;
        } else if (tile_cfg_name == "qai8dxp1x8_qsi4c32p8x8_1x8x32") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
        } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p4x8_8x4x32") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32;
        } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p4x8_16x4x32") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32;
        } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p8x8_4x8x32") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;
        } else if (tile_cfg_name == "qai8dxp1x4_qsi4c32p4x4_1x4") {
          tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4;
        }

        // pack_rhs_size return byte size.
        int32_t new_weights_size = kai_helper_.quant_pack_rhs_size(out_channels, in_channels, tile_cfg);

        // NOTE:
        // We used a flatter byte buffer to represent the packed weight.
        // The packed weight can't be read or manipulated as a normal tensor.
        mllm::Tensor new_weights = mllm::Tensor::empty({new_weights_size}, mllm::kByte, mllm::kCPU).alloc();

        // Perform quantize
        kai_helper_.quant_pack_rhs_offline(new_weights.ptr<mllm::mllm_byte_t>(), weight.ptr<mllm::mllm_fp32_t>(),
                                           bias ? bias.ptr<mllm::mllm_fp32_t>() : nullptr, out_channels, in_channels, tile_cfg);

        // Assign new weights to the linear op
        new_weights.setName(weight.name());

        return mllm::ffi::Tensor(new_weights);
      });
}
#endif

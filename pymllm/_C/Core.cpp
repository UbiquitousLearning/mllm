// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include "pymllm/_C/Core.hpp"

#include "mllm/mllm.hpp"

void registerCoreBinding(py::module_& m) {
  py::enum_<mllm::OpTypes>(m, "OpTypes")
      .value("OpType_Start", mllm::OpTypes::kOpType_Start)
      .value("Fill", mllm::OpTypes::kFill)
      .value("Add", mllm::OpTypes::kAdd)
      .value("Sub", mllm::OpTypes::kSub)
      .value("Mul", mllm::OpTypes::kMul)
      .value("Div", mllm::OpTypes::kDiv)
      .value("MatMul", mllm::OpTypes::kMatMul)
      .value("Embedding", mllm::OpTypes::kEmbedding)
      .value("Linear", mllm::OpTypes::kLinear)
      .value("RoPE", mllm::OpTypes::kRoPE)
      .value("Softmax", mllm::OpTypes::kSoftmax)
      .value("Transpose", mllm::OpTypes::kTranspose)
      .value("RMSNorm", mllm::OpTypes::kRMSNorm)
      .value("SiLU", mllm::OpTypes::kSiLU)
      .value("KVCache", mllm::OpTypes::kKVCache)
      .value("CausalMask", mllm::OpTypes::kCausalMask)
      .value("CastType", mllm::OpTypes::kCastType)
      .value("X2X", mllm::OpTypes::kX2X)
      .value("Split", mllm::OpTypes::kSplit)
      .value("View", mllm::OpTypes::kView)
      .value("FlashAttention2", mllm::OpTypes::kFlashAttention2)
      .value("Repeat", mllm::OpTypes::kRepeat)
      .value("Permute", mllm::OpTypes::kPermute)
      .value("Conv3D", mllm::OpTypes::kConv3D)
      .value("Conv2D", mllm::OpTypes::kConv2D)
      .value("Conv1D", mllm::OpTypes::kConv1D)
      .value("GELU", mllm::OpTypes::kGELU)
      .value("LayerNorm", mllm::OpTypes::kLayerNorm)
      .value("MultimodalRoPE", mllm::OpTypes::kMultimodalRoPE)
      .value("VisionRoPE", mllm::OpTypes::kVisionRoPE)
      .value("QuickGELU", mllm::OpTypes::kQuickGELU)
      .value("Copy", mllm::OpTypes::kCopy)
      .value("Clone", mllm::OpTypes::kClone)
      .value("Neg", mllm::OpTypes::kNeg)
      .value("Concat", mllm::OpTypes::kConcat)
      .value("ReLU", mllm::OpTypes::kReLU)
      .value("ReLU2", mllm::OpTypes::kReLU2)
      .value("ReduceMax", mllm::OpTypes::kReduceMax)
      .value("ReduceMin", mllm::OpTypes::kReduceMin)
      .value("ReduceSum", mllm::OpTypes::kReduceSum)
      .value("Contiguous", mllm::OpTypes::kContiguous)
      .value("Reshape", mllm::OpTypes::kReshape)
      .value("GraphBegin", mllm::OpTypes::kGraphBegin)
      .value("GraphEnd", mllm::OpTypes::kGraphEnd)
      .value("OpType_End", mllm::OpTypes::kOpType_End);

  py::enum_<mllm::DeviceTypes>(m, "DeviceTypes")
      .value("CPU", mllm::DeviceTypes::kCPU)
      .value("CUDA", mllm::DeviceTypes::kCUDA)
      .value("OpenCL", mllm::DeviceTypes::kOpenCL);

  py::enum_<mllm::DataTypes>(m, "DataTypes")
      .value("Float32", mllm::DataTypes::kFloat32)
      .value("Float16", mllm::DataTypes::kFloat16)
      .value("GGUF_Q4_0", mllm::DataTypes::kGGUF_Q4_0)
      .value("GGUF_Q4_1", mllm::DataTypes::kGGUF_Q4_1)
      .value("GGUF_Q8_0", mllm::DataTypes::kGGUF_Q8_0)
      .value("GGUF_Q8_1", mllm::DataTypes::kGGUF_Q8_1)
      .value("GGUF_Q8_Pertensor", mllm::DataTypes::kGGUF_Q8_Pertensor)
      .value("GGUF_Q4_K", mllm::DataTypes::kGGUF_Q4_K)
      .value("GGUF_Q6_K", mllm::DataTypes::kGGUF_Q6_K)
      .value("GGUF_Q8_K", mllm::DataTypes::kGGUF_Q8_K)
      .value("Int8", mllm::DataTypes::kInt8)
      .value("Int16", mllm::DataTypes::kInt16)
      .value("Int32", mllm::DataTypes::kInt32)
      .value("GGUF_Q4_0_4_4", mllm::DataTypes::kGGUF_Q4_0_4_4)
      .value("GGUF_Q4_0_4_8", mllm::DataTypes::kGGUF_Q4_0_4_8)
      .value("GGUF_Q4_0_8_8", mllm::DataTypes::kGGUF_Q4_0_8_8)
      .value("GGUF_Q8_0_4_4", mllm::DataTypes::kGGUF_Q8_0_4_4)
      .value("GGUF_Q3_K", mllm::DataTypes::kGGUF_Q3_K)
      .value("GGUF_Q2_K", mllm::DataTypes::kGGUF_Q2_K)
      .value("GGUF_Q1_K", mllm::DataTypes::kGGUF_Q1_K)
      .value("GGUF_IQ2_XXS", mllm::DataTypes::kGGUF_IQ2_XXS)
      .value("GGUF_IQ2_XS", mllm::DataTypes::kGGUF_IQ2_XS)
      .value("GGUF_IQ1_S", mllm::DataTypes::kGGUF_IQ1_S)
      .value("GGUF_IQ1_M", mllm::DataTypes::kGGUF_IQ1_M)
      .value("GGUF_IQ2_S", mllm::DataTypes::kGGUF_IQ2_S)
      .value("BFloat16", mllm::DataTypes::kBFloat16)
      .value("UInt8", mllm::DataTypes::kUInt8)
      .value("UInt16", mllm::DataTypes::kUInt16)
      .value("UInt32", mllm::DataTypes::kUInt32)
      .value("Int64", mllm::DataTypes::kInt64)
      .value("UInt64", mllm::DataTypes::kUInt64)
      .value("Byte", mllm::DataTypes::kByte);

  py::enum_<mllm::TensorMemTypes>(m, "TensorMemTypes")
      .value("TensorMemTypes_Start", mllm::TensorMemTypes::kTensorMemTypes_Start)
      .value("Normal", mllm::TensorMemTypes::kNormal)
      .value("ExtraInput", mllm::TensorMemTypes::kExtraInput)
      .value("ExtraOutput", mllm::TensorMemTypes::kExtraOutput)
      .value("Manual", mllm::TensorMemTypes::kManual)
      .value("Global", mllm::TensorMemTypes::kGlobal)
      .value("Params_Start", mllm::TensorMemTypes::kParams_Start)
      .value("ParamsMMAP", mllm::TensorMemTypes::kParamsMMAP)
      .value("ParamsNormal", mllm::TensorMemTypes::kParamsNormal)
      .value("Params_End", mllm::TensorMemTypes::kParams_End)
      .value("QnnAppRead", mllm::TensorMemTypes::kQnnAppRead)
      .value("QnnAppWrite", mllm::TensorMemTypes::kQnnAppWrite)
      .value("QnnAppReadWrite", mllm::TensorMemTypes::kQnnAppReadWrite)
      .value("TensorMemTypes_End", mllm::TensorMemTypes::kTensorMemTypes_End);

  py::class_<mllm::Tensor>(m, "Tensor")
      .def(py::init<>())
      .def("__bool__", [](const mllm::Tensor& t) { return !t.isNil(); })
      .def("is_nil", &mllm::Tensor::isNil)
      .def_static("nil", &mllm::Tensor::nil)
      .def_static("empty", &mllm::Tensor::empty, py::arg("shape"), py::arg("dtype") = mllm::DataTypes::kFloat32,
                  py::arg("device") = mllm::DeviceTypes::kCPU)
      .def("alloc", &mllm::Tensor::alloc, py::return_value_policy::reference_internal)
      .def("alloc_extra_tensor_view", &mllm::Tensor::allocExtraTensorView, py::arg("extra_tensor_name"), py::arg("shape"),
           py::arg("dtype") = mllm::DataTypes::kFloat32, py::arg("device") = mllm::DeviceTypes::kCPU,
           py::return_value_policy::reference_internal)
      .def("get_extra_tensor_view_in_tensor", &mllm::Tensor::getExtraTensorViewInTensor)
      .def_static("zeros", &mllm::Tensor::zeros, py::arg("shape"), py::arg("dtype") = mllm::DataTypes::kFloat32,
                  py::arg("device") = mllm::DeviceTypes::kCPU)
      .def_static("ones", &mllm::Tensor::ones, py::arg("shape"), py::arg("dtype") = mllm::DataTypes::kFloat32,
                  py::arg("device") = mllm::DeviceTypes::kCPU)
      .def_static("arange", &mllm::Tensor::arange, py::arg("start"), py::arg("end"), py::arg("step"),
                  py::arg("dtype") = mllm::DataTypes::kFloat32, py::arg("device") = mllm::DeviceTypes::kCPU)
      .def_static("random", &mllm::Tensor::random, py::arg("shape"), py::arg("start") = -1.f, py::arg("end") = 1.f,
                  py::arg("dtype") = mllm::DataTypes::kFloat32, py::arg("device") = mllm::DeviceTypes::kCPU)
      .def("__add__", [](mllm::Tensor& self, mllm::Tensor& other) { return self + other; })
      .def("__sub__", [](mllm::Tensor& self, mllm::Tensor& other) { return self - other; })
      .def("__mul__", [](mllm::Tensor& self, mllm::Tensor& other) { return self * other; })
      .def("__truediv__", [](mllm::Tensor& self, mllm::Tensor& other) { return self / other; })
      .def("__add__", [](mllm::Tensor& self, float other) { return self + other; })
      .def("__sub__", [](mllm::Tensor& self, float other) { return self - other; })
      .def("__mul__", [](mllm::Tensor& self, float other) { return self * other; })
      .def("__truediv__", [](mllm::Tensor& self, float other) { return self / other; })
      .def("__neg__", [](mllm::Tensor& self) { return -self; })
      .def("min", &mllm::Tensor::min, py::arg("keep_dim") = false, py::arg("dim") = 0x7fffffff)
      .def("max", &mllm::Tensor::max, py::arg("keep_dim") = false, py::arg("dim") = 0x7fffffff)
      .def("sum", &mllm::Tensor::sum, py::arg("keep_dim") = false, py::arg("dim") = 0x7fffffff)
      .def("transpose", &mllm::Tensor::transpose)
      .def("T", &mllm::Tensor::T)
      .def("to", [](mllm::Tensor& self, mllm::DeviceTypes device) { return self.to(device); })
      .def("to", [](mllm::Tensor& self, mllm::DataTypes dtype) { return self.to(dtype); })
      .def("cpu", &mllm::Tensor::cpu)
      .def("cuda", &mllm::Tensor::cuda)
      .def("name", &mllm::Tensor::name)
      .def("mem_type", &mllm::Tensor::memType)
      .def("set_name", &mllm::Tensor::setName, py::return_value_policy::reference_internal)
      .def("set_mem_type", &mllm::Tensor::setMemType, py::return_value_policy::reference_internal)
      .def("dtype", &mllm::Tensor::dtype)
      .def("device", &mllm::Tensor::device)
      .def("shape", &mllm::Tensor::shape)
      .def("stride", &mllm::Tensor::stride)
      .def("numel", &mllm::Tensor::numel)
      .def("uuid", &mllm::Tensor::uuid)
      .def("is_contiguous", &mllm::Tensor::isContiguous)
      .def("contiguous", &mllm::Tensor::contiguous)
      .def("reshape", &mllm::Tensor::reshape)
      .def("view", &mllm::Tensor::view)
      .def("repeat", &mllm::Tensor::repeat)
      .def("unsqueeze", &mllm::Tensor::unsqueeze)
      .def("clone", &mllm::Tensor::clone)
      .def("permute", &mllm::Tensor::permute)
      .def("bytes", &mllm::Tensor::bytes);

  //===----------------------------------------------------------------------===//
  // Parameter File
  //===----------------------------------------------------------------------===//
  py::class_<mllm::ParameterFile, std::shared_ptr<mllm::ParameterFile>>(m, "ParameterFile");

  //===----------------------------------------------------------------------===//
  // BaseOp
  //===----------------------------------------------------------------------===//
  py::class_<mllm::BaseOp, std::shared_ptr<mllm::BaseOp>>(m, "BaseOp")
      .def("load", &mllm::BaseOp::load)
      .def("trace", &mllm::BaseOp::trace)
      .def("forward", &mllm::BaseOp::forward)
      .def("reshape", &mllm::BaseOp::reshape)
      .def("setup", &mllm::BaseOp::setup)
      .def("get_params", &mllm::BaseOp::getParams)
      .def("get_name", &mllm::BaseOp::getName)
      .def("set_name", &mllm::BaseOp::setName)
      .def("get_device", &mllm::BaseOp::getDevice)
      .def("set_device_type", &mllm::BaseOp::setDeviceType)
      .def("get_op_type", &mllm::BaseOp::getOpType);

  py::class_<mllm::BaseOpOptionsBase>(m, "BaseOpOptionsBase");

  //===----------------------------------------------------------------------===//
  // Linear Options And LinearOp
  //===----------------------------------------------------------------------===//
  py::enum_<mllm::aops::LinearImplTypes>(m, "LinearImplTypes")
      .value("LinearImplTypes_Start", mllm::aops::LinearImplTypes::kLinearImplTypes_Start)
      .value("Default", mllm::aops::LinearImplTypes::kDefault)
      .value("Kleidiai_Start", mllm::aops::LinearImplTypes::kKleidiai_Start)
      .value("Kleidiai_End", mllm::aops::LinearImplTypes::kKleidiai_End)
      .value("GGUF_Start", mllm::aops::LinearImplTypes::kGGUF_Start)
      .value("GGUF_End", mllm::aops::LinearImplTypes::kGGUF_End)
      .value("LinearImplTypes_End", mllm::aops::LinearImplTypes::kLinearImplTypes_End);

  py::class_<mllm::aops::LinearOpOptions>(m, "LinearOpOptions")
      .def(py::init<>())
      .def(py::init([](int32_t in_channels, int32_t out_channels, bool bias, mllm::aops::LinearImplTypes impl_type) {
             auto options = mllm::aops::LinearOpOptions{};
             options.in_channels = in_channels;
             options.out_channels = out_channels;
             options.bias = bias;
             options.impl_type = impl_type;
             return options;
           }),
           py::arg("in_channels") = 0, py::arg("out_channels") = 0, py::arg("bias") = true,
           py::arg("impl_type") = mllm::aops::LinearImplTypes::kDefault)
      .def_readwrite("in_channels", &mllm::aops::LinearOpOptions::in_channels)
      .def_readwrite("out_channels", &mllm::aops::LinearOpOptions::out_channels)
      .def_readwrite("bias", &mllm::aops::LinearOpOptions::bias)
      .def_readwrite("impl_type", &mllm::aops::LinearOpOptions::impl_type)
      .def("set_inputs_dtype", &mllm::aops::LinearOpOptions::setInputsDtype, py::return_value_policy::reference_internal)
      .def("set_outputs_dtype", &mllm::aops::LinearOpOptions::setOutputsDtype, py::return_value_policy::reference_internal);

  py::class_<mllm::aops::LinearOp, mllm::BaseOp, std::shared_ptr<mllm::aops::LinearOp>>(m, "LinearOp")
      .def(py::init<const mllm::aops::LinearOpOptions&>())
      .def("weight", &mllm::aops::LinearOp::weight, py::return_value_policy::reference_internal)
      .def("bias", &mllm::aops::LinearOp::bias, py::return_value_policy::reference_internal)
      .def("options", &mllm::aops::LinearOp::options, py::return_value_policy::reference_internal)
      .def("load", &mllm::aops::LinearOp::load)
      .def("forward", &mllm::aops::LinearOp::forward)
      .def("setup", &mllm::aops::LinearOp::setup)
      .def("reshape", &mllm::aops::LinearOp::reshape);
}

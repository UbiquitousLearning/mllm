// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include "pymllm/_C/Core.hpp"

#include "mllm/mllm.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/aops/GraphOps.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/core/aops/IndexOp.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/core/aops/PermuteOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/aops/ReshapeOp.hpp"
#include "mllm/core/aops/RepeatOp.hpp"
#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/core/aops/CloneOp.hpp"
#include "mllm/core/aops/ContiguousOp.hpp"
#include "mllm/core/aops/GraphOps.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/CopyOp.hpp"

void registerCoreBinding(py::module_& m) {
  pybind11::enum_<mllm::ModelFileVersion>(m, "ModelFileVersion")
      .value("V1", mllm::ModelFileVersion::kV1)
      .value("V2", mllm::ModelFileVersion::kV2)
      .value("UserTemporary", mllm::ModelFileVersion::kUserTemporary);

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
      .value("STFT", mllm::OpTypes::kSTFT)
      .value("ISTFT", mllm::OpTypes::kISTFT)
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
      .value("Slice", mllm::OpTypes::kSlice)
      .value("Param", mllm::OpTypes::kParam)
      .value("Index", mllm::OpTypes::kIndex)
      .value("Abs", mllm::OpTypes::kAbs)
      .value("Log", mllm::OpTypes::kLog)
      .value("TopK", mllm::OpTypes::kTopK)
      .value("Mean", mllm::OpTypes::kMean)
      .value("Clip", mllm::OpTypes::kClip)
      .value("Exp", mllm::OpTypes::kExp)
      .value("Sin", mllm::OpTypes::kSin)
      .value("Cos", mllm::OpTypes::kCos)
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
      .def("bytes", &mllm::Tensor::bytes)
      .def("__str__", [](const mllm::Tensor& t) {
        mllm::print(t);
        return "";
      });

  //===----------------------------------------------------------------------===//
  // Parameter File
  //===----------------------------------------------------------------------===//
  py::class_<mllm::ParameterFile, std::shared_ptr<mllm::ParameterFile>>(m, "ParameterFile")
      .def(py::init<mllm::ModelFileVersion>(), py::arg("v") = mllm::ModelFileVersion::kUserTemporary)
      .def("push", &mllm::ParameterFile::push)
      .def("pull", &mllm::ParameterFile::pull)
      .def("has", &mllm::ParameterFile::has)
      .def("remove", &mllm::ParameterFile::remove);

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

  // Bind Options classes
  py::class_<mllm::aops::RMSNormOpOptions>(m, "RMSNormOpOptions")
      .def(py::init<>())
      .def_readwrite("epsilon", &mllm::aops::RMSNormOpOptions::epsilon)
      .def_readwrite("add_unit_offset", &mllm::aops::RMSNormOpOptions::add_unit_offset);
  py::class_<mllm::aops::SiLUOpOptions>(m, "SiLUOpOptions").def(py::init<>());
  py::class_<mllm::aops::EmbeddingOpOptions>(m, "EmbeddingOpOptions")
      .def(py::init<>())
      .def(py::init([](int32_t vocab_size, int32_t hidden_size) {
             auto options = mllm::aops::EmbeddingOpOptions{};
             options.vocab_size = vocab_size;
             options.hidden_size = hidden_size;
             return options;
           }),
           py::arg("vocab_size") = 0, py::arg("hidden_size") = 0)
      .def_readwrite("vocab_size", &mllm::aops::EmbeddingOpOptions::vocab_size)
      .def_readwrite("hidden_size", &mllm::aops::EmbeddingOpOptions::hidden_size);

  py::class_<mllm::aops::GELUOpOptions>(m, "GELUOpOptions").def(py::init<>());
  py::class_<mllm::aops::LayerNormOpOptions>(m, "LayerNormOpOptions")
      .def(py::init<>())
      .def(py::init([](const std::vector<int>& normalized_shape, bool elementwise_affine, bool bias, float eps) {
             auto options = mllm::aops::LayerNormOpOptions{};
             options.normalized_shape = normalized_shape;
             options.elementwise_affine = elementwise_affine;
             options.bias = bias;
             options.eps = eps;
             return options;
           }),
           py::arg("normalized_shape") = std::vector<int>{}, py::arg("elementwise_affine") = true, py::arg("bias") = true,
           py::arg("eps") = 1e-5f)
      .def_readwrite("normalized_shape", &mllm::aops::LayerNormOpOptions::normalized_shape)
      .def_readwrite("elementwise_affine", &mllm::aops::LayerNormOpOptions::elementwise_affine)
      .def_readwrite("bias", &mllm::aops::LayerNormOpOptions::bias)
      .def_readwrite("eps", &mllm::aops::LayerNormOpOptions::eps);
  py::class_<mllm::aops::SoftmaxOpOptions>(m, "SoftmaxOpOptions")
      .def(py::init<>())
      .def(py::init([](int axis) {
             auto options = mllm::aops::SoftmaxOpOptions{};
             options.axis = axis;
             return options;
           }),
           py::arg("axis") = 0)
      .def_readwrite("axis", &mllm::aops::SoftmaxOpOptions::axis);
  py::class_<mllm::aops::CausalMaskOpOptions>(m, "CausalMaskOpOptions")
      .def(py::init<>())
      .def(py::init([](bool sliding_window, int window_size) {
             auto options = mllm::aops::CausalMaskOpOptions{};
             options.sliding_window = sliding_window;
             options.window_size = window_size;
             return options;
           }),
           py::arg("sliding_window") = false, py::arg("window_size") = 0)
      .def_readwrite("sliding_window", &mllm::aops::CausalMaskOpOptions::sliding_window)
      .def_readwrite("window_size", &mllm::aops::CausalMaskOpOptions::window_size);
  py::class_<mllm::aops::KVCacheOpOptions>(m, "KVCacheOpOptions")
      .def(py::init<>())
      .def(py::init([](int layer_idx, int q_head, int kv_head, int head_dim, bool use_fa2) {
             auto options = mllm::aops::KVCacheOpOptions{};
             options.layer_idx = layer_idx;
             options.q_head = q_head;
             options.kv_head = kv_head;
             options.head_dim = head_dim;
             options.use_fa2 = use_fa2;
             return options;
           }),
           py::arg("layer_idx") = 0, py::arg("q_head") = 0, py::arg("kv_head") = 0, py::arg("head_dim") = 0,
           py::arg("use_fa2") = false)
      .def_readwrite("layer_idx", &mllm::aops::KVCacheOpOptions::layer_idx)
      .def_readwrite("q_head", &mllm::aops::KVCacheOpOptions::q_head)
      .def_readwrite("kv_head", &mllm::aops::KVCacheOpOptions::kv_head)
      .def_readwrite("head_dim", &mllm::aops::KVCacheOpOptions::head_dim)
      .def_readwrite("use_fa2", &mllm::aops::KVCacheOpOptions::use_fa2);
  py::class_<mllm::aops::AddOpOptions>(m, "AddOpOptions").def(py::init<>());
  py::class_<mllm::aops::SubOpOptions>(m, "SubOpOptions").def(py::init<>());
  py::class_<mllm::aops::MulOpOptions>(m, "MulOpOptions").def(py::init<>());
  py::class_<mllm::aops::DivOpOptions>(m, "DivOpOptions").def(py::init<>());
  py::class_<mllm::aops::NegOpOptions>(m, "NegOpOptions").def(py::init<>());
  py::class_<mllm::aops::AbsOpOptions>(m, "AbsOpOptions").def(py::init<>());
  py::class_<mllm::aops::LogOpOptions>(m, "LogOpOptions").def(py::init<>());
  py::class_<mllm::aops::ExpOpOptions>(m, "ExpOpOptions").def(py::init<>());
  py::class_<mllm::aops::SinOpOptions>(m, "SinOpOptions").def(py::init<>());
  py::class_<mllm::aops::CosOpOptions>(m, "CosOpOptions").def(py::init<>());
  py::class_<mllm::aops::ConcatOpOptions>(m, "ConcatOpOptions")
      .def(py::init<>())
      .def(py::init([](int dim) {
             auto options = mllm::aops::ConcatOpOptions{};
             options.dim = dim;
             return options;
           }),
           py::arg("dim") = 0)
      .def_readwrite("dim", &mllm::aops::ConcatOpOptions::dim);
  py::class_<mllm::aops::Conv1DOpOptions>(m, "Conv1DOpOptions")
      .def(py::init<>())
      .def_readwrite("in_channels", &mllm::aops::Conv1DOpOptions::in_channels)
      .def_readwrite("out_channels", &mllm::aops::Conv1DOpOptions::out_channels)
      .def_readwrite("kernel_size", &mllm::aops::Conv1DOpOptions::kernel_size)
      .def_readwrite("stride", &mllm::aops::Conv1DOpOptions::stride)
      .def_readwrite("bias", &mllm::aops::Conv1DOpOptions::bias)
      .def_readwrite("padding", &mllm::aops::Conv1DOpOptions::padding)
      .def_readwrite("groups", &mllm::aops::Conv1DOpOptions::groups);
  py::class_<mllm::aops::Conv3DOpOptions>(m, "Conv3DOpOptions")
      .def(py::init<>())
      .def_readwrite("in_channels", &mllm::aops::Conv3DOpOptions::in_channels)
      .def_readwrite("out_channels", &mllm::aops::Conv3DOpOptions::out_channels)
      .def_readwrite("kernel_size", &mllm::aops::Conv3DOpOptions::kernel_size)
      .def_readwrite("stride", &mllm::aops::Conv3DOpOptions::stride)
      .def_readwrite("bias", &mllm::aops::Conv3DOpOptions::bias)
      .def_readwrite("impl_type", &mllm::aops::Conv3DOpOptions::impl_type);
  py::class_<mllm::aops::FillOpOptions>(m, "FillOpOptions")
      .def(py::init<>())
      .def_readwrite("type", &mllm::aops::FillOpOptions::type)
      .def_readwrite("value", &mllm::aops::FillOpOptions::value)
      .def_readwrite("start", &mllm::aops::FillOpOptions::start)
      .def_readwrite("end", &mllm::aops::FillOpOptions::end)
      .def_readwrite("step", &mllm::aops::FillOpOptions::step)
      .def_readwrite("seed", &mllm::aops::FillOpOptions::seed);
  py::class_<mllm::aops::FlashAttention2OpOptions>(m, "FlashAttention2OpOptions")
      .def(py::init<>())
      .def(py::init([](int B, int q_head, int kv_head, int D, int hp_exp, bool causal_mask) {
             auto options = mllm::aops::FlashAttention2OpOptions{};
             options.B = B;
             options.q_head = q_head;
             options.kv_head = kv_head;
             options.D = D;
             options.hp_exp = hp_exp;
             options.causal_mask = causal_mask;
             return options;
           }),
           py::arg("B") = 0, py::arg("q_head") = 0, py::arg("kv_head") = 0, py::arg("D") = 0, py::arg("hp_exp") = 0,
           py::arg("causal_mask") = false)
      .def_readwrite("B", &mllm::aops::FlashAttention2OpOptions::B)
      .def_readwrite("q_head", &mllm::aops::FlashAttention2OpOptions::q_head)
      .def_readwrite("kv_head", &mllm::aops::FlashAttention2OpOptions::kv_head)
      .def_readwrite("D", &mllm::aops::FlashAttention2OpOptions::D)
      .def_readwrite("hp_exp", &mllm::aops::FlashAttention2OpOptions::hp_exp)
      .def_readwrite("causal_mask", &mllm::aops::FlashAttention2OpOptions::causal_mask);
  py::class_<mllm::aops::IndexOpOptions>(m, "IndexOpOptions")
      .def(py::init<>())
      .def_readwrite("indices_", &mllm::aops::IndexOpOptions::indices_);
  py::class_<mllm::aops::MatMulOpOptions>(m, "MatMulOpOptions")
      .def(py::init<>())
      .def(py::init([](bool transpose_a, bool transpose_b, const std::string& matmul_type) {
             auto options = mllm::aops::MatMulOpOptions{};
             options.transpose_a = transpose_a;
             options.transpose_b = transpose_b;
             options.matmul_type = mllm::aops::str2MatMulOpType(matmul_type);
             return options;
           }),
           py::arg("transpose_a") = false, py::arg("transpose_b") = false, py::arg("matmul_type") = "Default")
      .def_readwrite("transpose_a", &mllm::aops::MatMulOpOptions::transpose_a)
      .def_readwrite("transpose_b", &mllm::aops::MatMulOpOptions::transpose_b)
      .def_readwrite("matmul_type", &mllm::aops::MatMulOpOptions::matmul_type);
  py::class_<mllm::aops::PermuteOpOptions>(m, "PermuteOpOptions")
      .def(py::init<>())
      .def(py::init([](const std::vector<int>& axis) {
             auto options = mllm::aops::PermuteOpOptions{};
             options.axis = axis;
             return options;
           }),
           py::arg("axis") = std::vector<int>{})
      .def_readwrite("axis", &mllm::aops::PermuteOpOptions::axis);
  py::class_<mllm::aops::ReduceMaxOpOptions>(m, "ReduceMaxOpOptions")
      .def(py::init<>())
      .def_readwrite("dim", &mllm::aops::ReduceMaxOpOptions::dim)
      .def_readwrite("keep_dim", &mllm::aops::ReduceMaxOpOptions::keep_dim);
  py::class_<mllm::aops::ReduceMinOpOptions>(m, "ReduceMinOpOptions")
      .def(py::init<>())
      .def_readwrite("dim", &mllm::aops::ReduceMinOpOptions::dim)
      .def_readwrite("keep_dim", &mllm::aops::ReduceMinOpOptions::keep_dim);
  py::class_<mllm::aops::ReduceSumOpOptions>(m, "ReduceSumOpOptions")
      .def(py::init<>())
      .def_readwrite("dim", &mllm::aops::ReduceSumOpOptions::dim)
      .def_readwrite("keep_dim", &mllm::aops::ReduceSumOpOptions::keep_dim);
  py::class_<mllm::aops::MeanOpOptions>(m, "MeanOpOptions")
      .def(py::init<>())
      .def_readwrite("dim", &mllm::aops::MeanOpOptions::dim)
      .def_readwrite("keep_dim", &mllm::aops::MeanOpOptions::keep_dim);
  py::class_<mllm::aops::ReshapeOpOptions>(m, "ReshapeOpOptions")
      .def(py::init<>())
      .def(py::init([](const std::vector<int>& shape) {
             auto options = mllm::aops::ReshapeOpOptions{};
             options.shape = shape;
             return options;
           }),
           py::arg("shape") = std::vector<int>{})
      .def_readwrite("shape", &mllm::aops::ReshapeOpOptions::shape);
  py::class_<mllm::aops::SliceOpOptions>(m, "SliceOpOptions").def(py::init<>());
  py::class_<mllm::aops::SplitOpOptions>(m, "SplitOpOptions")
      .def(py::init<>())
      .def(py::init([](int32_t dim, const std::vector<int32_t>& split_size_or_sections) {
             auto options = mllm::aops::SplitOpOptions{};
             options.dim = dim;
             options.split_size_or_sections = split_size_or_sections;
             return options;
           }),
           py::arg("dim"), py::arg("split_size_or_sections"))
      .def_readwrite("dim", &mllm::aops::SplitOpOptions::dim)
      .def_readwrite("split_size_or_sections", &mllm::aops::SplitOpOptions::split_size_or_sections);
  py::class_<mllm::aops::STFTOpOptions>(m, "STFTOpOptions")
      .def(py::init<>())
      .def(py::init([](int n_fft, int hop_length, int win_length, bool onesided, bool center, const std::string& pad_mode,
                       bool return_complex) {
             auto options = mllm::aops::STFTOpOptions{};
             options.n_fft = n_fft;
             options.hop_length = hop_length;
             options.win_length = win_length;
             options.onesided = onesided;
             options.center = center;
             options.pad_mode = pad_mode;
             options.return_complex = return_complex;
             return options;
           }),
           py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"), py::arg("onesided"), py::arg("center"),
           py::arg("pad_mode") = "constant", py::arg("return_complex") = false)
      .def_readwrite("n_fft", &mllm::aops::STFTOpOptions::n_fft)
      .def_readwrite("hop_length", &mllm::aops::STFTOpOptions::hop_length)
      .def_readwrite("win_length", &mllm::aops::STFTOpOptions::win_length)
      .def_readwrite("onesided", &mllm::aops::STFTOpOptions::onesided)
      .def_readwrite("center", &mllm::aops::STFTOpOptions::center)
      .def_readwrite("pad_mode", &mllm::aops::STFTOpOptions::pad_mode)
      .def_readwrite("return_complex", &mllm::aops::STFTOpOptions::return_complex);
  py::class_<mllm::aops::ISTFTOpOptions>(m, "ISTFTOpOptions")
      .def(py::init<>())
      .def_readwrite("n_fft", &mllm::aops::ISTFTOpOptions::n_fft)
      .def_readwrite("hop_length", &mllm::aops::ISTFTOpOptions::hop_length)
      .def_readwrite("win_length", &mllm::aops::ISTFTOpOptions::win_length)
      .def_readwrite("onesided", &mllm::aops::ISTFTOpOptions::onesided)
      .def_readwrite("center", &mllm::aops::ISTFTOpOptions::center)
      .def_readwrite("pad_mode", &mllm::aops::ISTFTOpOptions::pad_mode)
      .def_readwrite("normalized", &mllm::aops::ISTFTOpOptions::normalized)
      .def_readwrite("length", &mllm::aops::ISTFTOpOptions::length);
  py::class_<mllm::aops::TopKOpOptions>(m, "TopKOpOptions")
      .def(py::init<>())
      .def_readwrite("k", &mllm::aops::TopKOpOptions::k)
      .def_readwrite("dim", &mllm::aops::TopKOpOptions::dim)
      .def_readwrite("largest", &mllm::aops::TopKOpOptions::largest)
      .def_readwrite("sorted", &mllm::aops::TopKOpOptions::sorted);
  py::class_<mllm::aops::CloneOpOptions>(m, "CloneOpOptions").def(py::init<>());
  py::class_<mllm::aops::ContiguousOpOptions>(m, "ContiguousOpOptions").def(py::init<>());
  py::class_<mllm::aops::CopyOpOptions>(m, "CopyOpOptions").def(py::init<>());
  py::class_<mllm::aops::QuickGELUOpOptions>(m, "QuickGELUOpOptions").def(py::init<>());
  py::class_<mllm::aops::ReLUOpOptions>(m, "ReLUOpOptions").def(py::init<>());
  py::class_<mllm::aops::RepeatOpOptions>(m, "RepeatOpOptions").def(py::init<>());
  py::class_<mllm::aops::TransposeOpOptions>(m, "TransposeOpOptions").def(py::init<>());
  py::class_<mllm::aops::ViewOpOptions>(m, "ViewOpOptions")
      .def(py::init<>())
      .def_readwrite("to_shape", &mllm::aops::ViewOpOptions::to_shape);
  py::class_<mllm::aops::X2XOpOptions>(m, "X2XOpOptions").def(py::init<>());
  py::class_<mllm::aops::CastTypeOpOptions>(m, "CastTypeOpOptions").def(py::init<>());
  py::class_<mllm::aops::ParamOpOptions>(m, "ParamOpOptions").def(py::init<>());
  py::class_<mllm::aops::GraphBeginOpOptions>(m, "GraphBeginOpOptions").def(py::init<>());
  py::class_<mllm::aops::GraphEndOpOptions>(m, "GraphEndOpOptions").def(py::init<>());
  py::class_<mllm::aops::Qwen2VLRoPEOpOptions>(m, "Qwen2VLRoPEOpOptions")
      .def(py::init<>())
      .def_readwrite("dims", &mllm::aops::Qwen2VLRoPEOpOptions::dims)
      .def_readwrite("spatial_merge_size", &mllm::aops::Qwen2VLRoPEOpOptions::spatial_merge_size)
      .def_readwrite("theta", &mllm::aops::Qwen2VLRoPEOpOptions::theta);
  py::class_<mllm::aops::VisionRoPEOpOptions>(m, "VisionRoPEOpOptions").def(py::init<>());
  py::class_<mllm::aops::MultimodalRoPEOpOptions>(m, "MultimodalRoPEOpOptions").def(py::init<>());
  py::class_<mllm::aops::ClipOpOptions>(m, "ClipOpOptions")
      .def(py::init<>())
      .def_readwrite("min_val", &mllm::aops::ClipOpOptions::min_val)
      .def_readwrite("max_val", &mllm::aops::ClipOpOptions::max_val);

  auto base_opt_base = py::class_<mllm::BaseOpOptionsBase>(m, "BaseOpOptionsBase");
  // FIXME others.

  implicit_convertible_with_concept<mllm::aops::LinearOpOptions>(base_opt_base);
  implicit_convertible_with_concept<mllm::aops::ClipOpOptions>(base_opt_base);
}

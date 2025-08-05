// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "pymllm/_C/Core.hpp"
#include "mllm/mllm.hpp"

void registerCoreBinding(py::module_& m) {
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
}

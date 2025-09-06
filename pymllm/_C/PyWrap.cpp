// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>

#include "mllm/mllm.hpp"

#include "pymllm/_C/Core.hpp"
#include "pymllm/_C/Engine.hpp"
#include "pymllm/_C/Nn.hpp"

PYBIND11_MODULE(_C, m) {
  //===----------------------------------------------------------------------===//
  // Binding things in mllm/core
  //===----------------------------------------------------------------------===//
  registerCoreBinding(m);

  //===----------------------------------------------------------------------===//
  // Binding things in mllm/engine
  //===----------------------------------------------------------------------===//
  registerEngineBinding(m);

  //===----------------------------------------------------------------------===//
  // Binding things in mllm/nn
  //===----------------------------------------------------------------------===//
  registerNnBinding(m);

  //===----------------------------------------------------------------------===//
  // Binding things in mllm/mllm.hpp
  //===----------------------------------------------------------------------===//
  m.def("initialize_context", &mllm::initializeContext, "Initialize the MLLM context");
  m.def("shutdown_context", &mllm::shutdownContext, "Shutdown the MLLM context");
  m.def("set_random_seed", &mllm::setRandomSeed, "Set random seed", pybind11::arg("seed"));
  m.def("set_maximum_num_threads", &mllm::setMaximumNumThreads, "Set maximum number of threads", pybind11::arg("num_threads"));
  m.def("memory_report", &mllm::memoryReport, "Print memory report");
  m.def("is_opencl_available", &mllm::isOpenCLAvailable, "Check if OpenCL is available");
  m.def("is_qnn_available", &mllm::isQnnAvailable, "Check if QNN is available");
  m.def("clean_this_thread", &mllm::cleanThisThread, "Clean current thread context");
  m.def("this_thread", &mllm::thisThread, "Get current thread context");
  m.def("load", &mllm::load, "Load parameter file", pybind11::arg("file_name"),
        pybind11::arg("version") = mllm::ModelFileVersion::kV1, pybind11::arg("map_2_device") = mllm::kCPU);
  m.def("save", &mllm::save, "Save parameter file", pybind11::arg("file_name"), pybind11::arg("parameter_file"),
        pybind11::arg("version") = mllm::ModelFileVersion::kV1, pybind11::arg("map_2_device") = mllm::kCPU);

  // Expose _GLIBCXX_USE_CXX11_ABI
#ifdef _GLIBCXX_USE_CXX11_ABI
  m.attr("_GLIBCXX_USE_CXX11_ABI") = pybind11::int_(_GLIBCXX_USE_CXX11_ABI);
#else
  m.attr("_GLIBCXX_USE_CXX11_ABI") = pybind11::none();
#endif
}

// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/nn/Layer.hpp"
#include "mllm/nn/Module.hpp"

#include "pymllm/_C/Nn.hpp"

#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void registerNnBinding(pybind11::module_& m) {
  using namespace mllm::nn;  // NOLINT
  using namespace mllm;      // NOLINT

  // Bind AbstractNnNode class
  py::class_<AbstractNnNode, AbstractNnNode::ptr_t>(m, "AbstractNnNode")
      .def("regChildNode", &AbstractNnNode::regChildNode)
      .def("refParentNode", &AbstractNnNode::refParentNode, py::return_value_policy::reference_internal)
      .def("refChildNodes", &AbstractNnNode::refChildNodes, py::return_value_policy::reference_internal)
      .def("setName", &AbstractNnNode::setName)
      .def("setAbsoluteName", &AbstractNnNode::setAbsoluteName)
      .def("setDepth", &AbstractNnNode::setDepth)
      .def("depthIncrease", &AbstractNnNode::depthIncrease)
      .def("depthDecrease", &AbstractNnNode::depthDecrease)
      .def("getName", &AbstractNnNode::getName)
      .def("getAbsoluteName", &AbstractNnNode::getAbsoluteName)
      .def("getDepth", &AbstractNnNode::getDepth)
      .def("getType", &AbstractNnNode::getType)
      .def("getDevice", &AbstractNnNode::getDevice)
      .def("setCompiledAsObj", &AbstractNnNode::setCompiledAsObj)
      .def("isCompiledAsObj", &AbstractNnNode::isCompiledAsObj);

  // Bind LayerImpl class
  py::class_<LayerImpl, LayerImpl::ptr_t, AbstractNnNode>(m, "LayerImpl")
      .def("load", &LayerImpl::load)
      .def("to", &LayerImpl::to)
      .def("opType", &LayerImpl::opType)
      .def("refOptions", &LayerImpl::refOptions, py::return_value_policy::reference_internal)
      .def("getInstancedOp", &LayerImpl::getInstancedOp)
      .def("setInstancedOp", &LayerImpl::setInstancedOp);

  // Bind ModuleImpl class
  py::class_<ModuleImpl, ModuleImpl::ptr_t, AbstractNnNode>(m, "ModuleImpl")
      .def(py::init<>())
      .def("load", &ModuleImpl::load)
      .def("params", &ModuleImpl::params)
      .def("to", &ModuleImpl::to);
}
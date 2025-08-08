// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/nn/Layer.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"

#include "pymllm/_C/Nn.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void registerNnBinding(pybind11::module_& m) {
  using namespace mllm::nn;  // NOLINT
  using namespace mllm;      // NOLINT

  // Bind AbstractNnNode class
  py::class_<AbstractNnNode, AbstractNnNode::ptr_t>(m, "AbstractNnNode")
      .def("reg_child_node", &AbstractNnNode::regChildNode)
      .def("ref_parent_node", &AbstractNnNode::refParentNode, py::return_value_policy::reference_internal)
      .def("ref_child_nodes", &AbstractNnNode::refChildNodes, py::return_value_policy::reference_internal)
      .def("set_name", &AbstractNnNode::setName)
      .def("set_absolute_name", &AbstractNnNode::setAbsoluteName)
      .def("set_depth", &AbstractNnNode::setDepth)
      .def("depth_increase", &AbstractNnNode::depthIncrease)
      .def("depth_decrease", &AbstractNnNode::depthDecrease)
      .def("get_name", &AbstractNnNode::getName)
      .def("get_absolute_name", &AbstractNnNode::getAbsoluteName)
      .def("get_depth", &AbstractNnNode::getDepth)
      .def("get_type", &AbstractNnNode::getType)
      .def("get_device", &AbstractNnNode::getDevice)
      .def("set_compiled_as_obj", &AbstractNnNode::setCompiledAsObj)
      .def("is_compiled_as_obj", &AbstractNnNode::isCompiledAsObj);

  // Bind LayerImpl class
  py::class_<LayerImpl, LayerImpl::ptr_t, AbstractNnNode>(m, "LayerImpl")
      .def(py::init([](OpTypes op_type, const mllm::aops::LinearOpOptions& options) {
             return std::make_shared<LayerImpl>(op_type, options);
           }),
           py::arg("op_type"), py::arg("options"))
      .def("load", &LayerImpl::load)
      .def("to", &LayerImpl::to)
      .def("op_type", &LayerImpl::opType)
      .def("ref_options", &LayerImpl::refOptions, py::return_value_policy::reference_internal)
      .def("get_instanced_op", &LayerImpl::getInstancedOp)
      .def("set_instanced_op", &LayerImpl::setInstancedOp);

  // Bind ModuleImpl class
  py::class_<ModuleImpl, ModuleImpl::ptr_t, AbstractNnNode>(m, "ModuleImpl")
      .def(py::init<>())
      .def("load", &ModuleImpl::load)
      .def("params", &ModuleImpl::params)
      .def("to", &ModuleImpl::to);

  py::class_<Layer>(m, "CXXLayer").def(py::init<const LayerImpl::ptr_t&>(), py::arg("impl")).def("__main", &Layer::__main);

  py::class_<Module>(m, "CXXModule")
      .def(py::init<const ModuleImpl::ptr_t&>(), py::arg("impl"))
      .def("__send_graph_begin", &Module::__send_graph_begin)
      .def("__send_graph_end", &Module::__send_graph_end)
      .def("__trace", &Module::__trace);
}
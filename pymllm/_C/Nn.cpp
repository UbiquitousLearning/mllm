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

  py::class_<Layer>(m, "CXXLayer").def(py::init<const LayerImpl::ptr_t&>(), py::arg("impl")).def("forward", &Layer::__main);

  py::class_<Module>(m, "CXXModule")
      .def(py::init<const ModuleImpl::ptr_t&>(), py::arg("impl"))
      .def("__send_graph_begin", &Module::__send_graph_begin)
      .def("__send_graph_end", &Module::__send_graph_end)
      .def("__trace", &Module::__trace);

  // Bind Options classes
  py::class_<mllm::aops::RMSNormOpOptions>(m, "RMSNormOpOptions")
      .def(py::init<>())
      .def_readwrite("epsilon", &mllm::aops::RMSNormOpOptions::epsilon)
      .def_readwrite("add_unit_offset", &mllm::aops::RMSNormOpOptions::add_unit_offset);

  py::class_<mllm::aops::SiLUOpOptions>(m, "SiLUOpOptions").def(py::init<>());

  py::class_<mllm::aops::EmbeddingOpOptions>(m, "EmbeddingOpOptions")
      .def(py::init<>())
      .def_readwrite("vocab_size", &mllm::aops::EmbeddingOpOptions::vocab_size)
      .def_readwrite("hidden_size", &mllm::aops::EmbeddingOpOptions::hidden_size);

  py::class_<mllm::aops::GELUOpOptions>(m, "GELUOpOptions").def(py::init<>());

  py::class_<mllm::aops::LayerNormOpOptions>(m, "LayerNormOpOptions")
      .def(py::init<>())
      .def_readwrite("normalized_shape", &mllm::aops::LayerNormOpOptions::normalized_shape)
      .def_readwrite("elementwise_affine", &mllm::aops::LayerNormOpOptions::elementwise_affine)
      .def_readwrite("bias", &mllm::aops::LayerNormOpOptions::bias)
      .def_readwrite("eps", &mllm::aops::LayerNormOpOptions::eps);

  py::class_<mllm::aops::SoftmaxOpOptions>(m, "SoftmaxOpOptions")
      .def(py::init<>())
      .def_readwrite("axis", &mllm::aops::SoftmaxOpOptions::axis);

  py::class_<mllm::aops::CausalMaskOpOptions>(m, "CausalMaskOpOptions")
      .def(py::init<>())
      .def_readwrite("sliding_window", &mllm::aops::CausalMaskOpOptions::sliding_window)
      .def_readwrite("window_size", &mllm::aops::CausalMaskOpOptions::window_size);

  py::class_<mllm::aops::KVCacheOpOptions>(m, "KVCacheOpOptions")
      .def(py::init<>())
      .def_readwrite("layer_idx", &mllm::aops::KVCacheOpOptions::layer_idx)
      .def_readwrite("q_head", &mllm::aops::KVCacheOpOptions::q_head)
      .def_readwrite("kv_head", &mllm::aops::KVCacheOpOptions::kv_head)
      .def_readwrite("head_dim", &mllm::aops::KVCacheOpOptions::head_dim)
      .def_readwrite("use_fa2", &mllm::aops::KVCacheOpOptions::use_fa2);
}
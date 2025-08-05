// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "pymllm/_C/Engine.hpp"

void registerEngineBinding(py::module_& m) {
  pybind11::enum_<mllm::ModelFileVersion>(m, "ModelFileVersion")
      .value("V1", mllm::ModelFileVersion::kV1)
      .value("V2", mllm::ModelFileVersion::kV2)
      .value("UserTemporary", mllm::ModelFileVersion::kUserTemporary);

  pybind11::class_<mllm::PerfFile, mllm::PerfFile::ptr_t>(m, "PerfFile")
      .def("finalize", &mllm::PerfFile::finalize)
      .def("save", &mllm::PerfFile::save);

  pybind11::class_<mllm::SessionTCB, mllm::SessionTCB::ptr_t>(m, "SessionTCB")
      .def_readwrite("trace_mode", &mllm::SessionTCB::trace_mode);

  pybind11::enum_<mllm::TaskTypes>(m, "TaskTypes")
      .value("ExecuteOp", mllm::TaskTypes::kExecuteOp)
      .value("ExecuteModule", mllm::TaskTypes::kExecuteModule);

  pybind11::class_<mllm::Task, mllm::Task::ptr_t>(m, "Task")
      .def_readwrite("type", &mllm::Task::type)
      .def_readwrite("inputs", &mllm::Task::inputs)
      .def_readwrite("outputs", &mllm::Task::outputs)
      .def_readwrite("custom_context_ptr", &mllm::Task::custom_context_ptr);

  pybind11::class_<mllm::MemoryManagerOptions>(m, "MemoryManagerOptions")
      .def(py::init<>())
      .def_readwrite("really_large_tensor_threshold", &mllm::MemoryManagerOptions::really_large_tensor_threshold)
      .def_readwrite("using_buddy_mem_pool", &mllm::MemoryManagerOptions::using_buddy_mem_pool);

  pybind11::class_<mllm::MemoryManager, mllm::MemoryManager::ptr_t>(m, "MemoryManager")
      .def("clearAll", &mllm::MemoryManager::clearAll)
      .def("report", &mllm::MemoryManager::report);

  pybind11::class_<mllm::DispatcherManagerOptions>(m, "DispatcherManagerOptions")
      .def(py::init<>())
      .def_readwrite("numa_policy", &mllm::DispatcherManagerOptions::numa_policy)
      .def_readwrite("num_threads", &mllm::DispatcherManagerOptions::num_threads);

  pybind11::class_<mllm::Dispatcher, mllm::Dispatcher::ptr_t>(m, "Dispatcher")
      .def("id", &mllm::Dispatcher::id)
      .def("process", &mllm::Dispatcher::process)
      .def("syncWait", &mllm::Dispatcher::syncWait);

  pybind11::class_<mllm::DispatcherManager, mllm::DispatcherManager::ptr_t>(m, "DispatcherManager")
      .def("syncWait", &mllm::DispatcherManager::syncWait)
      .def("submit", &mllm::DispatcherManager::submit);

  pybind11::class_<mllm::Context>(m, "Context")
      .def_static("instance", &mllm::Context::instance, py::return_value_policy::reference)
      .def("memoryManager", &mllm::Context::memoryManager)
      .def("dispatcherManager", &mllm::Context::dispatcherManager)
      .def("getUUID", &mllm::Context::getUUID)
      .def("thisThread", &mllm::Context::thisThread)
      .def("mainThread", &mllm::Context::mainThread)
      .def("setRandomSeed", &mllm::Context::setRandomSeed)
      .def("getRandomSeed", &mllm::Context::getRandomSeed)
      .def("setPerfMode", &mllm::Context::setPerfMode)
      .def("isPerfMode", &mllm::Context::isPerfMode)
      .def("getPerfFile", &mllm::Context::getPerfFile)
      .def("refSessionThreads", &mllm::Context::refSessionThreads);

  pybind11::class_<mllm::ConfigFile>(m, "ConfigFile")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("loadString", &mllm::ConfigFile::loadString)
      .def("load", &mllm::ConfigFile::load)
      .def("dump", &mllm::ConfigFile::dump)
      .def("save", &mllm::ConfigFile::save)
      .def(
          "data",
          [](mllm::ConfigFile& self) {
            nlohmann::json& json = self.data();
            return py::module_::import("json").attr("loads")(json.dump());
          },
          py::return_value_policy::copy);
}
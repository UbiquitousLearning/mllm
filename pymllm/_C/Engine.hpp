// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void registerEngineBinding(py::module_& m);

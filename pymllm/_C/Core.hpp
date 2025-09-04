// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void registerCoreBinding(py::module_& m);

template<typename Derived, typename Base>
void implicit_convertible_with_concept(py::class_<Base>& cl) {
  cl.def(py::init<Derived const&>());
  py::implicitly_convertible<Derived, Base>();
}

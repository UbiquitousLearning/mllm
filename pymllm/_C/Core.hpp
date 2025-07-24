/**
 * @file Core.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

void registerCoreBinding(py::module_& m);

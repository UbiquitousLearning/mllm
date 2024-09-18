/**
 * @file Core.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;

void registerCore(py::module_ &m);
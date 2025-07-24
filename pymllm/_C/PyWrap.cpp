/**
 * @file PyWarp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include <pybind11/pybind11.h>

#include "pymllm/_C/Core.hpp"
#include "pymllm/_C/Engine.hpp"
#include "pymllm/_C/Nn.hpp"

PYBIND11_MODULE(_C, m) { registerCoreBinding(m); }
/**
 * @file Core.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "Core.hpp"

void registerCore(py::module_ &m) { auto core_m = m.def_submodule("core"); }

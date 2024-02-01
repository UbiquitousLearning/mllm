//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#include "Module.hpp"

namespace mllm {

map<BackendType, Backend*> Module::backends;
ParamLoader *Module::loader;

} // namespace mllm
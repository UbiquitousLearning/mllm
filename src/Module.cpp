//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#include "Module.hpp"

namespace mllm {

map<BackendType, Backend*> Module::backends;
ParamLoader *Module::loader;
int Module::listIdx;
int Module::runlistIdx;
TensorStatus Module::tensor_status;
bool Module::doLoad = false;
} // namespace mllm
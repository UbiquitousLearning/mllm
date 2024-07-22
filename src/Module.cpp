//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#include "Module.hpp"
#include "Types.hpp"

namespace mllm {

map<BackendType, Backend*> Module::backends;
AbstructLoader *Module::loader;
int Module::listIdx;
int Module::runlistIdx;
// TensorStatus Module::tensor_status;
bool Module::doLoad = false;
bool Module::doToDevice = false;
BackendType Module::tmp_device = MLLM_CPU;
std::unordered_map<string, shared_ptr<Op>> Module::tensor_func_ops;
} // namespace mllm
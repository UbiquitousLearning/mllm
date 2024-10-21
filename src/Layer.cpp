//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#include "Layer.hpp"
namespace mllm {
map<string, string> Layer::layername_2_tensorname;
bool Layer::use_layername_2_tensorname = true;
}; // namespace mllm
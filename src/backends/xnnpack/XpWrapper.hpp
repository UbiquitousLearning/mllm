/**
 * @file XpWrapper.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Layer.hpp"
#include "Module.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <any>

namespace mllm::xnnpack {

class XpWrapperModule : public Module {
public:
    XpWrapperModule() = default;

    XpWrapperModule(int input_num, int output_num);

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override;

    void setWrappedModule(const std::shared_ptr<Module> &wrapped_module);

private:
    int intput_nums_ = 0;
    int output_nums_ = 0;

    std::shared_ptr<Module> wrapped_module_ = nullptr;
    std::vector<Layer> direct_input_layers_;
    std::vector<Layer> direct_output_layers_;
    std::vector<Layer> d2h_inputs_layers_;
    std::vector<Layer> d2h_outputs_layers_;
    Layer dispatch_all_;
    Layer subgraph_start_;
    Layer subgraph_finalize_;
};

template <typename T, typename... Args>
XpWrapperModule wrap2xnn(int inputs_nums, int output_nums, Args &&...args) {
    XpWrapperModule ret(inputs_nums, output_nums);

    auto based_module = std::make_shared<T>(std::forward<Args>(args)...);
    ret.setWrappedModule(based_module);

    based_module->to(MLLM_XNNPACK);
    ret.to(MLLM_XNNPACK);

    return ret;
}

} // namespace mllm::xnnpack
  // namespace mllm::xnnpack
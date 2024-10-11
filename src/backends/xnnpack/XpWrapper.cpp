#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"

namespace mllm::xnnpack {

XpWrapperModule::XpWrapperModule(int input_num, int output_num) :
    intput_nums_(input_num), output_nums_(output_num) {
    if (input_num + output_num > 16) {
        Log::error("input_num + output_num > 16, pls recompile with larger external values num with createSubgraph(k); where k > 16");
        exit(-1);
    }

    for (int i = 0; i < input_num; ++i) {
        direct_input_layers_.emplace_back(Direct(Direct::ExternalInput, "directinput_" + std::to_string(i)));
    }

    for (int o = 0; o < output_num; ++o) {
        direct_output_layers_.emplace_back(Direct(Direct::ExternalOutput, "directoutput_" + std::to_string(o)));
    }

    dispatch_all_ = Dispatch("dispatch_xnnpack_graph");
}

std::vector<Tensor> XpWrapperModule::Forward(std::vector<Tensor> inputs, std::vector<std::any> args) {
    std::vector<Tensor> registered_inputs;

    if (intput_nums_ != inputs.size()) {
        Log::error("In XpWrapperModule intput_nums_ != inputs.size(). Got input_nums_={}, but inputs.size()={}", intput_nums_, inputs.size());
        exit(-1);
    }

    for (int i = 0; i < intput_nums_; ++i) {
        auto x = direct_input_layers_[i](inputs[i]);
        registered_inputs.push_back(x);
    }

    auto outs = (*wrapped_module_)(registered_inputs);

    if (output_nums_ != outs.size()) {
        Log::error("In XpWrapperModule output_nums_ != outs.size(). Got output_nums_={}, but outs.size()={}", output_nums_, outs.size());
        exit(-1);
    }

    std::vector<Tensor> registered_outputs;

    for (int o = 0; o < output_nums_; ++o) {
        auto x = direct_output_layers_[o](outs[o]);
        registered_outputs.push_back(x);
    }

    // dispatch have no inputs and outputs. But we gave it a value to launch it.
    auto _ = dispatch_all_(registered_outputs[0]);

    for (auto &o : registered_outputs) {
        o.xnn();
    }

    return registered_outputs;
}

void XpWrapperModule::setWrappedModule(const std::shared_ptr<Module> &wrapped_module) {
    wrapped_module_ = wrapped_module;
}

} // namespace mllm::xnnpack
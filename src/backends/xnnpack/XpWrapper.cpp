#include "backends/xnnpack/XpWrapper.hpp"
#include "Layer.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "xnnpack.h"
#include <cassert>

namespace mllm::xnnpack {

XpWrapperModule::XpWrapperModule(int input_num, int output_num) :
    intput_nums_(input_num), output_nums_(output_num) {
    if (input_num + output_num > 16) {
        Log::error("input_num + output_num > 16, pls recompile with larger external values num with createSubgraph(k); where k > 16");
        exit(-1);
    }

    for (int i = 0; i < input_num; ++i) {
        direct_input_layers_.emplace_back(Direct(Direct::ExternalInput, "directinput_" + std::to_string(i)));
        d2h_inputs_layers_.emplace_back(Device2Host("d2h_i_" + std::to_string(i)));
    }

    for (int o = 0; o < output_num; ++o) {
        direct_output_layers_.emplace_back(Direct(Direct::ExternalOutput, "directoutput_" + std::to_string(o)));
        d2h_outputs_layers_.emplace_back(Device2Host("d2h_o_" + std::to_string(o)));
    }

    dispatch_all_ = Dispatch("dispatch_xnnpack_graph");
    subgraph_start_ = SubgraphStart("xnn_subgraph_start");
    subgraph_finalize_ = SubgraphFinalize("xnn_subgraph_finalize");
}

std::vector<Tensor> XpWrapperModule::Forward(std::vector<Tensor> inputs, std::vector<std::any> args) {
    std::vector<Tensor> registered_inputs;

    if (intput_nums_ != inputs.size()) {
        Log::error("In XpWrapperModule intput_nums_ != inputs.size(). Got input_nums_={}, but inputs.size()={}", intput_nums_, inputs.size());
        exit(-1);
    }

    auto _ = subgraph_start_(inputs[0]);

    for (int i = 0; i < intput_nums_; ++i) {
        inputs[i].uuid() = XNN_INVALID_VALUE_ID;
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
    _ = dispatch_all_(registered_outputs[0]);

    for (int o = 0; o < output_nums_; ++o) {
        registered_outputs[o] = d2h_outputs_layers_[o](registered_outputs[o]);
    }

    _ = subgraph_finalize_(registered_outputs[0]);

    return registered_outputs;
}

void XpWrapperModule::setWrappedModule(const std::shared_ptr<Module> &wrapped_module) {
    wrapped_module_ = wrapped_module;
}

} // namespace mllm::xnnpack
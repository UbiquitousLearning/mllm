#include <iostream>
#include "Types.hpp"
#include "cmdline.h"
#include "Context.hpp"

// tiny llama
#include "models/tinyllama/modeling_tinyllama.hpp"
#include "models/tinyllama/configuration_tinyllama.hpp"

// gemma
#include "models/gemma/modeling_gemma.hpp"
#include "models/gemma/configuration_gemma.hpp"

// qwen
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/configuration_qwen.hpp"

// stable llm
#include "models/stablelm/configuration_stablelm.hpp"
#include "models/stablelm/modeling_stablelm.hpp"

// opt
#include "models/opt/configuration_opt.hpp"
#include "models/opt/modeling_opt.hpp"

// mini cpm
#include "models/minicpm/configuration_minicpm.hpp"
#include "models/minicpm/modeling_minicpm.hpp"

// smollm
#include "models/smollm/configuration_smollm.hpp"
#include "models/smollm/modeling_smollm.hpp"

// qwen2.5
// #include "models/qwen2_5/configuration_qwen2_5.hpp"
// #include "models/qwen2_5/modeling_qwen2_5.hpp"

#include "processor/PostProcess.hpp"

using namespace mllm;

Tensor tokens2Input(int tokens_size, string name = "input", BackendType type = MLLM_CPU) {
    Tensor tensor1(1, 1, tokens_size, 1, Backend::global_backends[type].get(), true);
    tensor1.setName(name);
    Tensor::tensor_status = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    for (int idx = 0; idx < tokens_size; ++idx) {
        tensor1.setDataAt<float>(0, 0, idx, 0, 0);
    }
    return tensor1;
}

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<std::string>("model_name", 'n', "the name of model", false);
    cmdParser.add<int>("input_size", 'i', "input size", false, 64);
    cmdParser.add<int>("loop", 'p', "loop", false, 100);
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    auto model_name = cmdParser.get<std::string>("model_name");
    int input_size = cmdParser.get<int>("input_size");
    int loop = cmdParser.get<int>("loop");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    if (model_name == "tinyllama-1.1B") {
        TinyLLaMAConfig config(tokens_limit, "1.1B", HFHUBROPE);
        auto model = TinyLLaMAModel(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "gemma-2B") {
        GemmaConfig config(tokens_limit, "2B", RoPEType::HFHUBROPE);
        auto model = GemmaForCausalLM(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "qwen-0.5B") {
        QWenConfig config(tokens_limit, "0.5B", RoPEType::HFHUBROPE);
        auto model = QWenForCausalLM(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "qwen-1.8B") {
        QWenConfig config(tokens_limit, "1.8B", RoPEType::HFHUBROPE);
        auto model = QWenForCausalLM(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "stablelm-1.6B") {
        StableLMConfig config(tokens_limit, "1.6B", HFHUBROPE);
        auto model = StableLMModel(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "opt-1.3B") {
        OPTConfig config(tokens_limit, "1.3B");
        auto model = OPTModel(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "minicpm-2B") {
        MiniCPMConfig config(tokens_limit, "2B");
        auto model = MiniCPMForCausalLM(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "smollm-360M") {
        SmolLMConfig config(tokens_limit, "360M", RoPEType::HFHUBROPE, 49152);
        auto model = SmolLMModel(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "smollm-1.7B") {
        SmolLMConfig config(tokens_limit, "1.7B", RoPEType::HFHUBROPE, 49152);
        auto model = SmolLMModel(config);
        model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        auto input_tensor = tokens2Input(input_size);
        for (int step = 0; step < loop + 1; step++) {
            auto result = model({input_tensor});
            chatPostProcessing(0, input_tensor, {});
        }
        model.profiling();
    } else if (model_name == "dclm-1B") {
        // TODO
    } else if (model_name == "openelm-1.1B") {
        // TODO
    } else if (model_name == "openelm-450M") {
        // TODO
    } else if (model_name == "qwen2.5-0.5B") {
        // QWen2_5Config config(tokens_limit, "0.5B", RoPEType::HFHUBROPE);
        // auto model = QWen2_5ForCausalLM(config);
        // model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        // auto input_tensor = tokens2Input(input_size);
        // for (int step = 0; step < loop + 1; step++) {
        //     auto result = model({input_tensor});
        //     chatPostProcessing(0, input_tensor, {});
        // }
        // model.profiling();
    } else if (model_name == "qwen2.5-1.5B") {
        // QWen2_5Config config(tokens_limit, "1.5B", RoPEType::HFHUBROPE);
        // auto model = QWen2_5ForCausalLM(config);
        // model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);

        // auto input_tensor = tokens2Input(input_size);
        // for (int step = 0; step < loop + 1; step++) {
        //     auto result = model({input_tensor});
        //     chatPostProcessing(0, input_tensor, {});
        // }
        // model.profiling();
    }
    return 0;
}
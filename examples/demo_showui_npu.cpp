#include <cstdlib>
#include <cstring>
#include <iostream>
#include "QNNBackend.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include "cmdline.h"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"
#include "processor/PostProcess.hpp"
#include "memory/MemInspect.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/showui_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/showui_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/showui-w8-fpbias-noshadow-xdl-test.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1000);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    const string cpu_model_path = "../models/showui-2B-rotated-q40.mllm";
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // TODO: add a function to calculate the chunk size
    const int chunk_size = 256;

    Module::initBackend(MLLM_QNN);

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLConfig config(tokens_limit, "1.5b-rotated");
    auto model_config = Qwen2VLConfig(config);
    model_config.attn_implementation = "eager";

    auto prefill_embedding = Qwen2VL_ImagePatchAndEmbedding(config);
    auto prefill_body = Qwen2VL_PrefillBody(config, chunk_size);
    prefill_embedding.load(cpu_model_path);
    prefill_body.load(model_path);

    auto decoding_model = Qwen2VL_Decoding_Model(model_config);
    decoding_model.load(cpu_model_path);

    vector<string> in_imgs = {
        "../assets/showui.png"};
    vector<string> in_strs = {
        "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.<|vision_start|><|image_pad|><|vision_end|>桌面",
    };

    auto &in_str = in_strs[0];
    in_str = processor.tokenizer->apply_chat_template(in_str);
    auto input_tensors = processor.process(in_str, in_imgs[0]);

    const int real_seq_length = input_tensors[0].sequence();
    std::cout << "real seq length: " << real_seq_length << std::endl;

    const int num_iter = (real_seq_length + chunk_size - 1) / chunk_size;
    std::cout << "num_iter" << num_iter << std::endl;
    // padding the position_ids to total chunk length(example: 256*2) for CPUMultimodalRoPEPipeline
    prefill_embedding.get_position_ids(input_tensors, chunk_size * num_iter);

    // warm up (still need a warm up as the setup stage is not omitted now)
    auto merged_embd_warmup_tensor = Tensor(Backend::global_backends[MLLM_QNN]);
    merged_embd_warmup_tensor.reshape(1, 1, chunk_size, 1536);
    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    merged_embd_warmup_tensor.alloc();

    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    input_tensors.back().setTtype(INPUT_TENSOR);
    vector<Tensor> prefill_input = {merged_embd_warmup_tensor, input_tensors.back()};

    auto warm_start = mllm_time_ms();
    prefill_body(prefill_input);
    auto warm_end = mllm_time_ms();
    std::cout << "warm up " << warm_end - warm_start << " ms" << std::endl;

    Module::isFirstChunk = false;
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

    // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setTotalSequenceLength(real_seq_length);
    // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setChunkSize(chunk_size);

    for (auto &t : input_tensors) {
        t.setTtype(INPUT_TENSOR);
    }

    // 1. get the vit embedding using CPU
    auto vit_start = mllm_time_ms();
    auto merged_embd = prefill_embedding(input_tensors);
    auto vit_end = mllm_time_ms();
    std::cout << "vit embedding: " << vit_end - vit_start << " ms" << std::endl;

    // free prefill embedding tensor, approximately free 1GB for 59ms
    auto begin_free = mllm_time_ms();
    auto &embedding_act = prefill_embedding.activation_tensors;
    // go through the activation tensors to get the merged_embd
    for (auto iter = embedding_act.begin(); iter != embedding_act.end(); ++iter) {
        // std::cout << iter->first << std::endl;
        if (iter->first.find("input") != std::string::npos || iter->first.find("index_put") != std::string::npos) {
            continue;
        }
        iter->second->free();
    }
    auto end_free = mllm_time_ms();
    std::cout << "free time: " << end_free - begin_free << " ms" << std::endl;

    // 2. QNN LLM Prefill
    unsigned int out_token = 0;
    auto start_time = mllm_time_ms();
    for (auto i = 0; i < num_iter; ++i) {
        // copy the data from merged_embd[0] to merged_embd_warmup_tensor
        auto source = merged_embd[0].ptrAt<float>(0, 0, chunk_size * i, 0);
        auto dest = prefill_input[0].hostPtr<void>();
        if (i == 0) {
            memcpy(dest, source, prefill_input[0].cntSize());
        }
        {
            memcpy(dest, source, (merged_embd[0].sequence() % chunk_size) * merged_embd[0].dimension() * sizeof(float));
        }

        auto result = prefill_body(prefill_input);

        if (i == 0) { // turn off switching to avoid RoPE h_cnt_ reset to curSequenceLength in next chunk
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
        }

        if (i == 1) {
            auto end_time = mllm_time_ms();
            std::cout << "Prefill:" << end_time - start_time << " ms" << std::endl;

            auto outputs = processor.detokenize(result[0], real_seq_length % chunk_size);
            auto out_string = outputs.first;
            out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
            std::cout << output_string << std::flush;
        }
    }

    chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(real_seq_length);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(AUTOREGRESSIVE);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

    // 3. CPU LLM Decoding
    for (auto &t : input_tensors) { // set to INPUT_TENSOR to let decoding module update act
        t.setTtype(INPUT_TENSOR);
    }

    const int last_position_id = input_tensors[3].dataAt<float>(0, 0, 0, real_seq_length - 1);
    for (int step = 0; step < 100; step++) {
        // use the last position id(no padding position) in decoding
        prefill_embedding.get_position_ids(input_tensors, 0, last_position_id + 1 + step);

        auto result = decoding_model(input_tensors);
        auto outputs = processor.detokenize(result[0]);
        auto out_string = outputs.first;
        auto out_token = outputs.second;
        auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
        if (!not_end) { break; }
        std::cout << output_string << std::flush;
        chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

        if (step == 0) static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
    }

    std::cout << std::endl;

    if (!std::filesystem::exists("qnn_context.bin")) {
        static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
    }

    return 0;
}
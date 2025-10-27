#include "Context.hpp"
#include "QNNBackend.hpp"
#include <cstdlib>
#include <cstring>
#include "Types.hpp"
#include "cmdline.h"
#include "memory/MemInspect.hpp"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npuvit.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2vl_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2vl_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen2_vl_vit_lm_rota_noshadow.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1000);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    const string cpu_model_path = "../models/Qwen2-VL-2B-Instruct_vit_lm_rotated-Q40.mllm";
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // TODO: add a function to calculate the chunk size
    const int chunk_size = 128;

    Module::initBackend(MLLM_QNN);

    Context::Instance().inference_state().setCPUViT(false);

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLNPUConfig npu_config(tokens_limit, "1.5b-vl-rotated");

    // npu vit embedding
    auto prefill_embedding = npu::Qwen2VL_ImagePatchAndEmbedding(npu_config);
    prefill_embedding.load(model_path);

    // npu llm
    auto prefill_body = Qwen2VL_PrefillBody(npu_config, chunk_size, npu_config.shadow_layers);
    prefill_body.load(model_path);

    // cpu model
    auto cpu_model_config = Qwen2VLConfig(tokens_limit, "1.5b");
    cpu_model_config.attn_implementation = "eager_notrans";
    auto decoding_model = Qwen2VL_Decoding_Model(cpu_model_config);
    decoding_model.load(cpu_model_path);

    vector<string> in_imgs = {
        "../assets/bus.png"};
    vector<string> in_strs = {
        "<|vision_start|><|image_pad|><|vision_end|>Imagine you are describing this image to someone who cannot see it. Explain everything you observe, including the background, subjects, their expressions, and any activities they appear to be doing.",
    };

    auto &in_str = in_strs[0];
    in_str = processor.tokenizer->apply_chat_template(in_str);
    auto input_tensors = processor.process(in_str, in_imgs[0]);

    const int real_seq_length = input_tensors[0].sequence();
    std::cout << "real seq length: " << real_seq_length << std::endl;

    const int num_iter = (real_seq_length + chunk_size - 1) / chunk_size;
    std::cout << "num_iter: " << num_iter << std::endl;
    // padding the position_ids to total chunk length(example: 256*2) for CPUMultimodalRoPEPipeline
    prefill_embedding.get_position_ids(input_tensors, chunk_size * num_iter);

    // 1. QNN vit embedding
    // NOTE: put vit here is because compatible with older qnn_context.bin.
    // In QNNBackend, the graph should be executed in the order of the context
    // TODO: better QNNBackend graph indexing and management
    auto vit_start = mllm_time_ms();
    auto merged_embd = prefill_embedding(input_tensors);
    auto vit_end = mllm_time_ms();

    auto merged_embd_warmup_tensor = Tensor(0, MLLM_QNN);
    merged_embd_warmup_tensor.reshape(1, 1, chunk_size, 1536);
    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    merged_embd_warmup_tensor.alloc();

    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    input_tensors.back().setTtype(INPUT_TENSOR);
    vector<Tensor> prefill_input = {merged_embd_warmup_tensor, input_tensors.back()};

    auto llm_start = mllm_time_ms();
    prefill_body(prefill_input);
    auto llm_end = mllm_time_ms();
    std::cout << "after warm up" << std::endl;

    if (!std::filesystem::exists("qnn_context.bin")) {
        static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
    }

    Context::Instance().inference_state().setQnnGraphFrozen(true);
    Context::Instance().inference_state().setCurSequenceLength(0);
    Context::Instance().inference_state().setExecutionType(PROMPT);
    Context::Instance().inference_state().toggleSwitching();

    // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
    Context::Instance().inference_state().setTotalSequenceLength(real_seq_length);
    // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
    Context::Instance().inference_state().setChunkSize(chunk_size);

    std::cout << "[Q] " << in_strs[0] << std::endl;
    std::cout << "[A] " << std::flush;

    for (auto &t : input_tensors) {
        t.setTtype(INPUT_TENSOR);
    }

    // 2. QNN LLM Prefill
    unsigned int out_token = 0;
    auto start_time = mllm_time_ms();
    int64_t prefill_time;
    for (auto i = 0; i < num_iter; ++i) {
        // copy the data from merged_embd[0] to merged_embd_warmup_tensor
        auto source = merged_embd[0].ptrAt<float>(0, 0, chunk_size * i, 0);
        auto dest = prefill_input[0].hostPtr<void>();
        if (i == 0) {
            memcpy(dest, source, std::min(prefill_input[0].cntSize(), merged_embd[0].cntSize()));
        } else {
            memcpy(dest, source, (merged_embd[0].sequence() % chunk_size) * merged_embd[0].dimension() * sizeof(float));
        }

        auto result = prefill_body(prefill_input);

        if (i == 0) { // turn off switching to avoid RoPE h_cnt_ reset to curSequenceLength in next chunk
            Context::Instance().inference_state().toggleSwitching();
        }

        if (i == num_iter - 1) {
            auto end_time = mllm_time_ms();
            prefill_time = end_time - start_time;
            auto outputs = processor.detokenize(result[0], real_seq_length % chunk_size);
            auto out_string = outputs.first;
            out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
            std::cout << output_string << std::flush;
        }
    }

    chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

    Context::Instance().inference_state().setCurSequenceLength(real_seq_length);
    Context::Instance().inference_state().setExecutionType(AUTOREGRESSIVE);
    Context::Instance().inference_state().toggleSwitching();

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

        if (step == 0) Context::Instance().inference_state().toggleSwitching();
    }

    std::cout << std::endl;
    std::cout << "vit embedding time: " << vit_end - vit_start << " ms" << std::endl;
    std::cout << "Prefill:" << prefill_time << " ms" << std::endl;
    return 0;
}
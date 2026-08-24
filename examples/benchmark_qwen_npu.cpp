#include "Backend.hpp"
#include "Context.hpp"
#include "Parallel.hpp"
#include "QNNBackend.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "backends/cpu/AttentionProfiler.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/modeling_qwen_npu_v2.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "shadownpu_head_schedule.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

using namespace mllm;

namespace {

class BackendCleanup {
public:
    ~BackendCleanup() {
        // Traced wrappers retain tensors backed by the global backends. Drop
        // them first, then destroy QNN while its dlopened libraries are live.
        Tracer::model_.clear();
        Backend::global_backends.clear();
    }
};

ProfilingLevel parseProfilingLevel(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "off") return ProfilingLevel::OFF;
    if (value == "basic") return ProfilingLevel::BASIC;
    if (value == "detailed") return ProfilingLevel::DETAILED;
    throw std::invalid_argument("qnn-profile must be one of: off, basic, detailed");
}

void pinCurrentThread(int cpu) {
    if (cpu < 0) return;
#if defined(__linux__)
    if (cpu >= CPU_SETSIZE) {
        throw std::invalid_argument(
            "main-cpu exceeds CPU_SETSIZE: " + std::to_string(cpu));
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
        throw std::runtime_error(
            "failed to pin main thread to CPU " + std::to_string(cpu)
            + ": " + std::strerror(errno));
    }
#else
    (void)cpu;
    throw std::runtime_error("main-cpu requires Linux/Android");
#endif
}

std::string jsonEscape(const std::string &input) {
    std::ostringstream escaped;
    for (const unsigned char c : input) {
        switch (c) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (c < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
            } else {
                escaped << static_cast<char>(c);
            }
        }
    }
    return escaped.str();
}

uint64_t hashTokens(const std::vector<token_id_t> &tokens) {
    uint64_t hash = 1469598103934665603ULL;
    for (const token_id_t token : tokens) {
        uint32_t value = static_cast<uint32_t>(token);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<unsigned char>(value & 0xffU);
            hash *= 1099511628211ULL;
            value >>= 8;
        }
    }
    return hash;
}

std::string answerBeforeEndMarker(const std::string &text) {
    size_t end = text.size();
    for (const std::string marker : {std::string("\0", 1),
                                     std::string("<|im_end|>"),
                                     std::string("<|endoftext|>")}) {
        const size_t position = text.find(marker);
        if (position != std::string::npos) end = std::min(end, position);
    }
    return text.substr(0, end);
}

std::string extractAccessCode(const std::string &text) {
    static const std::regex code_pattern(R"([A-Z]{3}-[0-9]{4})");
    std::smatch match;
    return std::regex_search(text, match, code_pattern) ? match.str()
                                                        : std::string();
}

} // namespace

int main(int argc, char **argv) {
    cmdline::parser cmd_parser;
    cmd_parser.add<std::string>("vocab", 'v', "tokenizer model", false, "../vocab/qwen2.5_vocab.mllm");
    cmd_parser.add<std::string>("merge", 'e', "tokenizer merges", false, "../vocab/qwen2.5_merges.txt");
    cmd_parser.add<std::string>("qnn-model", 'm', "NPU prefill model", false, "../models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm");
    cmd_parser.add<std::string>("decoding-model", '\0', "CPU decoding model", false, "../models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm");
    cmd_parser.add<std::string>("model-size", 'b', "Qwen model size", false, "1.5B-rotated");
    cmd_parser.add<std::string>("qnn-profile", '\0', "QNN profiling: off, basic, or detailed", false, "off");
    cmd_parser.add<int>("limits", 'l', "KV cache size", false, 2048);
    cmd_parser.add<int>("thread", 't', "CPU decoding threads", false, 4);
    cmd_parser.add<int>("prompt-tokens", 'p', "effective synthetic prompt tokens", false, 1024);
    cmd_parser.add<int>("decode-tokens", 'd', "CPU tokens generated after the first token", false, 32);
    cmd_parser.add<int>("chunk-size", 'c', "NPU prefill graph chunk size", false, 256);
    cmd_parser.add<int>(
        "main-cpu", '\0', "pin the main/model CPU thread; -1 disables pinning",
        false, -1);
    cmd_parser.add<std::string>(
        "chunk-overlap", '\0', "chunk pipeline overlap: auto, on, or off",
        false, "auto");
    cmd_parser.add<std::string>(
        "attention-mode", '\0', "dense or hmx-topk", false,
        "dense");
    cmd_parser.add<std::string>(
        "head-retain-profile", '\0',
        "ShadowNPU per-head retention profile (required for hmx-topk)",
        false, "");
    cmd_parser.add<std::string>(
        "needle", '\0', "quality-test access code ([A-Z]{3}-[0-9]{4})", false,
        "NPU-7391");
    cmd_parser.add<std::string>(
        "quality-prompt", '\0', "quality prompt: legacy or records", false,
        "legacy");
    cmd_parser.parse_check(argc, argv);

    const int chunk_size = cmd_parser.get<int>("chunk-size");
    const int prompt_token_count = cmd_parser.get<int>("prompt-tokens");
    const int decode_token_count = cmd_parser.get<int>("decode-tokens");
    const int token_limit = cmd_parser.get<int>("limits");
    const int main_cpu = cmd_parser.get<int>("main-cpu");
    const std::string chunk_overlap_policy =
        cmd_parser.get<std::string>("chunk-overlap");
    const std::string attention_mode =
        cmd_parser.get<std::string>("attention-mode");
    const std::string head_retain_profile =
        cmd_parser.get<std::string>("head-retain-profile");
    const std::string needle = cmd_parser.get<std::string>("needle");
    const std::string quality_prompt =
        cmd_parser.get<std::string>("quality-prompt");
    if (prompt_token_count <= 0 || chunk_size <= 0 || decode_token_count <= 0) {
        std::cerr << "prompt-tokens, decode-tokens, and chunk-size must be positive" << std::endl;
        return 2;
    }
    if (attention_mode != "dense" && attention_mode != "hmx-topk") {
        std::cerr << "attention-mode must be dense or hmx-topk"
                  << std::endl;
        return 2;
    }
    if (attention_mode == "hmx-topk" && head_retain_profile.empty()) {
        std::cerr << "hmx-topk requires --head-retain-profile" << std::endl;
        return 2;
    }
    if (!std::regex_match(needle, std::regex(R"([A-Z]{3}-[0-9]{4})"))) {
        std::cerr << "needle must match [A-Z]{3}-[0-9]{4}" << std::endl;
        return 2;
    }
    if (quality_prompt != "legacy" && quality_prompt != "records") {
        std::cerr << "quality-prompt must be legacy or records" << std::endl;
        return 2;
    }
    if (main_cpu < -1) {
        std::cerr << "main-cpu must be -1 or a non-negative CPU id"
                  << std::endl;
        return 2;
    }
    if (chunk_overlap_policy != "auto" && chunk_overlap_policy != "on"
        && chunk_overlap_policy != "off") {
        std::cerr << "chunk-overlap must be auto, on, or off" << std::endl;
        return 2;
    }
    const int padded_prompt_tokens =
        ChunkPipeline::paddedSequenceLengthFor(prompt_token_count, chunk_size);
    const int required_token_limit =
        std::max(padded_prompt_tokens, prompt_token_count + decode_token_count + 1);
    if (token_limit < required_token_limit) {
        std::cerr << "limits must be at least " << required_token_limit
                  << " to cover padded prefill and CPU decode tokens" << std::endl;
        return 2;
    }

    try {
        QNNBackend::setDefaultProfilingLevel(
            parseProfilingLevel(cmd_parser.get<std::string>("qnn-profile")));
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }

    try {
        pinCurrentThread(main_cpu);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
    if (main_cpu >= 0) {
        const std::string main_cpu_value = std::to_string(main_cpu);
        setenv("MLLM_HMX_PIPELINE_MAIN_CPU", main_cpu_value.c_str(), 1);
    } else {
        unsetenv("MLLM_HMX_PIPELINE_MAIN_CPU");
    }
    CPUBackend::cpu_threads = cmd_parser.get<int>("thread");
    Module::initBackend(MLLM_QNN);
    BackendCleanup backend_cleanup;

    QWenTokenizer tokenizer(cmd_parser.get<std::string>("vocab"),
                            cmd_parser.get<std::string>("merge"));
    QWenNPUConfig config(token_limit, cmd_parser.get<std::string>("model-size"),
                         RoPEType::HFHUBROPE);
    config.prefill_attention_pattern_sparse = attention_mode == "hmx-topk";
    config.prefill_attention_hmx_selector = attention_mode == "hmx-topk";
    mllm::examples::ShadowNPUHeadSchedule head_schedule;
    if (!head_retain_profile.empty()) {
        try {
            head_schedule = mllm::examples::loadShadowNPUHeadSchedule(
                head_retain_profile, config.num_hidden_layers,
                config.num_attention_heads);
        } catch (const std::invalid_argument &error) {
            std::cerr << error.what() << std::endl;
            return 2;
        }
        config.prefill_attention_head_retentions =
            head_schedule.retentions;
    }
    const float requested_sparsity = attention_mode == "dense"
        ? 0.0F : 1.0F - head_schedule.average_retention;

    std::string synthetic_user_prompt;
    if (quality_prompt == "legacy") {
        synthetic_user_prompt =
            "Memorize this exact access code: " + needle
            + ". You will be asked for it after a long archive.\n";
        const std::string sentence =
            "Archive note: mobile inference processes context in deterministic chunks. ";
        const int sentence_repetitions = std::max(512, prompt_token_count / 4);
        synthetic_user_prompt.reserve(
            synthetic_user_prompt.size()
            + sentence.size() * sentence_repetitions + 160);
        for (int i = 0; i < sentence_repetitions; ++i) {
            synthetic_user_prompt += sentence;
        }
        synthetic_user_prompt +=
            "\nQuestion: What is the exact access code stated at the beginning? "
            "Answer with the code only.";
    } else {
        synthetic_user_prompt =
            "Archive target record 0000 contains access code " + needle
            + ". Preserve this target exactly.\n";
        const int records = std::max(1024, prompt_token_count / 2);
        for (int i = 1; i <= records; ++i) {
            synthetic_user_prompt += "Archive record " + std::to_string(i)
                + ": routine mobile inference note, checksum "
                + std::to_string((i * 7919) % 10000) + ".\n";
        }
        synthetic_user_prompt +=
            "Decoy warning: CPU-1842 is obsolete and is not the target.\n"
            "Question: What access code is stored in archive target record 0000? "
            "Answer with the target code only.";
    }

    auto formatted_prompt = tokenizer.apply_chat_template(synthetic_user_prompt);
    auto [available_tokens, source_tensor] =
        tokenizer.tokenizePaddingByChunk(
            formatted_prompt, chunk_size, config.vocab_size);
    if (available_tokens < prompt_token_count) {
        std::cerr << "synthetic prompt unexpectedly produced only "
                  << available_tokens << " tokens" << std::endl;
        return 3;
    }

    std::vector<token_id_t> prompt_tokens;
    prompt_tokens.reserve(padded_prompt_tokens);
    const int tail_count = std::min(256, prompt_token_count / 3);
    for (int i = 0; i < prompt_token_count - tail_count; ++i) {
        prompt_tokens.push_back(
            static_cast<token_id_t>(source_tensor.dataAt<float>(0, 0, i, 0)));
    }
    for (int i = available_tokens - tail_count; i < available_tokens; ++i) {
        prompt_tokens.push_back(
            static_cast<token_id_t>(source_tensor.dataAt<float>(0, 0, i, 0)));
    }
    const uint64_t prompt_hash = hashTokens(prompt_tokens);
    prompt_tokens.resize(padded_prompt_tokens, config.vocab_size);
    auto input_tensor = Tokenizer::tokens2Input(prompt_tokens);

    config.attn_implementation = "eager_notrans";

    v2::QWenForCausalLM_NPU model(config, chunk_size);
    model.load(cmd_parser.get<std::string>("qnn-model"));
    QWenForCausalLM decoding_model(config);
    decoding_model.load(cmd_parser.get<std::string>("decoding-model"));

    std::vector<token_id_t> trace_tokens(chunk_size, 1);
    auto trace_input = Tokenizer::tokens2Input(trace_tokens);
    Tracer::trace(&model, {trace_input});
    bool context_generated = false;
    if (std::getenv("MLLM_SKIP_QNN_CONTEXT_SAVE") == nullptr
        && !std::filesystem::exists("qnn_context.bin")) {
        if (!static_cast<QNNBackend *>(
                 Backend::global_backends[MLLM_QNN].get())
                 ->saveQNNContext()) {
            std::cerr << "QNN context generation failed" << std::endl;
            return 1;
        }
        context_generated = true;
    }
    if (context_generated
        && std::getenv("MLLM_QNN_CONTEXT_GENERATE_ONLY") != nullptr) {
        std::cout << "QNN_CONTEXT_GENERATE_ONLY success chunk_size="
                  << chunk_size << std::endl;
        return 0;
    }

    auto &state = Context::Instance().inference_state();
    state.reset();
    state.setTotalSequenceLength(prompt_token_count);
    state.setChunkSize(chunk_size);

    LlmTextGeneratorOpts prefill_opt{
        .max_new_tokens = 1,
        .do_sample = false,
        .is_padding = true,
        .seq_before_padding = prompt_token_count,
        .chunk_size = chunk_size,
    };

    bool is_switched = false;
    ChunkPipeline pipeline(prompt_token_count, chunk_size);
    CPUAttentionProfiler::reset();
    CPUSparseSelectionStats::reset();
    const char *last_token_refine_env =
        std::getenv("MLLM_QWEN_CPU_LAST_TOKEN_REFINE");
    const bool cpu_last_token_refine = last_token_refine_env != nullptr
        && std::strcmp(last_token_refine_env, "0") != 0;
    const int64_t request_start_us = mllm_time_us();
    const bool parallel_chunks = chunk_overlap_policy == "on"
        || (chunk_overlap_policy == "auto" && attention_mode != "hmx-topk");
    const std::string chunk_schedule = parallel_chunks
        ? "overlap" : "chunk-major";
    auto prefill_result = pipeline.run(
        input_tensor, prefill_opt, tokenizer, model, is_switched, {},
        parallel_chunks, !cpu_last_token_refine);
    const int64_t npu_prefill_end_us = mllm_time_us();
    const auto attention_profile = CPUAttentionProfiler::snapshot();
    const auto sparse_selection = CPUSparseSelectionStats::snapshot();

    state.setQnnGraphFrozen(true);
    state.setCurSequenceLength(
        prompt_token_count - (cpu_last_token_refine ? 1 : 0));
    state.setExecutionType(AUTOREGRESSIVE);
    state.toggleSwitching();

    Tensor decoding_input;
    decoding_input.setBackend(Backend::global_backends[MLLM_CPU].get());
    decoding_input.setTtype(INPUT_TENSOR);
    decoding_input.reshape(1, 1, 1, 1);
    decoding_input.setName("input0");
    decoding_input.alloc();
    const auto npu_first_token = static_cast<token_id_t>(
        prefill_result->dataAt<float>(0, 0, 0, 0));

    LlmTextGeneratorOpts decoding_opt{
        .max_new_tokens = static_cast<size_t>(decode_token_count),
        .do_sample = false,
        .temperature = 0.3F,
        .top_k = 50,
        .top_p = 0.F,
        .is_padding = false,
    };

    int64_t first_token_us = npu_prefill_end_us;
    std::vector<int64_t> decode_completion_us;
    std::vector<token_id_t> generated_ids;
    decode_completion_us.reserve(decode_token_count);
    generated_ids.reserve(decode_token_count + 1);

    auto finish_stage_switch = [&]() {
        if (state.isStageSwitching()) state.toggleSwitching();
        is_switched = true;
    };
    if (cpu_last_token_refine) {
        LlmTextGeneratorOpts refine_opt = decoding_opt;
        refine_opt.max_new_tokens = 1;
        bool refined_token_produced = false;
        token_id_t refined_first_token = 0;
        decoding_input.setDataAt(
            0, 0, 0, 0,
            prompt_tokens[static_cast<std::size_t>(prompt_token_count - 1)]);
        decoding_model.generate(
            decoding_input, refine_opt,
            [&](unsigned int token) -> bool {
                finish_stage_switch();
                refined_first_token = static_cast<token_id_t>(token);
                refined_token_produced = true;
                return true;
            });
        if (!refined_token_produced) {
            std::cerr << "CPU last-token refinement produced no token"
                      << std::endl;
            return 4;
        }
        first_token_us = mllm_time_us();
        generated_ids.push_back(refined_first_token);
        decoding_input.setDataAt(0, 0, 0, 0, refined_first_token);
    } else {
        generated_ids.push_back(npu_first_token);
        decoding_input.setDataAt(0, 0, 0, 0, npu_first_token);
    }

    decoding_model.generate(decoding_input, decoding_opt,
                            [&](unsigned int token) -> bool {
        finish_stage_switch();
        generated_ids.push_back(static_cast<token_id_t>(token));
        decode_completion_us.push_back(mllm_time_us());
        return true;
    });

    if (static_cast<int>(decode_completion_us.size()) != decode_token_count) {
        std::cerr << "decoded " << decode_completion_us.size()
                  << " tokens, expected " << decode_token_count << std::endl;
        return 4;
    }

    const double ttft_ms = static_cast<double>(first_token_us - request_start_us) / 1000.0;
    const double decode_elapsed_ms =
        static_cast<double>(decode_completion_us.back() - first_token_us) / 1000.0;
    const double tpot_ms = decode_elapsed_ms / decode_completion_us.size();

    std::vector<double> inter_token_ms;
    inter_token_ms.reserve(decode_completion_us.size());
    int64_t previous_token_us = first_token_us;
    for (const auto completion_us : decode_completion_us) {
        inter_token_ms.push_back(
            static_cast<double>(completion_us - previous_token_us) / 1000.0);
        previous_token_us = completion_us;
    }
    auto sorted_inter_token_ms = inter_token_ms;
    std::sort(sorted_inter_token_ms.begin(), sorted_inter_token_ms.end());
    const size_t middle = sorted_inter_token_ms.size() / 2;
    const double p50_tpot_ms = sorted_inter_token_ms.size() % 2 == 0
        ? (sorted_inter_token_ms[middle - 1] + sorted_inter_token_ms[middle]) / 2.0
        : sorted_inter_token_ms[middle];
    const size_t p95_index = (sorted_inter_token_ms.size() * 95 + 99) / 100 - 1;
    const double p95_tpot_ms = sorted_inter_token_ms[p95_index];
    const double steady_tpot_ms = inter_token_ms.size() > 1
        ? std::accumulate(inter_token_ms.begin() + 1, inter_token_ms.end(), 0.0)
              / static_cast<double>(inter_token_ms.size() - 1)
        : inter_token_ms.front();

    const std::string generated_text = tokenizer.detokenize(generated_ids);
    const uint64_t generated_hash = hashTokens(generated_ids);
    const std::string quality_answer = answerBeforeEndMarker(generated_text);
    const std::string extracted_code = extractAccessCode(quality_answer);
    const bool retrieval_exact = extracted_code == needle;
    const bool needle_contains = generated_text.find(needle) != std::string::npos;
    const auto first_non_space = generated_text.find_first_not_of(" \t\r\n");
    const auto last_non_space = generated_text.find_last_not_of(" \t\r\n");
    const std::string trimmed_text = first_non_space == std::string::npos
        ? std::string()
        : generated_text.substr(
              first_non_space, last_non_space - first_non_space + 1);
    const bool needle_exact = trimmed_text == needle;
    std::cout << "GENERATED_IDS";
    for (const token_id_t token : generated_ids) std::cout << ' ' << token;
    std::cout << '\n'
              << "GENERATED_TEXT \"" << jsonEscape(generated_text) << "\"\n";
    std::cout << "QUALITY_ANSWER \"" << jsonEscape(quality_answer) << "\"\n"
              << "QUALITY_RESULT expected_code=" << needle
              << " extracted_code="
              << (extracted_code.empty() ? "none" : extracted_code)
              << " retrieval_exact=" << retrieval_exact << '\n';

    const bool using_head_profile = !head_schedule.retentions.empty();
    const int sparse_prefill_layers = using_head_profile
        ? static_cast<int>(std::count_if(
              head_schedule.retentions.begin(),
              head_schedule.retentions.end(),
              [](const std::vector<float> &retentions) {
                  return std::any_of(
                      retentions.begin(), retentions.end(),
                      [](float retention) { return retention < 1.0F; });
              })) : 0;
    const double actual_retained_fraction = sparse_selection.eligible == 0
        ? 1.0
        : static_cast<double>(sparse_selection.retained)
            / static_cast<double>(sparse_selection.eligible);

    std::cout << std::fixed << std::setprecision(3)
              << "\nBENCH_RESULT"
              << " prompt_tokens=" << prompt_token_count
              << " padded_tokens=" << padded_prompt_tokens
              << " chunks=" << pipeline.chunkCount()
              << " parallel_chunks=" << parallel_chunks
              << " chunk_overlap_policy=" << chunk_overlap_policy
              << " chunk_schedule=" << chunk_schedule
              << " main_cpu=" << main_cpu
              << " decode_tokens=" << decode_completion_us.size()
              << " ttft_ms=" << ttft_ms
              << " tpot_ms=" << tpot_ms
              << " first_decode_ms=" << inter_token_ms.front()
              << " steady_tpot_ms=" << steady_tpot_ms
              << " p50_tpot_ms=" << p50_tpot_ms
              << " p95_tpot_ms=" << p95_tpot_ms
              << " decode_tok_s=" << 1000.0 / tpot_ms
              << " attention_mode=" << attention_mode
              << " quality_prompt=" << quality_prompt
              << " requested_sparsity=" << std::setprecision(6)
              << (attention_mode == "dense" ? 0.0F : requested_sparsity)
              << " retained_fraction="
              << (attention_mode == "dense" ? 1.0F
                                               : 1.0F - requested_sparsity)
              << " actual_retained_fraction=" << actual_retained_fraction
              << " scheduled_overall_retained_fraction="
              << (using_head_profile ? head_schedule.average_retention : 1.0F)
              << " head_profile="
              << (using_head_profile ? head_retain_profile : "none")
              << " dense_heads="
              << (using_head_profile ? head_schedule.dense_heads : 0)
              << std::setprecision(3)
              << " dense_layers="
              << (using_head_profile
                      ? head_schedule.normalized_dense_layers : "all")
              << " sparse_prefill_layers=" << sparse_prefill_layers
              << " total_layers=" << config.num_hidden_layers
              << " prompt_hash=" << prompt_hash
              << " generated_hash=" << generated_hash
              << " decode_attention=dense"
              << " needle=" << needle
              << " needle_contains=" << needle_contains
              << " needle_exact=" << needle_exact
              << " retrieval_exact=" << retrieval_exact
              << " qnn_profile=" << cmd_parser.get<std::string>("qnn-profile")
              << " cpu_last_token_refine=" << cpu_last_token_refine
              << std::endl;
    if (CPUAttentionProfiler::enabled()) {
        const auto profile_ms = [&](AttentionProfileStage stage) {
            return attention_profile.milliseconds(stage);
        };
        const auto profile_calls = [&](AttentionProfileStage stage) {
            return attention_profile.callCount(stage);
        };
        const double legacy_profiled_attention_wall_ms =
            profile_ms(AttentionProfileStage::QK)
            + profile_ms(AttentionProfileStage::DENSE_SOFTMAX)
            + profile_ms(AttentionProfileStage::DENSE_PV)
            + profile_ms(AttentionProfileStage::SPARSE_TOTAL);
        const double attention_outer_wall_ms =
            profile_ms(AttentionProfileStage::ATTENTION_OUTER);
        const double profiled_attention_wall_ms =
            profile_calls(AttentionProfileStage::ATTENTION_OUTER) == 0
            ? legacy_profiled_attention_wall_ms
            : attention_outer_wall_ms;
        std::cout << std::fixed << std::setprecision(3)
                  << "QWEN_ATTN_PROFILE"
                  << " profiled_attention_wall_ms="
                  << profiled_attention_wall_ms
                  << " attention_outer_wall_ms="
                  << attention_outer_wall_ms
                  << " attention_outer_calls="
                  << profile_calls(AttentionProfileStage::ATTENTION_OUTER)
                  << " legacy_profiled_attention_wall_ms="
                  << legacy_profiled_attention_wall_ms
                  << " qk_ms=" << profile_ms(AttentionProfileStage::QK)
                  << " qk_calls=" << profile_calls(AttentionProfileStage::QK)
                  << " dense_softmax_ms="
                  << profile_ms(AttentionProfileStage::DENSE_SOFTMAX)
                  << " dense_softmax_calls="
                  << profile_calls(AttentionProfileStage::DENSE_SOFTMAX)
                  << " dense_pv_ms="
                  << profile_ms(AttentionProfileStage::DENSE_PV)
                  << " dense_pv_calls="
                  << profile_calls(AttentionProfileStage::DENSE_PV)
                  << " hmx_select_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SELECT)
                  << " hmx_select_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SELECT)
                  << " hmx_npu_produce_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_NPU_PRODUCE_WALL)
                  << " hmx_npu_produce_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_NPU_PRODUCE_WALL)
                  << " hmx_topk_stage_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_TOPK_STAGE_WALL)
                  << " hmx_topk_stage_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_TOPK_STAGE_WALL)
                  << " hmx_sparse_stage_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SPARSE_STAGE_WALL)
                  << " hmx_sparse_stage_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SPARSE_STAGE_WALL)
                  << " hmx_mutex_wait_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_MUTEX_WAIT)
                  << " hmx_mutex_wait_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_MUTEX_WAIT)
                  << " hmx_prepare_q_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_PREPARE_Q)
                  << " hmx_prepare_q_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_PREPARE_Q)
                  << " hmx_k_layout_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_K_LAYOUT)
                  << " hmx_k_layout_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_K_LAYOUT)
                  << " hmx_k_cache_load_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_K_CACHE_LOAD)
                  << " hmx_k_cache_load_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_K_CACHE_LOAD)
                  << " hmx_k_prepare_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_K_PREPARE)
                  << " hmx_k_prepare_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_K_PREPARE)
                  << " hmx_fused_prepare_execute_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_FUSED_PREPARE_EXECUTE)
                  << " hmx_fused_prepare_execute_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_FUSED_PREPARE_EXECUTE)
                  << " hmx_k_cache_store_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_K_CACHE_STORE)
                  << " hmx_k_cache_store_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_K_CACHE_STORE)
                  << " hmx_session_init_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SESSION_INIT)
                  << " hmx_session_init_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SESSION_INIT)
                  << " hmx_scope_begin_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SCOPE_BEGIN)
                  << " hmx_scope_begin_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SCOPE_BEGIN)
                  << " hmx_scope_end_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SCOPE_END)
                  << " hmx_scope_end_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SCOPE_END)
                  << " hmx_buffer_alloc_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_BUFFER_ALLOC)
                  << " hmx_buffer_alloc_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_BUFFER_ALLOC)
                  << " hmx_q_copy_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_Q_COPY)
                  << " hmx_q_copy_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_Q_COPY)
                  << " hmx_w_copy_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_W_COPY)
                  << " hmx_w_copy_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_W_COPY)
                  << " hmx_q_layout_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_Q_LAYOUT)
                  << " hmx_q_layout_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_Q_LAYOUT)
                  << " hmx_scale_profile_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SCALE_PROFILE)
                  << " hmx_scale_profile_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SCALE_PROFILE)
                  << " hmx_mm_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_MM)
                  << " hmx_mm_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_MM)
                  << " hmx_output_layout_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_OUTPUT_LAYOUT)
                  << " hmx_output_layout_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_OUTPUT_LAYOUT)
                  << " hmx_topk_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_TOPK)
                  << " hmx_topk_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_TOPK)
                  << " hmx_buffer_free_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_BUFFER_FREE)
                  << " hmx_buffer_free_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_BUFFER_FREE)
                  << " hmx_session_finalize_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::HMX_SESSION_FINALIZE)
                  << " hmx_session_finalize_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::HMX_SESSION_FINALIZE)
                  << " sparse_total_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::SPARSE_TOTAL)
                  << " sparse_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::SPARSE_TOTAL)
                  << " sparse_pack_wall_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::SPARSE_PACK)
                  << " sparse_pack_calls="
                  << attention_profile.callCount(
                         AttentionProfileStage::SPARSE_PACK)
                  << " pattern_qk_softmax_cpu_work_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::PATTERN_QK_SOFTMAX_WORK)
                  << " pattern_qk_cpu_work_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::PATTERN_QK_WORK)
                  << " pattern_softmax_cpu_work_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::PATTERN_SOFTMAX_WORK)
                  << " pattern_normalize_prep_cpu_work_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::PATTERN_NORMALIZE_PREP_WORK)
                  << " sparse_pv_cpu_work_ms="
                  << attention_profile.milliseconds(
                         AttentionProfileStage::SPARSE_PV_WORK)
                  << std::endl;
    }

    const auto qnn_tile_profile = static_cast<QNNBackend *>(
        Backend::global_backends[MLLM_QNN].get())->sequenceTileProfile();
    if (qnn_tile_profile.graph_calls != 0) {
        std::cout << std::fixed << std::setprecision(3)
                  << "QNN_SEQUENCE_TILE_PROFILE"
                  << " input_copy_ms="
                  << qnn_tile_profile.input_copy_us / 1000.0
                  << " execute_ms="
                  << qnn_tile_profile.execute_us / 1000.0
                  << " output_copy_ms="
                  << qnn_tile_profile.output_copy_us / 1000.0
                  << " graph_calls=" << qnn_tile_profile.graph_calls
                  << " tile_calls=" << qnn_tile_profile.tile_calls
                  << " scratch_bytes=" << qnn_tile_profile.scratch_bytes
                  << std::endl;
    }

    return 0;
}

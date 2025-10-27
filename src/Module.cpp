//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#include "Module.hpp"
#include "Types.hpp"
#include <iostream>
#include <stack>
#include <stdlib.h>
#include <vector>

namespace mllm {

// The llm_model_ptr is a pointer to the outmost module
Module *Module::llm_model_ptr;

int Module::listIdx;
std::stack<int> Module::listIdxStack;
// int Module::runlistIdx;
// TensorStatus Tensor::tensor_status;
BackendType Module::tmp_device = MLLM_CPU;
std::unordered_map<string, shared_ptr<Op>> Module::tensor_func_ops;
bool Module::alloc_mmap = true;

vector<double> Module::profiling(string name) {
    vector<double> output;
    // printf("\n");
    std::cout << "===========================================" << std::endl;
    if (!name.empty()) {
        std::cout << "            " << name << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }
    double load_time_s = load_time_ / 1000.0F;
    std::cout << "  Load time: " << load_time_ / 1000.0F << " s" << std::endl;
    if (inference_times_.size() > 1 && decoding_token_size_ != prefilling_token_size_) {
        double prefile_speed = 1000 * prefilling_token_size_ / inference_times_[0];
        std::cout << "  Prefilling speed: " << prefile_speed << " tokens/s , TTFT: " << inference_times_[0] / 1000.0F << " s" << std::endl;
        double sum_decoding_time = std::accumulate(std::begin(inference_times_) + 1, std::end(inference_times_), 0.0);
        double mean_decoding_time = sum_decoding_time / (inference_times_.size() - 1);
        double decoding_speed = 1000 / mean_decoding_time;
        std::cout << "  Decoding speed: " << decoding_speed << " tokens/s" << std::endl;
        output = {load_time_s, prefile_speed, decoding_speed};
    } else {
        double sum_time = std::accumulate(std::begin(inference_times_), std::end(inference_times_), 0.0);
        double mean_time = sum_time / (inference_times_.size());
        double inference_time_s = mean_time / 1000.0F;
        std::cout << "  Inference latency: " << mean_time / 1000.0F << " s" << std::endl;
        output = {load_time_s, inference_time_s};
    }
    // double sum_time = std::accumulate(std::begin(inference_times_), std::end(inference_times_), 0.0);
    // std::cout<<sum_time<< " - "<<Tensor::forward_times<<" = "<<sum_time-Tensor::forward_times<<std::endl;
    // std::cout<<Tensor::forward_times<< " - "<<Tensor::forward_times_2<<" = "<<Tensor::forward_times-Tensor::forward_times_2<<std::endl;

    std::cout << "===========================================" << std::endl;

    prefilling_token_size_ = 0;
    decoding_token_size_ = 0;
    inference_times_.clear();

    return output;
}

void Module::generate(
    Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back) {
    auto chatPostProcessing = [](unsigned token_idx, Tensor &tokens_tensor, const vector<Tensor *> &clean_tensors) {
        tokens_tensor.cpu();
        tokens_tensor.reshape(1, 1, 1, 1);
        tokens_tensor.alloc();
        tokens_tensor.setDataAt<float>(0, 0, 0, 0, token_idx);

        for (auto tensor : clean_tensors) {
            tensor->reshape(0, 0, 0, 0);
            tensor->alloc();
        }
    };

    if (!opt.do_sample) {
        // fail to greedy search
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearch, opt);
    } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.F) {
        // fail to top p sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kToppSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kToppSampling, opt);
    } else if (opt.do_sample && opt.top_k) {
        // fail to top k sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kTopkSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kTopkSampling, opt);
    }

    for (int step = 0; step < opt.max_new_tokens; ++step) {
        auto _out = (*this)({input_ids});
        if (_out[0].backend()->type() != MLLM_CPU) {
            _out[0].cpu();
        }
        auto out_token = text_generator_->generate(_out[0]);
        if (!call_back(out_token)) break;
        chatPostProcessing(out_token, input_ids, {});
    }
}

/*
vector<unsigned> Module::generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token) {
    auto chatPostProcessing = [](unsigned token_idx, Tensor &tokens_tensor, const vector<Tensor *> &clean_tensors) {
        tokens_tensor.reshape(1, 1, 1, 1);
        tokens_tensor.alloc();
        tokens_tensor.setDataAt<float>(0, 0, 0, 0, token_idx);

        for (auto tensor : clean_tensors) {
            tensor->reshape(0, 0, 0, 0);
            tensor->alloc();
        }
    };

    if (!opt.do_sample) {
        // fail to greedy search
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearch, opt);
    } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.F) {
        // fail to top p sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kToppSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kToppSampling, opt);
    } else if (opt.do_sample && opt.top_k) {
        // fail to top k sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kTopkSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kTopkSampling, opt);
    }
    vector<unsigned> result;
    for (int step = 0; step < opt.max_new_tokens; ++step) {
        auto _out = (*this)({input_ids});
        auto out_token = text_generator_->generate(_out[0]);
        result.push_back(out_token);
        if (end_token != -1 && out_token == end_token) break;
        chatPostProcessing(out_token, input_ids, {});
    }
    return result;
}
*/
/**
 * @brief 使用模型生成文本序列，支持批处理输入。
 * @param input_ids 输入的 token ID 张量，形状应为 [batch_size, 1, seq_len, 1]。
 * @param opt 生成选项，如最大新 token 数。
 * @param end_token 序列生成的结束符 ID。
 * @return 一个包含多个生成序列的向量，每个子向量是一个完整的 token ID 序列。
 */
vector<vector<unsigned>> Module::generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token) {
    auto chatPostProcessing = [](vector<unsigned> token_idxs, Tensor &tokens_tensor, const vector<Tensor *> &clean_tensors) {
        tokens_tensor.reshape(token_idxs.size(), 1, 1, 1);
        tokens_tensor.alloc();
        for (size_t idx = 0; idx < token_idxs.size(); ++idx) {
            unsigned int token_idx = token_idxs[idx];
            tokens_tensor.setDataAt<float>(idx, 0, 0, 0, token_idx);
        }
        for (auto tensor : clean_tensors) {
            tensor->reshape(0, 0, 0, 0);
            tensor->alloc();
        }
    };

    if (!opt.do_sample) {
        // fail to greedy search
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearch, opt);
    } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.F) {
        // fail to top p sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kToppSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kToppSampling, opt);
    } else if (opt.do_sample && opt.top_k) {
        // fail to top k sampling
        if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kTopkSampling)
            text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kTopkSampling, opt);
    }
    auto batch_size = input_ids.batch();
    vector<vector<unsigned>> results(batch_size);
    vector<bool> is_end(batch_size, false);
    for (int step = 0; step < opt.max_new_tokens; ++step) {
        auto _out = (*this)({input_ids});
        // _out[0].saveData<float>();
        // exit(1);
        vector<unsigned> out_tokens;
        for (int batch_ = 0; batch_ < batch_size; ++batch_) {
            Tensor _outt(1, 1, _out[0].sequence(), _out[0].dimension(), MLLM_CPU, true);
            memcpy(_outt.hostPtr<float>(), _out[0].ptrAt<float>(batch_, 0, 0, 0), _outt.cntSize());
            auto out_token = text_generator_->generate(_outt);
            if (end_token != -1 && out_token == end_token) {
                // std::cout << "End batch_: " << batch_ << std::endl;
                is_end[batch_] = true; // 标记该 batch 已经结束
                // out_tokens.push_back(0);
                // continue;
            }
            if (!is_end[batch_]) {
                out_tokens.push_back(out_token);
                results[batch_].push_back(out_token);
            } else {
                out_tokens.push_back(0); // 如果该 batch 已经结束，则填充
            }
        }
        chatPostProcessing(out_tokens, input_ids, {});
        if (std::all_of(is_end.begin(), is_end.end(), [](bool v) { return v; })) {
            // std::cout << "All batches ended." << std::endl;
            break; // 如果所有 batch 都结束，则退出循环
        }
    }
    return results;
}

} // namespace mllm
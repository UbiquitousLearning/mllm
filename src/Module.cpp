//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#include "Module.hpp"
#include "Types.hpp"
#include <stack>

namespace mllm {

// AbstructLoader *Module::loader;
// TensorStatus Tensor::tensor_status;
// bool Module::doLoad = false;
// The llm_model_ptr is a pointer to the outmost module
Module *Module::llm_model_ptr;

bool Module::isMultiChunkPrefilling = false;
bool Module::isFirstChunk = true;

int Module::listIdx;
std::stack<int> Module::listIdxStack;
// int Module::runlistIdx;
// TensorStatus Tensor::tensor_status;
BackendType Module::tmp_device = MLLM_CPU;
std::unordered_map<string, shared_ptr<Op>> Module::tensor_func_ops;

int Module::graphIdx = 0;

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
        std::cout << "  Prefilling speed: " << prefile_speed << " tokens/s" << std::endl;
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
    last_shape_bshd_.clear();

    return output;
}

void Module::generate(
    Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back) {
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

    for (int step = 0; step < opt.max_new_tokens; ++step) {
        auto _out = (*this)({input_ids});
        auto out_token = text_generator_->generate(_out[0]);
        if (!call_back(out_token)) break;
        chatPostProcessing(out_token, input_ids, {});
    }
}
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
} // namespace mllm
//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#ifndef MODULE_HPP
#define MODULE_HPP
#include "Generate.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Timing.hpp"
#include "backends/cpu/CPUBackend.hpp"

#include <any>
#include <functional>
#include <iostream>
#include <memory/SystemMemoryManager.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace mllm {

class Module {
private:
    double load_time_;
    int prefilling_token_size_ = 0;
    int decoding_token_size_ = 0;
    vector<double> inference_times_;
    vector<vector<int>> last_shape_bshd_;
    std::shared_ptr<LlmTextGenerator> text_generator_ = nullptr;

public:
    static map<BackendType, Backend *> backends;
    static AbstructLoader *loader;
    // static TensorStatus tensor_status;
    static bool doLoad;

    Module() = default;
    virtual ~Module() = default;

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        if (Module::backends.find(type) == Module::backends.end()) {
            switch (type) {
            case BackendType::MLLM_CPU: {
                shared_ptr<MemoryManager> mm = nullptr;
                mm = std::make_shared<SystemMemoryManager>();
                backends[MLLM_CPU] = new CPUBackend(mm);
                break;
            }
            default: {
            }
            }
        }
    }
    void to(BackendType type) {
        initBackend(type);
    }
    static void initLoader(string path) {
        loader = new ParamLoader(std::move(path));
    }

    void load(string path) {
        Tensor::graphs.clear();
        Tensor::tensor_status = TENSOR_STATIC_INIT;

        mllm_time_init();
        initLoader(path);
        Module::doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor::graphs["input" + std::to_string(i)] = std::make_shared<Tensor>(Module::backends[MLLM_CPU]);
            Tensor::graphs["input" + std::to_string(i)]->setName("input" + std::to_string(i));
            tmps.push_back(*Tensor::graphs["input" + std::to_string(i)]);
        }
        vector<std::any> alternate_args = {
            {},
            vector<int>{0, 0},
            std::vector<std::vector<int>>(32, std::vector<int>(2))};
        uint64_t time_start = 0;
        for (auto args : alternate_args) {
            time_start = mllm_time_us();
            try {
                operator()(tmps, args);
                break;
            } catch (const std::exception &e) {
#if not defined(__ARM_NEON)
                if (std::string("bad any_cast") != e.what()) {
                    std::cerr << e.what() << std::endl;
                    exit(0);
                }
#endif
            } catch (...) {
                std::cerr << "load error" << std::endl;
                exit(0);
            }
        }
        uint64_t time_end = mllm_time_us();
        load_time_ = (time_end - time_start) / 1000.0F; // ms
        Module::doLoad = false;
        // Tensor::graphs.clear();
    }

    void load(AbstructLoader &param_loader) {
        Tensor::graphs.clear();
        Tensor::tensor_status = TENSOR_STATIC_INIT;

        loader = &param_loader;
        Module::doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor::graphs["input" + std::to_string(i)] = std::make_shared<Tensor>(Module::backends[MLLM_CPU]);
            Tensor::graphs["input" + std::to_string(i)]->setName("input" + std::to_string(i));
            tmps.push_back(*Tensor::graphs["input" + std::to_string(i)]);
        }
        vector<int> tmpt = {0, 0};
        operator()(tmps, tmpt);
        Module::doLoad = false;
        // Tensor::graphs.clear();
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) = 0;

    template <typename... Args>
    vector<std::any> convertArgsToAnyVector(Args... args) {
        return vector<std::any>{std::any(args)...};
    }
    template <typename... Args>
    vector<Tensor> operator()(vector<Tensor> inputs, Args... args) {
        vector<std::any> anyArgs = convertArgsToAnyVector(args...);
        if (doLoad) {
            return Forward(inputs, anyArgs);
        }
        if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
            if (prefilling_token_size_ == 0) { // first time init
                // if(!Tensor::graphs.empty()){
                //     Tensor::graphs.clear();
                // }
                prefilling_token_size_ = inputs[0].sequence();
            } else if (decoding_token_size_ == 0) {
                decoding_token_size_ = inputs[0].sequence();
            }
            bool need_setup = true;
            for (int i = 0; i < inputs.size(); i++) {
                auto &input = inputs[i];
                input.setName("input" + std::to_string(i));
                input.setTtype(TensorType::NORMAL_TENSOR);
                Tensor::graphs[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
                if (inputs[0].sequence() != 1 && !last_shape_bshd_.empty()) {
                    // if LLM/VLLM model, the `need_setup` should be `true`
                    if (input.batch() == last_shape_bshd_[i][0] & input.sequence() == last_shape_bshd_[i][1] & input.head() == last_shape_bshd_[i][2] & input.dimension() == last_shape_bshd_[i][3]) {
                        need_setup = false;
                    }
                }
            }
            Tensor::tensor_status = TENSOR_STATIC_INIT;

            uint64_t time_start = mllm_time_us();
            if (need_setup) {
                Forward(inputs, anyArgs);
            }
            Tensor::tensor_status = TENSOR_STATIC_READY;
            // uint64_t time_start = mllm_time_us();
            auto output = Forward(inputs, anyArgs);
            uint64_t time_end = mllm_time_us();

            double inference_time_ = (time_end - time_start) / 1000.0F; // ms
            inference_times_.push_back(inference_time_);
            last_shape_bshd_.clear();
            for (auto &input : inputs) {
                last_shape_bshd_.push_back({input.batch(), input.sequence(),
                                            input.head(), input.dimension()});
            }

            return output;
        } else {
            return Forward(inputs, anyArgs);
        }
    }

    static int listIdx;
    static int runlistIdx;

    template <typename T>
    static vector<T> List(int n) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            modules.push_back(T());
            listIdx++;
        }
        listIdx = 0;
        return modules;
    }

    // 递归终止函数
    template <typename T>
    static auto change_last(T value) {
        return std::make_tuple(value + std::to_string(listIdx) + ".");
    }
    // 递归函数
    template <typename T, typename... Args>
    static auto change_last(T head, Args... tail) {
        auto tail_tuple = change_last(tail...);
        return std::tuple_cat(std::make_tuple(head), tail_tuple);
    }
    template <typename T, typename... Args>
    static vector<T> List(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...); // 创建新的参数包，最后一个参数被修改为原来的值+ std::to_string(listIdx)+ "."
            modules.push_back(std::move(T(std::apply([&](auto &&...args) { return T(std::forward<decltype(args)>(args)...); }, new_args))));
            listIdx++;
        }
        listIdx = 0;
        return modules;
    }

    void free() {
        Tensor::graphs.clear();
    }

    void profiling(string name = "") {
        // printf("\n");
        std::cout << "===========================================" << std::endl;
        if (name != "") {
            std::cout << "            " << name << std::endl;
            std::cout << "-------------------------------------------" << std::endl;
        }
        std::cout << "  Load time: " << load_time_ / 1000.0F << " s" << std::endl;
        if (inference_times_.size() > 1 && decoding_token_size_ != prefilling_token_size_) {
            std::cout << "  Prefilling speed: " << 1000 * prefilling_token_size_ / inference_times_[0] << " tokens/s" << std::endl;
            double sum_decoding_time = std::accumulate(std::begin(inference_times_) + 1, std::end(inference_times_), 0.0);
            double mean_decoding_time = sum_decoding_time / (inference_times_.size() - 1);
            std::cout << "  Decoding speed: " << 1000 / mean_decoding_time << " tokens/s" << std::endl;
        } else {
            double sum_time = std::accumulate(std::begin(inference_times_), std::end(inference_times_), 0.0);
            double mean_time = sum_time / (inference_times_.size());
            std::cout << "  Inference latency: " << mean_time / 1000.0F << " s" << std::endl;
        }
        // double sum_time = std::accumulate(std::begin(inference_times_), std::end(inference_times_), 0.0);
        // std::cout<<sum_time<< " - "<<Tensor::forward_times<<" = "<<sum_time-Tensor::forward_times<<std::endl;
        // std::cout<<Tensor::forward_times<< " - "<<Tensor::forward_times_2<<" = "<<Tensor::forward_times-Tensor::forward_times_2<<std::endl;

        std::cout << "===========================================" << std::endl;

        prefilling_token_size_ = 0;
        decoding_token_size_ = 0;
        inference_times_.clear();
        last_shape_bshd_.clear();
    }

    virtual void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back = [](unsigned int) -> bool { return true; }) {
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
        } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.f) {
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
};

} // namespace mllm

#endif // MODULE_HPP

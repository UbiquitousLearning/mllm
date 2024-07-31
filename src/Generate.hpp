/**
 * @file Generate.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief The Mllm Generator
 * @version 0.1
 * @date 2024-07-30
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#ifndef MLLM_GENERATE_HPP
#define MLLM_GENERATE_HPP
#include <cstdint>
#include <cassert>
#include <vector>
#include <random>
#include <utility>
#include "Tensor.hpp"

namespace mllm {

struct LlmTextGeneratorOpts {
    size_t max_new_tokens = 100;
    size_t min_new_tokens = 10;
    bool do_sample = true;
    float temperature = 0.7;
    int top_k = 5;
    float top_p = 0.92;
};

template <typename T>
T _sample_element(const std::vector<T> &elements, const std::vector<float> &probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    size_t index = dist(gen);
    return elements[index];
}

enum class LLmTextGeneratorType : int32_t {
    kNone = 0,
    kGreedySearch,
    kTopkSampling,
    kToppSampling,
    KLast,
};

class _LlmTextGenerateMethod {
public:
    virtual ~_LlmTextGenerateMethod() = default;
    virtual unsigned int generate(Tensor &t) = 0;
    inline void _tensor_to_vec(Tensor &t, std::vector<float> &scores) {
        assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        int _dims = t.dimension();
        int _seq = t.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = t.dataAt<float>(0, 0, _seq, i);
            scores.push_back(value);
        }
    }

    inline void _tensor_to_vec_with_idx(Tensor &t, std::vector<std::pair<float, unsigned int>> &scores) {
        assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        int _dims = t.dimension();
        int _seq = t.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = t.dataAt<float>(0, 0, _seq, i);
            scores.push_back(std::make_pair(value, i));
        }
    }
};

class _LlmTextGenerateGreedySearchMethod : public _LlmTextGenerateMethod {
public:
    _LlmTextGenerateGreedySearchMethod() = default;
    ~_LlmTextGenerateGreedySearchMethod() = default;
    unsigned int generate(Tensor &t) override;
};

class _LlmTextGenerateTopkSamplingMethod : public _LlmTextGenerateMethod {
public:
    ~_LlmTextGenerateTopkSamplingMethod() = default;
    _LlmTextGenerateTopkSamplingMethod(int32_t k = 5, float temperature = 0.f) :
        m_k(k),
        m_temperature(temperature) {
    }
    unsigned int generate(Tensor &t) override;

private:
    int32_t m_k;
    float m_temperature = 0.f;
};

class _LlmTextGenerateToppSamplingMethod : public _LlmTextGenerateMethod {
public:
    ~_LlmTextGenerateToppSamplingMethod() = default;
    _LlmTextGenerateToppSamplingMethod(float p = 5, float temperature = 0.f) :
        m_p(p),
        m_temperature(temperature) {
    }
    unsigned int generate(Tensor &t) override;

private:
    float m_p;
    float m_temperature = 0.f;
};

// Usage:
// LlmTextGeneratorOpts opt{
//     .max_new_tokens = 100,
//     .do_sample = true,
//     .temperature = 0.7f,
//     .top_k = 50,
//     .top_p = 0.f,
// };
// model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
//     auto out_string = tokenizer.detokenize({out_token});
//     auto [isOk, print_string] = processOutput(out_string);
//     if (isOk) {
//         std::cout << print_string << std::flush;
//     } else {
//         return false;
//     }
//     return true;
// });
// printf("\n");

class LlmTextGenerator {
public:
    ~LlmTextGenerator() {
        delete m_method_class;
    }

    LlmTextGenerator(const LLmTextGeneratorType &type, const LlmTextGeneratorOpts &opt) :
        m_type(type) {
        switch (type) {
        case LLmTextGeneratorType::kGreedySearch: m_method_class = new _LlmTextGenerateGreedySearchMethod(); break;
        case LLmTextGeneratorType::kTopkSampling: m_method_class = new _LlmTextGenerateTopkSamplingMethod(opt.top_k, opt.temperature); break;
        case LLmTextGeneratorType::kToppSampling: m_method_class = new _LlmTextGenerateToppSamplingMethod(opt.top_p, opt.temperature); break;
        default:
            assert(false && "NIY");
            break;
        }
    }

    inline unsigned int generate(Tensor &t) {
        return m_method_class->generate(t);
    }

    inline LLmTextGeneratorType type() {
        return m_type;
    }

private:
    LLmTextGeneratorType m_type;
    _LlmTextGenerateMethod *m_method_class = nullptr;
};

} // namespace mllm

#endif //! MLLM_GENERATE_HPP
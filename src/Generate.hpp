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
#include "Draft.hpp"

namespace mllm {

struct LlmTextGeneratorOpts {
    size_t max_new_tokens = 100;
    size_t min_new_tokens = 10;
    bool do_sample = true;
    float temperature = 0.7;
    int top_k = 5;
    float top_p = 0.92;
    bool is_padding = false;
    int seq_before_padding = 0;
    int chunk_size = -1;
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
    kGreedySearchForSD,
};

class _LlmTextGenerateMethod {
protected:
    bool is_padding = false;
    int seq_before_padding = 0;
    int chunk_size = -1;

public:
    virtual ~_LlmTextGenerateMethod() = default;
    virtual unsigned int generate(Tensor &t) = 0;
    inline void setPadding(bool is_padding, int seq_before_padding, int chunk_size) {
        this->is_padding = is_padding;
        this->seq_before_padding = seq_before_padding;
        this->chunk_size = chunk_size;
    }
    inline void _tensor_to_vec(Tensor &t, std::vector<float> &scores) {
        assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        int _dims = t.dimension();
        int _seq = t.sequence() - 1;
        // padding prefill for QNN
        if (is_padding) {
            if (chunk_size > 0) {
                _seq = (seq_before_padding - 1) % chunk_size;
            } else {
                _seq = seq_before_padding - 1;
            }
        }
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

    inline void _tensor_to_multivec(Tensor &t, std::vector<std::vector<float>> &scores) {
        assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        int _dims = t.dimension();
        int n_seq = t.sequence();
        // TODO: 考虑QNN进行padding
        // padding prefill for QNN
        // if (is_padding) {
        //     if (chunk_size > 0) {
        //         _seq = (seq_before_padding - 1) % chunk_size;
        //     } else {
        //         _seq = seq_before_padding - 1;
        //     }
        // }
        for (int s = 0; s < n_seq; ++s) {
            std::vector<float> values(t.dimension());
            for (int i = 0; i < _dims; ++i) {
                auto value = t.dataAt<float>(0, 0, s, i);
                values[i] = value;
            }
            scores.push_back(values);
        }
    }

    inline void _tensor_to_multivec_with_idx(Tensor &t, std::vector<std::vector<std::pair<float, unsigned int>>> &scores) {
        assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        int _dims = t.dimension();
        int n_seq = t.sequence();
        // TODO: 考虑QNN进行padding
        // padding prefill for QNN
        // if (is_padding) {
        //     if (chunk_size > 0) {
        //         _seq = (seq_before_padding - 1) % chunk_size;
        //     } else {
        //         _seq = seq_before_padding - 1;
        //     }
        // }
        for (int s = 0; s < n_seq; ++s) {
            std::vector<std::pair<float, unsigned int>> values(t.dimension());
            for (int i = 0; i < _dims; ++i) {
                auto value = t.dataAt<float>(0, 0, s, i);
                values[i] = std::make_pair(value, i);
            }
            scores.push_back(values);
        }
    }
};

class _LlmTextGenerateGreedySearchMethod : public _LlmTextGenerateMethod {
public:
    _LlmTextGenerateGreedySearchMethod() = default;
    ~_LlmTextGenerateGreedySearchMethod() = default;
    unsigned int generate(Tensor &t) override;
};

class _LlmTextGenerateGreedySearchMethodForSD : public _LlmTextGenerateMethod {
    public:
        _LlmTextGenerateGreedySearchMethodForSD() = default;
        ~_LlmTextGenerateGreedySearchMethodForSD() = default;
        inline void _tensor_to_vec_of_multiIndices(Tensor &t, std::vector<std::vector<float>> &scores, std::vector<int> indices) {
            assert(t.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
            assert(t.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
            int _dims = t.dimension();
            // TODO: 考虑QNN进行padding
            // padding prefill for QNN
            // if (is_padding) {
            //     if (chunk_size > 0) {
            //         _seq = (seq_before_padding - 1) % chunk_size;
            //     } else {
            //         _seq = seq_before_padding - 1;
            //     }
            // }
            for (int idx = 0; idx < indices.size(); ++idx) {
                std::vector<float> values(t.dimension());
                int _seq = indices[idx];
                for (int i = 0; i < _dims; ++i) {
                    auto value = t.dataAt<float>(0, 0, _seq, i);
                    values[i] = value;
                }
                scores.push_back(values);
            }
        }
        unsigned int generate(Tensor &t) override {
            std::cerr << "Should use generate_SD" << std::endl;
            assert(false);
            return -1;
        };
        unsigned int generate_SD(Tensor &t, TracePool &tp);
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

class _LlmTextGenerateNucleusSamplingMethodForSD : public _LlmTextGenerateMethod {
public:
    _LlmTextGenerateNucleusSamplingMethodForSD(int k, float p, float temp) : samplingConfig(SamplingConfig(temp, p, k)) {}
    ~_LlmTextGenerateNucleusSamplingMethodForSD() = default;

    unsigned int generate(Tensor &t) override {
        std::cerr << "Should use generate_SD" << std::endl;
        assert(false);
        return -1;
    };
    unsigned int generate_SD(Tensor &t, TracePool &tp);
    std::vector<unsigned int> evalPosterior(const std::vector<std::vector<float>> &logit_scores, const std::vector<unsigned int> &sampled_token_ids, TracePool &tp);
private:
    float temperature = 1.0;
    float top_p = 1.0;
    int top_k = -1;
    struct SamplingConfig {
        float temperature = 1.0;
        float top_p = 1.0;
        int top_k = -1;
        SamplingConfig(float _temperature, float _top_p, float _top_k): temperature(_temperature), top_p(_top_p), top_k(_top_k) {}
    } samplingConfig;

    void apply_logits_processor(std::vector<std::pair<float, unsigned int>>& logits_with_indices, const SamplingConfig& config) {
        const size_t vocab_size = logits_with_indices.size();
        if (vocab_size == 0) return;

        // 温度调整
        if (config.temperature > 0 && config.temperature != 1.0f) {
            const float inv_temp = 1.0f / config.temperature;
            for (auto& v : logits_with_indices) v.first *= inv_temp;
        }

        // Top-k处理
        if (config.top_k > 0 && config.top_k < (int)vocab_size) {
            // 部分排序找出topk
            std::partial_sort(
                logits_with_indices.begin(),
                logits_with_indices.begin() + config.top_k,
                logits_with_indices.end()
            );

            // 构建屏蔽掩码
            std::vector<bool> mask(vocab_size, false);
            for (int i=0; i<config.top_k; ++i) {
                mask[logits_with_indices[i].second] = true;
            }
            
            // 应用掩码
            for (size_t i=0; i<vocab_size; ++i) {
                if (!mask[i]) logits_with_indices[i].first = -INFINITY;
            }
        }
        // Top-p处理
        else if (config.top_p > 0.0f && config.top_p < 1.0f) {
            // 计算softmax
            std::vector<float> probs(vocab_size);
            std::pair<float, unsigned int> max_logit_with_index = *std::max_element(logits_with_indices.begin(), logits_with_indices.end(),
                [](std::pair<float, unsigned int> a, std::pair<float, unsigned int> b) { return a.first > b.first; });
            float max_logit = max_logit_with_index.first;
            float sum_exp = 0.0f;
            for (size_t i=0; i<vocab_size; ++i) {
                probs[i] = std::exp(logits_with_indices[i].first - max_logit);
                sum_exp += probs[i];
            }
            for (float& p : probs) p /= sum_exp;

            // 带索引排序
            std::vector<std::pair<float, unsigned int>> sorted_probs(vocab_size);
            for (size_t i=0; i<vocab_size; ++i) {
                sorted_probs[i] = {probs[i], i};
            }
            std::sort(sorted_probs.begin(), sorted_probs.end(), [](std::pair<float, unsigned int> a, std::pair<float, unsigned int> b) { return a.first > b.first; });

            // 计算累积概率
            float cumulative = 0.0f;
            size_t cutoff = 0;
            for (; cutoff < vocab_size; ++cutoff) {
                cumulative += sorted_probs[cutoff].first;
                if (cumulative > config.top_p) break;
            }
            cutoff = std::min(cutoff+1, vocab_size-1);

            // 构建有效集合
            std::vector<bool> valid(vocab_size, false);
            for (size_t i=0; i<cutoff; ++i) {
                valid[sorted_probs[i].second] = true;
            }

            // 应用过滤
            for (size_t i=0; i<vocab_size; ++i) {
                if (!valid[i]) logits_with_indices[i].first = -INFINITY;
            }
        }
    }
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
        case LLmTextGeneratorType::kGreedySearchForSD: m_method_class = new _LlmTextGenerateGreedySearchMethodForSD(); break;
        case LLmTextGeneratorType::kTopkSampling: m_method_class = new _LlmTextGenerateTopkSamplingMethod(opt.top_k, opt.temperature); break;
        case LLmTextGeneratorType::kToppSampling: m_method_class = new _LlmTextGenerateToppSamplingMethod(opt.top_p, opt.temperature); break;
        default:
            assert(false && "NIY");
            break;
        }

        // padding prefill for QNN
        if (opt.is_padding) {
            m_method_class->setPadding(opt.is_padding, opt.seq_before_padding, opt.chunk_size);
        }
    }

    inline unsigned int generate(Tensor &t) {
        return m_method_class->generate(t);
    }

    inline unsigned int generate_SD(Tensor &t, TracePool &tp) {
        // 检查m_method_class类型是_LlmTextGenerateGreedySearchMethodForSD然后调用，否则报错
        if (m_type != LLmTextGeneratorType::kGreedySearchForSD) {
            std::cerr << "Should use generate_SD in _LlmTextGenerateGreedySearchMethodForSD only" << std::endl;
            assert(false);
            return -1;
        }
        return dynamic_cast<_LlmTextGenerateGreedySearchMethodForSD*>(m_method_class)->generate_SD(t, tp);
    };

    inline unsigned int generate(Tensor &t, const LlmTextGeneratorOpts &opt) {
        if (opt.is_padding) {
            m_method_class->setPadding(opt.is_padding, opt.seq_before_padding, opt.chunk_size);
        }
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
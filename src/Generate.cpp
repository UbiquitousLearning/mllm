/**
 * @file Generate.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief The Mllm Generator Impl
 * @version 0.1
 * @date 2024-07-30
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "Generate.hpp"
#include <algorithm>
#include <numeric>

namespace mllm {

unsigned int _LlmTextGenerateGreedySearchMethod::generate(Tensor &t) {
    std::vector<float> scores;
    this->_tensor_to_vec(t, scores);
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

unsigned int _LlmTextGenerateTopkSamplingMethod::generate(Tensor &t) {
    auto argmax = [](const std::vector<float> &vec) -> unsigned int {
        return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
    };

    if (m_k == 0 || m_k == 1) {
        std::vector<float> scores;
        this->_tensor_to_vec(t, scores);
        return argmax(scores);
    }

    std::vector<std::pair<float, unsigned int>> scores;
    this->_tensor_to_vec_with_idx(t, scores);

    // find top k
    std::partial_sort(scores.begin(), scores.begin() + m_k, scores.end(),
                      [](std::pair<float, unsigned int> a, std::pair<float, unsigned int> b) { return a.first > b.first; });
    std::vector<float> top_k_elements(m_k, 0.f);
    std::vector<unsigned int> top_k_elements_idx(m_k, 0);
    for (int i = 0; i < m_k; ++i) {
        top_k_elements[i] = scores[i].first;
        top_k_elements_idx[i] = scores[i].second;
    }

    // softmax with temperature
    std::vector<float> softmax(top_k_elements.size(), 0.f);
    double max_logit = top_k_elements[argmax(top_k_elements)];
    double sum_exp = 0.f;

    for (size_t i = 0; i < top_k_elements.size(); ++i) {
        softmax[i] = exp((top_k_elements[i] - max_logit) / m_temperature);
        sum_exp += softmax[i];
    }

    for (float &value : softmax) {
        value /= sum_exp;
    }

    // sampling
    float _sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);
    for (float &value : softmax) {
        value /= _sum;
    }

    auto idx = _sample_element(top_k_elements_idx, softmax);
    return idx;
}

unsigned int _LlmTextGenerateToppSamplingMethod::generate(Tensor &t) {
    auto argmax = [](const std::vector<float> &vec) -> unsigned int {
        return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
    };
    std::vector<std::pair<float, unsigned int>> scores;
    this->_tensor_to_vec_with_idx(t, scores);

    std::sort(scores.begin(), scores.end(), [](std::pair<float, unsigned int> a, std::pair<float, unsigned int> b) { return a.first > b.first; });
    std::vector<float> top_k_elements;
    std::vector<unsigned int> top_k_elements_idx;

    if (scores[0].first > 1.f) {
        throw std::runtime_error("The input tensor t should go through softmax first.(0.f - 1.f is acceptable)");
    }

    float p = 0.f;
    size_t idx = 0;
    while (p < m_p) {
        top_k_elements.emplace_back(scores[idx].first);
        top_k_elements_idx.emplace_back(scores[idx].second);
        p += scores[idx].first;
        idx++;
    }

    if (top_k_elements.size() == 1) {
        return top_k_elements_idx[0];
    }

    // softmax with temperature
    std::vector<float> softmax(top_k_elements.size(), 0.f);
    double max_logit = top_k_elements[argmax(top_k_elements)];
    double sum_exp = 0.f;

    for (size_t i = 0; i < top_k_elements.size(); ++i) {
        softmax[i] = exp((top_k_elements[i] - max_logit) / m_temperature);
        sum_exp += softmax[i];
    }

    for (float &value : softmax) {
        value /= sum_exp;
    }

    // sampling
    float _sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);
    for (float &value : softmax) {
        value /= _sum;
    }

    auto ret = _sample_element(top_k_elements_idx, softmax);
    return ret;
}

} // namespace mllm
//
// Created by Xiang Li on 2023/12/21.
//

#ifndef HELPER_HPP
#define HELPER_HPP
#include <Net.hpp>
#include <iostream>
#include <Types.hpp>
#include <utility>
#include <valarray>
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Tokenizer.hpp"
using namespace  mllm;

inline void fullTensor(shared_ptr<Tensor> input_tensor, mllm::Net *net, vector<int> shape, float value) {
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    input_tensor->fullData<float>(value);
}

inline void token2Tensor(shared_ptr<Tensor> input_tensor, Net *net, vector<token_id_t> tokens) {
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, 1, static_cast<int>(tokens.size()), 1);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}
inline void patches2Tensor(shared_ptr<Tensor> input_tensor, Net *net, vector<vector<vector<float>>> image_patches) {
    if(image_patches.empty()) {
        fullTensor(input_tensor, net, {0, 0, 0, 0},1.0F);
        return;
    }
    const int batch = image_patches.size();
    const int seq =  image_patches[0].size();
    const int dims = image_patches[0][0].size();
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(batch, 1, seq, dims);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            for (int k = 0; k < dims; ++k) {
                input_tensor->setDataAt<float>(i, 0, j, k, image_patches[i][j][k]);
            }
        }
    }
}
inline void patchIdx2Tensor(shared_ptr<Tensor> input_tensor, Net *net, vector<vector<int>> image_patches_indices) {
    if(image_patches_indices.empty()) {
        fullTensor(input_tensor, net, {0, 0, 0, 0},1.0F);
        return;
    }
    const int batch = image_patches_indices.size();
    const int seq =  image_patches_indices[0].size();
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(batch, 1, seq, 1);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            input_tensor->setDataAt<float>(i, 0, j, 0, image_patches_indices[i][j]);
        }
    }
}

inline unsigned int argmax(const std::vector<float> &scores) {
    if (scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
#endif //HELPER_HPP

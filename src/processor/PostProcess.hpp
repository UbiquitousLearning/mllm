//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP
#include <Tensor.hpp>

namespace mllm {
inline void chatPostProcessing(unsigned token_idx, Tensor &tokens_tensor, const vector<Tensor *>& clean_tensors) {
    tokens_tensor.reshape(1, 1, 1, 1);
    tokens_tensor.alloc();
    tokens_tensor.setDataAt<float>(0, 0, 0, 0, token_idx);

    for (auto tensor : clean_tensors) {
        tensor->reshape(0, 0, 0, 0);
        tensor->alloc();
    }
}
}


#endif //POSTPROCESS_HPP



#include "CPUKVCache.hpp"

namespace mllm {

CPUKVCache::CPUKVCache(Backend *bn, string opName, bool multiThread) :
    Op(bn, opName) {
    cache_.setBackend(bn);
    cache_.setDtype(MLLM_TYPE_F32);
    cache_.reshape(0, 0, 0, 0);
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache  reshape" << std::endl;
    int a_dim = inputs[0]->dimension();
    int a_sen = inputs[0]->sequence();
    int b_dim = cache_.dimension();
    int b_sen = cache_.sequence();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), a_sen + b_sen, a_dim);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(ParamLoader &loader) {
    // std::cout<<name() << "  CPUKVCache load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache()" << std::endl;
    if (cache_.count() > 0) {
        int a_dim = cache_.dimension();
        int a_sen = cache_.sequence();
        int c_sen = outputs[0]->sequence();

        // Calculate size of the block to copy
        size_t block_size = a_dim * sizeof(float);

        for (int h = 0; h < cache_.head(); ++h) {
            for (int b = 0; b < cache_.batch(); ++b) {
                #pragma omp parallel for num_threads(8)
                for (int s = 0; s < c_sen; ++s) {
                    float *dest_ptr = outputs[0]->ptrAt<float>(b, h, s, 0);
                    if (s < a_sen) {
                        float *src_ptr = cache_.ptrAt<float>(b, h, s, 0);
                        memcpy(dest_ptr, src_ptr, block_size);
                    } else {
                        float *src_ptr = inputs[0]->ptrAt<float>(b, h, s - a_sen, 0);
                        memcpy(dest_ptr, src_ptr, block_size);
                    }
                }
            }
        }

    } else {
        outputs[0]->copyFrom(inputs[0]);
    }
    cache_.reshape(outputs[0]->shape());
    cache_.alloc();
    cache_.copyFrom(outputs[0]);
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache() free" << std::endl;
    return Op::free(inputs, outputs);
}
} // namespace mllm

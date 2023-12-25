

#include "CPUKVCache.hpp"
#include "ParamLoader.hpp"

namespace mllm {
/*
CPUKVCache::CPUKVCache(Backend *bn, string opName, bool isK, bool multiThread) :
    Op(bn, opName) {
    isK_ = isK;
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
    if (isK_)
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), a_sen + b_sen, a_dim);
    else
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), a_dim, a_sen + cache_.dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(AbstructLoader &loader) {
    // std::cout<<name() << "  CPUKVCache load" << std::endl;
    return Op::load(loader);
}

void tensor_trans(shared_ptr<Tensor> src, Tensor *dst) {
    for (int b = 0; b < src->batch(); b++) {
        for (int h = 0; h < src->head(); h++) {
#pragma omp parallel for num_threads(4)
            for (int n = 0; n < src->sequence(); n++) {
                for (int m = 0; m < src->dimension(); m++) {
                    dst->setDataAt<float>({b, h, m, n}, src->dataAt<float>({b, h, n, m}));
                }
            }
        }
    }
}
ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache()" << std::endl;
    if (cache_.count() > 0) {
        if (isK_) {
            int a_dim = cache_.dimension();
            int a_sen = cache_.sequence();
            int c_sen = outputs[0]->sequence();

            // Calculate size of the block to copy
            size_t block_size = a_dim * sizeof(float);

            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = 0; h < cache_.head(); ++h) {
                    #pragma omp parallel for num_threads(4)
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
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = 0; h < cache_.head(); ++h) {
                    #pragma omp parallel for num_threads(4)
                    for (int s = 0; s < outputs[0]->sequence(); ++s) {
                        float *output_ptr = outputs[0]->ptrAt<float>({b, h, s, 0});
                        if (cache_.dimension() > 0) {
                            float *cache_ptr = cache_.ptrAt<float>({b, h, s, 0});
                            memcpy(output_ptr, cache_ptr, cache_.dimension() * sizeof(float));
                        }
                        if (cache_.dimension() < outputs[0]->dimension()) {
                            float *input_ptr = inputs[0]->ptrAt<float>({b, h, 0, s});
                            memcpy(output_ptr + cache_.dimension(), input_ptr, (outputs[0]->dimension() - cache_.dimension()) * sizeof(float));
                        }
                    }
                }
            }
        }
    } else {
        if (isK_)
            outputs[0]->copyFrom(inputs[0]);
        else
            tensor_trans(inputs[0], outputs[0].get());
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
*/

CPUKVCache::CPUKVCache(Backend *bn, string opName, bool isK, bool multiThread) :
    Op(bn, opName) {
    isK_ = isK;
    cache_.setBackend(bn);
    cache_.setDtype(MLLM_TYPE_F16);
    cache_limit_ = 500;
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    if(cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head(), cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        cache_seq_len_ = 0;
    }

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + cache_seq_len_, inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(AbstructLoader &loader) {
    // std::cout<<name() << "  CPUKVCache load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache()" << std::endl;
    cache_seq_len_ += inputs[0]->sequence();
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache() free" << std::endl;
    return Op::free(inputs, outputs);
}


ErrorCode CPUKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->setDtype(cache_.dtype());
    outputs[0]->deepCopyFrom(cache_, {0,0,cache_seq_len_/cache_limit_,0});
    if (inputs[0]->masterTensor() ==nullptr) {
        inputs[0]->free();
    }
    inputs[0]->deepCopyFrom(cache_, {0,0,cache_seq_len_%cache_limit_,0});
#ifdef DEBUG
    std::cout << "*"<<name()<<" setUp*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}
} // namespace mllm

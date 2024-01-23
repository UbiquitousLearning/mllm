

#include "CPUKVCache.hpp"
#include "ParamLoader.hpp"

namespace mllm {
CPUKVCache::CPUKVCache(Backend *bn, string opName, int n_rep, int cache_max, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    cache_.setBackend(bn);
    cache_.setDtype(MLLM_TYPE_F16);
    cache_limit_ = cache_max;
    n_rep_ = n_rep;
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if(cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head()*n_rep_, cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        cache_seq_len_ = 0;
    }

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head()*n_rep_, inputs[0]->sequence() + cache_seq_len_, inputs[0]->dimension());
    if(inputs[0]->sequence() + cache_seq_len_ >cache_limit_){
        std::cerr<<"\n[ERROR]: Current tokens exceed cache limit: "<<inputs[0]->sequence() + cache_seq_len_<<">"<<cache_limit_<<";";
        std::cerr<<"\n         Please set args `--limits` >"<<cache_limit_<<std::endl;

        exit(1);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head()*n_rep_, cache_limit_, inputs[0]->dimension());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(AbstructLoader &loader) {

    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    int cache_seq_len_old = cache_seq_len_;
    cache_seq_len_ += inputs[0]->sequence();
    if(n_rep_ >1) {
        if(cache_.ctype() == BSHD) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = inputs[0]->head()-1; h >= 0; --h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int seq = cache_seq_len_old; seq < cache_seq_len_; ++seq) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if(cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr = inputs[0]->ptrAt<float>(b, h, seq-cache_seq_len_old, 0);
                                auto dest_ptr = cache_.ptrAt<float>(b, cache_head, seq, 0);
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(float));
                            }else if(cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, seq-cache_seq_len_old, 0);
                                auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, cache_head, seq, 0);
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        }else if(cache_.ctype() == BHDS) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = inputs[0]->head() - 1; h >= 0; --h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if (cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr = inputs[0]->ptrAt<float>(b, h, 0, d);
                                auto dest_ptr = cache_.ptrAt<float>(b, cache_head, cache_seq_len_old, d);
                                int copy_size = cache_seq_len_ - cache_seq_len_old;
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(float));
                            } else if (cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, 0, d);
                                auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, cache_head, cache_seq_len_old, d);
                                int copy_size = cache_seq_len_ - cache_seq_len_old;
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        }else {
            std::cout<<"ERROR Ctype in KVCcache;"<<std::endl;
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    return Op::free(inputs, outputs);
}


ErrorCode CPUKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->setDtype(cache_.dtype());
    outputs[0]->deepCopyFrom(cache_, false, {0,0,cache_seq_len_/cache_limit_,0});
    if(inputs[0]->sequence() + cache_seq_len_ >cache_limit_) {
        outputs[0]->deepCopyFrom(cache_, false, {0,0,cache_seq_len_%cache_limit_ +1,0});
    }
    if (inputs[0]->masterTensor() ==nullptr) {
        inputs[0]->free();
    }
    inputs[0]->deepCopyFrom(cache_, false, {0,0,cache_seq_len_%cache_limit_,0});
    return MLLM_NO_ERROR;
}
} // namespace mllm

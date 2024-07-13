

#include "CPUKVCacheNPU.hpp"
#include "ParamLoader.hpp"
#include "Types.hpp"
#include <cstdint>

namespace mllm {
CPUKVCacheNPU::CPUKVCacheNPU(Backend *bn, string opName, int n_rep, int cache_max, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    cache_.setBackend(bn);

    // TODO: Chaning it to FP16
    cache_.setDtype(MLLM_TYPE_F32);
    cache_limit_ = cache_max;
}

ErrorCode CPUKVCacheNPU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head(), cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        cache_seq_len_ = 0;
    }

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + cache_seq_len_, inputs[0]->dimension());

    if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
        std::cerr << "\n[ERROR]: Current tokens exceed cache limit: " << inputs[0]->sequence() + cache_seq_len_ << ">" << cache_limit_ << ";";
        std::cerr << "\n         Please set args `--limits` >" << cache_limit_ << std::endl;

        exit(1);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_, inputs[0]->dimension());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUKVCacheNPU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (cache_.ctype() == BSHD && inputs[0]->ctype() == BSHD) { // 'K'
        std::cout << "NOTE: BSHD input, use deepCopy" << std::endl;
// #pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < cache_.batch(); ++b) {
            for (int h = 0; h < inputs[0]->head(); ++h) {
                for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
                    if (cache_.dtype() == MLLM_TYPE_F32) {
                        auto src_ptr = inputs[0]->ptrAt<float>(b, h, seq, 0);
                        auto dest_ptr = cache_.ptrAt<float>(b, h, cache_seq_len_ + seq, 0);
                        memcpy(dest_ptr, src_ptr, cache_.dimension() * sizeof(float));
                    } else if (cache_.dtype() == MLLM_TYPE_F16) {
                        auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, seq, 0);
                        auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, h, cache_seq_len_ + seq, 0);
                        memcpy(dest_ptr, src_ptr, cache_.dimension() * sizeof(mllm_fp16_t));
                    } else if (cache_.dtype() == MLLM_TYPE_I8) {
                        auto src_ptr = inputs[0]->ptrAt<int8_t>(b, h, seq, 0);
                        auto dest_ptr = cache_.ptrAt<int8_t>(b, h, cache_seq_len_ + seq, 0);
                        memcpy(dest_ptr, src_ptr, cache_.dimension() * sizeof(int8_t));
                    }
                }
            }
        }
    } else if (cache_.ctype() == BHDS && inputs[0]->ctype() == BHDS) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < cache_.batch(); ++b) {
            for (int h = 0; h < inputs[0]->head(); ++h) {
                for (int d = 0; d < inputs[0]->dimension(); ++d) {
                    if (cache_.dtype() == MLLM_TYPE_F32) {
                        auto src_ptr = inputs[0]->ptrAt<float>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<float>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(float));
                    } else if (cache_.dtype() == MLLM_TYPE_F16) {
                        auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(mllm_fp16_t));
                    } else if (cache_.dtype() == MLLM_TYPE_I8) {
                        auto src_ptr = inputs[0]->ptrAt<int8_t>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<int8_t>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(int8_t));
                    }
                }
            }
        }
    } else { // 'V': the data of input is BHDS, but the ctype is BSHD
        std::cout << "WARNING: cache_ ctype is different from input ctype for " << name() << std::endl;
        std::cout << "input ctype: " << inputs[0]->ctype() << std::endl;
        std::cout << "cache_ ctype: " << cache_.ctype() << std::endl;

        inputs[0]->transShape(SEQUENCE, DIMENSION);
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < cache_.batch(); ++b) {
            for (int h = 0; h < inputs[0]->head(); ++h) {
                for (int d = 0; d < inputs[0]->dimension(); ++d) {
                    if (cache_.dtype() == MLLM_TYPE_F32) {
                        auto src_ptr = inputs[0]->ptrAt<float>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<float>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(float));
                    } else if (cache_.dtype() == MLLM_TYPE_F16) {
                        auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(mllm_fp16_t));
                    } else if (cache_.dtype() == MLLM_TYPE_I8) {
                        auto src_ptr = inputs[0]->ptrAt<int8_t>(b, h, 0, d);
                        auto dest_ptr = cache_.ptrAt<int8_t>(b, h, cache_seq_len_, d);
                        memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(int8_t));
                    }
                }
            }
        }
    }

    int cache_seq_len_old = cache_seq_len_;
    cache_seq_len_ += inputs[0]->sequence();

    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    // output setup
    outputs[0]->setDtype(cache_.dtype());
    outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ / cache_limit_, 0});
    if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
        outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_ + 1, 0});
    }

    std::cout << "cache ctype" << cache_.ctype() << std::endl;

    inputs[0]->setDtype(cache_.dtype());
    return MLLM_NO_ERROR;
}
} // namespace mllm


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
    cache_.setDtype(MLLM_TYPE_F16);
    cache_limit_ = cache_max;
    n_rep_ = n_rep;
}

ErrorCode CPUKVCacheNPU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        cache_seq_len_ = 0;

        // when using the old frontend, the V will be transposed here; while in the module API, the V will be transposed in the QNNTranspose
        if (name().find("v_cache") != std::string::npos && inputs[0]->ctype() != BHDS) {
            inputs[0]->transShape(SEQUENCE, DIMENSION);
        }
    }

#ifdef USE_QNN
    // when the execution is switched from pref to dec, the sequence length should be set to the no padding length
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching() && cpuBackend->getExecutionType() == AUTOREGRESSIVE) {
        cache_seq_len_ = cpuBackend->getSequenceLength();
        isDecoding = true;
    }
    // if a new prompt is given, the cache should be updated
    if (cpuBackend->isStageSwitching() && cpuBackend->getExecutionType() == PROMPT) {
        cache_seq_len_ = cpuBackend->getSequenceLength();
        isDecoding = false;
    }
#endif

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, inputs[0]->sequence() + cache_seq_len_, inputs[0]->dimension());

    if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
        MLLM_LOG_ERROR_STREAM << "\n[ERROR]: Current tokens exceed cache limit: " << inputs[0]->sequence() + cache_seq_len_ << ">" << cache_limit_ << ";" << "\n         Please set args `--limits` >" << cache_limit_ << std::endl;

        exit(1);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_, inputs[0]->dimension());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUKVCacheNPU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Group Query Attention
    if (n_rep_ > 1 && isDecoding) {
        int cache_seq_len_old = cache_seq_len_;
        cache_seq_len_ += inputs[0]->sequence();
        if (cache_.ctype() == BSHD) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = inputs[0]->head() - 1; h >= 0; --h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int seq = cache_seq_len_old; seq < cache_seq_len_; ++seq) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if (cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr =
                                    inputs[0]->ptrAt<float>(b, h, seq - cache_seq_len_old, 0);
                                auto dest_ptr = cache_.ptrAt<float>(b, cache_head, seq, 0);
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(float));
                            } else if (cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr =
                                    inputs[0]->ptrAt<mllm_fp16_t>(b, h, seq - cache_seq_len_old, 0);
                                auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, cache_head, seq, 0);
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        } else if (cache_.ctype() == BHDS) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = inputs[0]->head() - 1; h >= 0; --h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if (cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr = inputs[0]->ptrAt<float>(b, h, 0, d);
                                auto dest_ptr =
                                    cache_.ptrAt<float>(b, cache_head, cache_seq_len_old, d);
                                int copy_size = cache_seq_len_ - cache_seq_len_old;
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(float));
                            } else if (cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, 0, d);
                                auto dest_ptr =
                                    cache_.ptrAt<mllm_fp16_t>(b, cache_head, cache_seq_len_old, d);
                                int copy_size = cache_seq_len_ - cache_seq_len_old;
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        } else {
            std::cout << "ERROR Ctype in KVCcache;" << std::endl;
        }
        return MLLM_NO_ERROR;
    }
    if (n_rep_ > 1 && !isDecoding) {
        if (cache_.ctype() == BSHD) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = 0; h < inputs[0]->head(); ++h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if (cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr =
                                    inputs[0]->ptrAt<float>(b, h, seq, 0);
                                auto dest_ptr = cache_.ptrAt<float>(b, cache_head, cache_seq_len_ + seq, 0);
                                memcpy(dest_ptr, src_ptr, inputs[0]->dimension() * sizeof(float));
                            } else if (cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr =
                                    inputs[0]->ptrAt<mllm_fp16_t>(b, h, seq, 0);
                                auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, cache_head, cache_seq_len_ + seq, 0);
                                memcpy(dest_ptr, src_ptr, inputs[0]->dimension() * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        } else if (cache_.ctype() == BHDS) {
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = 0; h < inputs[0]->head(); ++h) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                            auto cache_head = h * n_rep_ + i_rep;
                            if (cache_.dtype() == MLLM_TYPE_F32) {
                                auto src_ptr = inputs[0]->ptrAt<float>(b, h, 0, d);
                                auto dest_ptr =
                                    cache_.ptrAt<float>(b, cache_head, cache_seq_len_, d);
                                memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(float));
                            } else if (cache_.dtype() == MLLM_TYPE_F16) {
                                auto src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(b, h, 0, d);
                                auto dest_ptr =
                                    cache_.ptrAt<mllm_fp16_t>(b, cache_head, cache_seq_len_, d);
                                memcpy(dest_ptr, src_ptr, inputs[0]->sequence() * sizeof(mllm_fp16_t));
                            }
                        }
                    }
                }
            }
        } else {
            std::cout << "ERROR Ctype in KVCcache;" << std::endl;
        }
        cache_seq_len_ += inputs[0]->sequence();
        return MLLM_NO_ERROR;
    }

    // q head == kv cache head
    // when decoding, the input will deepCopy from cache, no need to execute
    if (n_rep_ <= 1 && isDecoding) {
        cache_seq_len_ += inputs[0]->sequence();
        return MLLM_NO_ERROR;
    }

    if (cache_.ctype() == BSHD && inputs[0]->ctype() == BSHD) { // 'K'
#pragma omp parallel for collapse(3) num_threads(thread_count)
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
    } else if (cache_.ctype() == BHDS && inputs[0]->ctype() == BHDS) { // 'V'
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
    } else { // naive case
        std::cout << "ERROR Ctype in KVCcache;" << std::endl;
    }

    cache_seq_len_ += inputs[0]->sequence();

    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUKVCacheNPU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    // when decoding, the input will deepCopy from cache, no need to execute
    if (isDecoding) {
        outputs[0]->setDtype(cache_.dtype());
        outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ / cache_limit_, 0});
        if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
            outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_ + 1, 0});
        }
        if (inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        inputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_, 0});
        return MLLM_NO_ERROR;
    }

    // output setup
    outputs[0]->setDtype(cache_.dtype());
    outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ / cache_limit_, 0});
    if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
        outputs[0]->deepCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_ + 1, 0});
    }

    inputs[0]->setDtype(cache_.dtype());
    return MLLM_NO_ERROR;
}
} // namespace mllm


#include "CPUKVCache.hpp"
#include "ParamLoader.hpp"
#include "Types.hpp"

int n_pack = 16;
namespace mllm {
CPUKVCache::CPUKVCache(Backend *bn, string opName, int hidden, int head, int n_rep, int cache_max, int threadCount) :
    thread_count(threadCount), Op(bn, opName) {
    cache_.setBackend(bn);
    switch (KVCache_TYPE) {
    case 16: {
        cache_.setDtype(MLLM_TYPE_F16);
        break;
    }
    case 8: {
        if (opName.find("k_cache") != std::string::npos) {
            cache_.setDtype(MLLM_TYPE_Q8_0);
            n_pack = QK8_0;
        } else {
            cache_.setDtype(MLLM_TYPE_F16);
        }
        break;
    }
    case 32: {
        cache_.setDtype(MLLM_TYPE_F32);
        break;
    }
    default: {
        cache_.setDtype(MLLM_TYPE_F32);
        break;
    }
    }
// #endif
#ifdef LLAMAFILE_SGEMM
    cache_max = ((cache_max + (n_pack - 1)) / n_pack) * n_pack;
#endif
    cache_limit_ = cache_max;
    n_rep_ = n_rep;
    if (head > 0) {
        if (for_xnn_) cache_.setDtype(MLLM_TYPE_F32);

        cache_.reshape(1, head * n_rep_, cache_limit_, hidden);
        cache_.setName(name() + ".Cache");
        cache_.alloc();

        switch (cache_.dtype()) {
        case MLLM_TYPE_F32:
            memset(cache_.hostPtr<float>(), 0, cache_.count() * sizeof(float));
            break;
        case MLLM_TYPE_F16:
            memset(cache_.hostPtr<mllm_fp16_t>(), 0, cache_.count() * sizeof(mllm_fp16_t));
            break;
        case MLLM_TYPE_Q8_0:
            memset((char *)cache_.rawHostPtr(), 0, cache_.count() * sizeof(block_q8_0) / QK8_0);
            break;
        default:
            break;
        };
        cache_seq_len_ = 0;
        cache_.cache_seq_len_ = cache_seq_len_;
    }
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs,
                              vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (cache_seq_len_ < 0) {
        if (for_xnn_) cache_.setDtype(MLLM_TYPE_F32);

        cache_.reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_,
                       inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();

        switch (cache_.dtype()) {
        case MLLM_TYPE_F32:
            memset(cache_.hostPtr<float>(), 0, cache_.count() * sizeof(float));
            break;
        case MLLM_TYPE_F16:
            memset(cache_.hostPtr<mllm_fp16_t>(), 0, cache_.count() * sizeof(mllm_fp16_t));
            break;
        case MLLM_TYPE_Q8_0:
            memset((char *)cache_.rawHostPtr(), 0, cache_.count() * sizeof(block_q8_0) / QK8_0);
            break;
        default:
            break;
        };
        cache_seq_len_ = 0;
        cache_.cache_seq_len_ = cache_seq_len_;
    }

    // for sd
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isUsingDraft()) {
        unsigned int last_draft_length = cpuBackend->getLastDraftLength();
        const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
        cache_seq_len_ = cache_seq_len_ - (last_draft_length) + last_verified_position_ids.size();
        cache_.cache_seq_len_ = cache_seq_len_;
    }

    int sequence = inputs[0]->sequence() + cache_seq_len_;
#ifdef LLAMAFILE_SGEMM
    if (!for_xnn_ && sequence % n_pack != 0) sequence = ((sequence + (n_pack - 1)) / n_pack) * n_pack;
#endif
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, sequence,
                        inputs[0]->dimension());
    if (sequence > cache_limit_) {
        MLLM_LOG_ERROR_STREAM << "\n[ERROR]: Current tokens exceed cache limit: " << sequence << ">"
                              << cache_limit_ << ";"
                              << "\n         Please set args `--limits` >" << cache_limit_ << std::endl;

        exit(1);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_,
                            inputs[0]->dimension());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs,
                              vector<shared_ptr<Tensor>> outputs) {
    // for sd
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isUsingDraft()) {
        const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
        if (!last_verified_position_ids.empty()) {
            this->updateVerifiedKVCache(last_verified_position_ids);
        }
    }

    int cache_seq_len_old = cache_seq_len_;
    cache_seq_len_ += inputs[0]->sequence();
    cache_.cache_seq_len_ = cache_seq_len_;
    if (n_rep_ > 1) {
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
                            } else if (cache_.dtype() == MLLM_TYPE_Q8_0) {
                                auto src_ptr =
                                    (char *)inputs[0]->rawHostPtr() + inputs[0]->offset(b, h, seq - cache_seq_len_old, 0) * sizeof(block_q8_0) / QK8_0;
                                auto dest_ptr = (char *)cache_.rawHostPtr() + cache_.offset(b, cache_head, seq, 0) * sizeof(block_q8_0) / QK8_0;
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(block_q8_0) / QK8_0);
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
                            } else if (cache_.dtype() == MLLM_TYPE_Q8_0) {
                                auto src_ptr =
                                    (char *)inputs[0]->rawHostPtr() + inputs[0]->offset(b, h, 0, d) * sizeof(block_q8_0) / QK8_0;
                                auto dest_ptr = (char *)cache_.rawHostPtr() + cache_.offset(b, cache_head, cache_seq_len_old, d) * sizeof(block_q8_0) / QK8_0;
                                int copy_size = cache_.dimension();
                                memcpy(dest_ptr, src_ptr, copy_size * sizeof(block_q8_0) / QK8_0);
                            }
                        }
                    }
                }
            }
        } else {
            std::cout << "ERROR Ctype in KVCcache;" << std::endl;
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
    outputs[0]->shallowCopyFrom(cache_, false, {0, 0, cache_seq_len_ / cache_limit_, 0});
    if (inputs[0]->sequence() + cache_seq_len_ > cache_limit_) {
        outputs[0]->shallowCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_ + 1, 0});
    }
    if (inputs[0]->masterTensor() == nullptr) { inputs[0]->free(); }
    inputs[0]->shallowCopyFrom(cache_, false, {0, 0, cache_seq_len_ % cache_limit_, 0});
    return MLLM_NO_ERROR;
}

ErrorCode CPUKVCache::updateVerifiedKVCache(const std::vector<unsigned int> &verified_position_ids) {
    if (cache_.ctype() == BSHD) {
        unsigned int dest_pid = cache_seq_len_ - verified_position_ids.size();
        for (unsigned int src_pid : verified_position_ids) {
            if (src_pid == dest_pid) {
                dest_pid += 1;
                continue;
            }
            // #pragma omp parallel for collapse(1) num_threads(thread_count)
            for (int b = 0; b < cache_.batch(); ++b) {
                if (cache_.dtype() == MLLM_TYPE_F32) {
                    auto src_ptr = cache_.ptrAt<float>(b, 0, src_pid, 0);
                    auto dest_ptr = cache_.ptrAt<float>(b, 0, dest_pid, 0);
                    int copy_size = cache_.dimension() * cache_.head();
                    memcpy(dest_ptr, src_ptr, copy_size * sizeof(float));
                } else if (cache_.dtype() == MLLM_TYPE_F16) {
                    auto src_ptr = cache_.ptrAt<mllm_fp16_t>(b, 0, src_pid, 0);
                    auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, 0, dest_pid, 0);
                    int copy_size = cache_.dimension() * cache_.head();
                    memcpy(dest_ptr, src_ptr, copy_size * sizeof(mllm_fp16_t));
                } else if (cache_.dtype() == MLLM_TYPE_Q8_0) {
                    // TODO: Q8 Check
                    auto src_ptr =
                        (char *)cache_.rawHostPtr() + cache_.offset(b, 0, src_pid, 0) * sizeof(block_q8_0) / QK8_0;
                    auto dest_ptr = (char *)cache_.rawHostPtr() + cache_.offset(b, 0, dest_pid, 0) * sizeof(block_q8_0) / QK8_0;
                    int copy_size = cache_.dimension() * cache_.head();
                    memcpy(dest_ptr, src_ptr, copy_size * sizeof(block_q8_0) / QK8_0);
                }
            }
            dest_pid += 1;
        }
    } else if (cache_.ctype() == BHDS) {
        unsigned int dest_pid = cache_seq_len_ - verified_position_ids.size();
        for (unsigned int src_pid : verified_position_ids) {
            if (src_pid == dest_pid) {
                dest_pid += 1;
                continue;
            }
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < cache_.batch(); ++b) {
                for (int h = 0; h < cache_.head(); ++h) {
                    for (int d = 0; d < cache_.dimension(); ++d) {
                        if (cache_.dtype() == MLLM_TYPE_F32) {
                            auto src_data = cache_.dataAt<float>(b, h, src_pid, d);
                            cache_.setDataAt<float>(b, h, dest_pid, d, src_data);
                        } else if (cache_.dtype() == MLLM_TYPE_F16) {
                            auto src_data = cache_.dataAt<mllm_fp16_t>(b, h, src_pid, d);
                            cache_.setDataAt<mllm_fp16_t>(b, h, dest_pid, d, src_data);
                        } else if (cache_.dtype() == MLLM_TYPE_Q8_0) {
                            // TODO: Q8 Check 不知道q8能不能直接setDataAt
                            // auto src_data = cache_.dataAt<block_q8_0>(b, h, src_pid, d);
                            // cache_.setDataAt<block_q8_0>(b, h, dest_pid, d, src_data);
                            auto src_ptr =
                                (char *)cache_.rawHostPtr() + cache_.offset(b, h, src_pid, d) * sizeof(block_q8_0) / QK8_0;
                            auto dest_ptr = (char *)cache_.rawHostPtr() + cache_.offset(b, h, dest_pid, d) * sizeof(block_q8_0) / QK8_0;
                            int copy_size = 1;
                            memcpy(dest_ptr, src_ptr, copy_size * sizeof(block_q8_0) / QK8_0);
                        }
                    }
                }
            }
            dest_pid += 1;
        }
    } else {
        std::cout << "ERROR Ctype in KVCcache;" << std::endl;
    }

    // clear kv cache
    // if (cache_seq_len_ < cache_seq_len_old) {
    //     if (n_rep_ > 1) {
    //         if (cache_.ctype() == BSHD) {
    //             for (int b = 0; b < cache_.batch(); ++b) {
    //                 for (int h = cache_.head() - 1; h >= 0; --h) {
    // // #pragma omp parallel for collapse(2) num_threads(thread_count)
    //                     for (int seq = cache_seq_len_; seq < cache_seq_len_old; ++seq) {
    //                         for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
    //                             auto cache_head = h * n_rep_ + i_rep;
    //                             if (cache_.dtype() == MLLM_TYPE_F32) {
    //                                 auto dest_ptr = cache_.ptrAt<float>(b, cache_head, seq, 0);
    //                                 int copy_size = cache_.dimension();
    //                                 memset(dest_ptr, 0, copy_size * sizeof(float));
    //                             } else if (cache_.dtype() == MLLM_TYPE_F16) {
    //                                 auto dest_ptr = cache_.ptrAt<mllm_fp16_t>(b, cache_head, seq, 0);
    //                                 int copy_size = cache_.dimension();
    //                                 memset(dest_ptr, 0, copy_size * sizeof(mllm_fp16_t));
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (cache_.ctype() == BHDS) {
    //             for (int b = 0; b < cache_.batch(); ++b) {
    //                 for (int h = cache_.head() - 1; h >= 0; --h) {
    // // #pragma omp parallel for collapse(2) num_threads(thread_count)
    //                     for (int d = 0; d < cache_.dimension(); ++d) {
    //                         for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
    //                             auto cache_head = h * n_rep_ + i_rep;
    //                             if (cache_.dtype() == MLLM_TYPE_F32) {
    //                                 auto dest_ptr =
    //                                     cache_.ptrAt<float>(b, cache_head, cache_seq_len_, d);
    //                                 int copy_size = cache_seq_len_old - cache_seq_len_;
    //                                 memset(dest_ptr, 0, copy_size * sizeof(float));
    //                             } else if (cache_.dtype() == MLLM_TYPE_F16) {
    //                                 auto dest_ptr =
    //                                     cache_.ptrAt<mllm_fp16_t>(b, cache_head, cache_seq_len_, d);
    //                                 int copy_size = cache_seq_len_old - cache_seq_len_;
    //                                 memset(dest_ptr, 0, copy_size * sizeof(mllm_fp16_t));
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else {
    //             std::cout << "ERROR Ctype in KVCcache;" << std::endl;
    //         }
    //     }
    // }
    return MLLM_NO_ERROR;
}

} // namespace mllm
#include "backends/cpu/op/CPUKVCacheXp.hpp"
#include "Types.hpp"

namespace mllm {

CPUKVCacheXp::CPUKVCacheXp(Backend *bn, const string &op_name, int n_rep, int cache_max, int thread_count) :
    Op(bn, op_name), n_rep_(n_rep), cache_limit_(cache_max), thread_count_(thread_count) {
    cache_.setBackend(bn);
    cache_.setDtype(MLLM_TYPE_F32);
}

ErrorCode CPUKVCacheXp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    if (cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        memset(cache_.hostPtr<float>(), 0, cache_.count() * sizeof(float));
        cache_seq_len_ = 0;
    }

    int sequence = inputs[0]->sequence() + cache_seq_len_;
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, sequence, inputs[0]->dimension());

    if (sequence > cache_limit_) {
        MLLM_LOG_ERROR_STREAM << "\n[ERROR]: Current tokens exceed cache limit: " << sequence << ">"
                              << cache_limit_ << ";"
                              << "\n         Please set args `--limits` >" << cache_limit_ << std::endl;
        exit(-1);
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCacheXp::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUKVCacheXp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int cache_seq_len_old = cache_seq_len_;
    cache_seq_len_ += inputs[0]->sequence();

    // copy input to cache
    for (int b = 0; b < cache_.batch(); ++b) {
        for (int h = 0; h < cache_.head(); ++h) {
#pragma omp parallel for collapse(2) num_threads(thread_count_)
            for (int seq = cache_seq_len_old; seq < cache_seq_len_; ++seq) {
                for (int i_rep = 0; i_rep < n_rep_; ++i_rep) {
                    auto cache_head = h * n_rep_ + i_rep;
                    auto src_ptr = inputs[0]->ptrAt<float>(b, h, seq - cache_seq_len_old, 0);
                    auto dst_ptr = cache_.ptrAt<float>(b, cache_head, seq, 0);
                    int copy_size = cache_.dimension();
                    memcpy(dst_ptr, src_ptr, copy_size * sizeof(float));
                }
            }
        }
    }

    // copy cache to output
    // memcpy(outputs[0]->rawHostPtr(), cache_.rawHostPtr(), outputs[0]->count() * sizeof(float));

    return MLLM_NO_ERROR;
}

ErrorCode CPUKVCacheXp::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUKVCacheXp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->forceResetHostPointer(cache_.rawHostPtr());
    return MLLM_NO_ERROR;
}
} // namespace mllm
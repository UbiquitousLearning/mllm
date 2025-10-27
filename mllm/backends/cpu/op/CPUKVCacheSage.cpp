

#include "CPUKVCacheSage.hpp"
#include "ParamLoader.hpp"
#include "Types.hpp"
#include "../compute/SageQuantize.hpp"
#include <ostream>

// int n_pack = 16;
namespace mllm {
CPUKVCacheSage::CPUKVCacheSage(Backend *bn, string opName, int hidden, int head, int n_rep, bool fa2, int cache_max, int threadCount) :
    thread_count(threadCount), Op(bn, opName) {
    cache_ = std::make_shared<Tensor>(bn);
    cache_->setDtype(MLLM_TYPE_Q8_0F);
    n_rep = 1;

    cache_limit_ = cache_max;
    n_rep_ = n_rep;
    if (head > 0) {
        // cache_->setCtype(BHSD);
        cache_->reshape(1, head * n_rep_, cache_limit_, hidden);
        cache_->setName(name() + ".Cache");
        cache_->alloc();

        // memset((char *)cache_->rawHostPtr(), 0, cache_->count() * sizeof(block_q8_0f) / QK8_0F);
        cache_->seqMeans().resize(1 * head * hidden);
        cache_seq_len_ = 0;
        cache_->cache_seq_len_ = cache_seq_len_;
    }
}

ErrorCode CPUKVCacheSage::reshape(vector<shared_ptr<Tensor>> inputs,
                                  vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (cache_seq_len_ < 0) {
        // cache_->setCtype(BHSD);
        cache_->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_,
                        inputs[0]->dimension());
        cache_->setName(name() + ".Cache");
        cache_->alloc();

        // memset((char *)cache_->rawHostPtr(), 0, cache_->count() * sizeof(block_q8_0f) / QK8_0F);
        cache_->seqMeans().resize(inputs[0]->batch() * inputs[0]->head() * inputs[0]->dimension());
        cache_seq_len_ = 0;
        cache_->cache_seq_len_ = cache_seq_len_;
    }

    int sequence = inputs[0]->sequence() + cache_seq_len_;
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

ErrorCode CPUKVCacheSage::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUKVCacheSage::execute(vector<shared_ptr<Tensor>> inputs,
                                  vector<shared_ptr<Tensor>> outputs) {
    int cache_seq_len_old = cache_seq_len_;

    auto new_tokens = inputs[0];
    const int batch_size = new_tokens->batch();
    const int kv_head = new_tokens->head();
    const int seq_len = new_tokens->sequence();
    const int dim = new_tokens->dimension();
    const int num_k_blocks = dim / QK8_0F;
#pragma omp parallel for collapse(2) num_threads(thread_count)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < kv_head; ++h) {
            float *p_mean = &cache_->seqMeans()[(b * kv_head + h) * dim];

            if (seq_len > 1) { // Prefill 阶段
                const size_t bshd_s_stride = (size_t)new_tokens->head() * new_tokens->dimension();
                const float *head_start_ptr = new_tokens->ptrAt<float>(b, h, 0, 0);

                sage_kv_cache::compute_sage_mean_for_one_head_bshd(head_start_ptr, p_mean, seq_len, dim, bshd_s_stride);

                for (int s = 0; s < seq_len; ++s) {
                    const float *token_to_quantize = new_tokens->ptrAt<float>(b, h, s, 0);

                    size_t block_offset = ((size_t)b * cache_->sequence() + s) * cache_->head() + h;
                    block_q8_0f *p_dest = reinterpret_cast<block_q8_0f *>(cache_->rawHostPtr()) + block_offset * num_k_blocks;

                    sage_kv_cache::quantize_new_token_to_sage_blocks(token_to_quantize, p_mean, p_dest, dim);
                }
            } else { // Decode 阶段
                const float *p_new_token = new_tokens->ptrAt<float>(b, h, 0, 0);
                sage_kv_cache::update_sage_mean_vector_incremental(p_mean, p_new_token, cache_seq_len_old, dim);

                size_t block_offset = ((size_t)b * cache_->sequence() + cache_seq_len_old) * cache_->head() + h;
                block_q8_0f *p_dest = reinterpret_cast<block_q8_0f *>(cache_->rawHostPtr()) + block_offset * num_k_blocks;

                sage_kv_cache::quantize_new_token_to_sage_blocks(p_new_token, p_mean, p_dest, dim);
            }
        }
    }

    cache_seq_len_ += inputs[0]->sequence();
    cache_->cache_seq_len_ = cache_seq_len_;
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCacheSage::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUKVCacheSage::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    // for BSHD attention end
    outputs[0]->setDtype(cache_->dtype());
    outputs[0]->shallowCopyFrom(cache_, false, {0, 0, 0, 0});

    return MLLM_NO_ERROR;
}

} // namespace mllm
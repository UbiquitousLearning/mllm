#include "backends/xnnpack/Ops/XpKVCache.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

ErrorCode XpKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (cache_seq_len_ < 0) {
        // FIXME: memory waste if n_rep_ > 1
        cache_params_.reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, cache_limit_,
                              inputs[0]->dimension());
        cache_params_.setName(name() + ".Cache");
        cache_params_.alloc();
        memset(cache_params_.hostPtr<float>(), 0, cache_params_.count() * sizeof(float));
        cache_seq_len_ = 0;
    }
    int sequence = inputs[0]->sequence() + cache_seq_len_;
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep_, sequence,
                        inputs[0]->dimension());
    if (sequence > cache_limit_) {
        Log::error("sequence={} > cache_limit_={}", sequence, cache_limit_);
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode XpKVCache::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode XpKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);
    defineWeightTensor(xpb, &cache_params_);

    int cache_seq_len_old = cache_seq_len_;
    cache_seq_len_ += inputs[0]->sequence();

    if (cache_params_.ctype() != BSHD) {
        Log::error("cache_params_.ctype() != BSHD, XpKVCache only support BSHD data layout");
        exit(-1);
    }

    size_t b = inputs[0]->batch();
    size_t s = inputs[0]->sequence();
    size_t h = inputs[0]->head();
    size_t d = inputs[0]->dimension();
    auto o_dtype = outputs[0]->dtype();

    for (int i = 0; i < 8; ++i) {
        Log::error("cache idx={}, number={}", i, *(cache_params_.hostPtr<float>() + i));
    }

    // TODO buggy
    // copy inputs[0] to cache_params_
    for (int i = 0; i < n_rep_; ++i) {
        // slice a tensor from cache_param_, but share same memory
        // SHAPE: [b, s, h, d]
        // OFFSET: [0, cache_seq_len_old, i * h, 0]
        std::array<size_t, 4> new_shape{b, s, h, d};
        std::array<size_t, 4> offsets{0, (size_t)cache_seq_len_old, i * h, 0};

        uint32_t o1 = 0;

        auto status = xnn_define_tensor_value(
            xpb->getXnnSubgraph(), xnn_datatype_fp32,
            new_shape.size(), new_shape.data(),
            /*data=*/cache_params_.hostPtr<float>() + cache_params_.offset(0, i * (int)h, cache_seq_len_old, 0),
            XNN_INVALID_VALUE_ID, 0, &o1);

        status = xnn_define_copy(xpb->getXnnSubgraph(), inputs[0]->uuid(), o1, 0);
        if (status != xnn_status_success) {
            Log::error("XpKVCache xnn_define_copy from inputs to cache_params_ failed");
            exit(-1);
        }
    }

    // copy cache_params_ to outputs[0]
    {
        std::array<size_t, 4> new_shape{b, (size_t)cache_seq_len_old, h * n_rep_, d};
        std::array<size_t, 4> offsets{0, 0, 0, 0};
        auto o1 = defineTemporaryTensor(xpb, {b, (size_t)cache_seq_len_old, h * n_rep_, d}, cache_params_.dtype());
        auto status = xnn_define_static_slice(xpb->getXnnSubgraph(), 4, offsets.data(), new_shape.data(), cache_params_.uuid(), o1, 0);
        if (status != xnn_status_success) {
            Log::error("XpKVCache xnn_define_static_slice for cache_params_ failed");
            exit(-1);
        }

        status = xnn_define_copy(xpb->getXnnSubgraph(), o1, outputs[0]->uuid(), 0);
        if (status != xnn_status_success) {
            Log::error("XpKVCache xnn_define_copy from cache_params to outputs failed");
            exit(-1);
        }
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    cache_params_.free();
    return MLLM_NO_ERROR;
}

ErrorCode XpKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

Tensor &XpKVCache::getCacheTensor() {
    return cache_params_;
}
} // namespace mllm::xnnpack
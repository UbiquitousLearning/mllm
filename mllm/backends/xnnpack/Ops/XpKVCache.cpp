#include "backends/xnnpack/Ops/XpKVCache.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include <cassert>
#include <cstddef>

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
    return MLLM_NO_ERROR;
}

ErrorCode XpKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    cache_params_.free();
    return MLLM_NO_ERROR;
}

ErrorCode XpKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);
    defineWeightTensor(xpb->getCurProcessingGraph(), &cache_params_);

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

    if (n_rep_ == 0) {
        Log::warn("XpKVCache found n_rep_=0, XpKVCache will reset it to 1");
        n_rep_ = 1;
    }

    // step 1. inputs[0] copy to cache_params_(wrap to external output)
    {
        for (int i = 0; i < n_rep_; ++i) {
            std::array<size_t, 4> new_shape{b, s, h, d};
            std::array<size_t, 4> offsets{0, (size_t)cache_seq_len_old, i * h, 0};

            auto o1 = defineKVCacheTensorAsExternalOutput(xpb->getCurProcessingGraph(), &cache_params_, cache_params_.offset(0, (int)i * h, cache_seq_len_old, 0), {b, s, h, d});

            auto status = xnn_define_copy(xpb->getCurProcessingGraph()->getXnnSubgraph(), inputs[0]->uuid(), o1, 0);
            if (status != xnn_status_success) {
                Log::error("xnn_define_copy inputs[0] copy to cache_params_(wrap to external output) failed");
                exit(-1);
            }
        }
    }

    // step 2. cache_params_(sliced) to outputs[0]
    {
        std::array<size_t, 4> new_shape{b, (size_t)cache_seq_len_, h * n_rep_, d};

        assert(b == outputs[0]->batch());
        assert(cache_seq_len_ == outputs[0]->sequence());
        assert(h * n_rep_ == outputs[0]->head());
        assert(d == outputs[0]->dimension());

        std::array<size_t, 4> offsets{0, 0, 0, 0};

        auto o1 = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, (size_t)cache_seq_len_, h * n_rep_, d}, cache_params_.dtype());

        auto status = xnn_define_static_slice(xpb->getCurProcessingGraph()->getXnnSubgraph(), 4, offsets.data(), new_shape.data(), cache_params_.uuid(), o1, 0);
        if (status != xnn_status_success) {
            Log::error("xnn_define_static_slice cache_params_(sliced) copy to outputs[0] failed");
            exit(-1);
        }

        status = xnn_define_copy(xpb->getCurProcessingGraph()->getXnnSubgraph(), o1, outputs[0]->uuid(), 0);
        if (status != xnn_status_success) {
            Log::error("xnn_define_copy cache_params_(wrap to external inputs) copy to outputs[0] failed");
            exit(-1);
        }
    }
    return MLLM_NO_ERROR;
}

Tensor &XpKVCache::getCacheTensor() {
    return cache_params_;
}
} // namespace mllm::xnnpack
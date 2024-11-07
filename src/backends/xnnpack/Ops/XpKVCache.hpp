/**
 * @file XpKVCache.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Op.hpp"
#include "ParamLoader.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "backends/xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {
class XpKVCache final : public Op, public XpTensorDefineInterface<XpKVCache> {
public:
    XpKVCache(Backend *bk, const std::string &op_name, int n_rep, int cache_max = 100, int thread_count = 4) :
        Op(bk, op_name), n_rep_(n_rep), cache_limit_(cache_max), thread_count_(thread_count) {
        cache_params_.setBackend(bk);
        Log::warn("XpKVCache only support fp32 data type right now ");
    }

    ~XpKVCache() override = default;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    int getCacheSeqLen() override {
        return cache_seq_len_;
    }

    void clearCache() override {
        cache_seq_len_ = 0;
    }

    Tensor &getCacheTensor();

private:
    Tensor cache_params_;
    int thread_count_ = 4;
    int cache_seq_len_ = -999;
    int n_rep_ = 1;
    int cache_limit_ = 512;
};

class XpKVCacheCreator : public XnnpackBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override {
        int n_rep = (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        return new XpKVCache(bk, name, n_rep, cache_max, thread_count);
    }
};
} // namespace mllm::xnnpack
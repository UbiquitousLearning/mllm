/**
 * @file CPUKVCacheXp.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {

class CPUKVCacheXp final : public Op {
public:
    ~CPUKVCacheXp() override = default;
    CPUKVCacheXp(Backend *bn, const string &op_name, int n_rep, int cache_max = 100, int thread_count = 4);
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

private:
    Tensor cache_;
    int thread_count_ = 4;
    int cache_seq_len_ = -999;
    int n_rep_ = 1;
    int cache_limit_;
};

class CPUKVCacheXpCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int thread_count) const {
        int n_rep = (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        auto ret = new CPUKVCacheXp(bn, name, n_rep, cache_max, thread_count);
        return ret;
    }
};
} // namespace mllm
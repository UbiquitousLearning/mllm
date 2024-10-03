/**
 * @file XnnpackBackend.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "MemoryManager.hpp"
#include "Backend.hpp"
#include <functional>
#include <memory>
#include <cstdint>

#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

class XnnpackModelRuntime {
public:
    explicit XnnpackModelRuntime(int32_t num_threads);
    ~XnnpackModelRuntime();

    bool createModel(const std::function<xnn_subgraph_t()> &model_factory);

    bool createRuntime(uint32_t flags);

    bool reshapeRuntime();

    bool setupRuntime();

    bool invoke();

private:
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model_;
    pthreadpool_t threadpool_ = nullptr;
    xnn_runtime_t runtime_ = nullptr;
    std::vector<xnn_external_value> external_values_;
    int32_t num_threads_;
};

struct XnnpackBackendOpts {
    int32_t num_threads = 4;
};

class XnnpackBackend : public Backend {
public:
    XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts);
    ~XnnpackBackend();

public:
    struct Creator {
        virtual Op *create(OpParam op_param, Backend *bn, const string &name, int thread_count) const = 0;
    };

    bool addCreator(OpType t, Creator *c);

    Op *opCreate(const OpParam &op_param, string name, int thread_count) override;

    TensorFunction *funcCreate(TensorFuncType type) override;

    void registerOps() override;

    void registerFuncs() override;

private:
    XnnpackBackendOpts opts_;
    std::map<OpType, XnnpackBackend::Creator *> map_op_creator_;
    std::map<TensorFuncType, TensorFunction *> map_tensor_function_;
};

} // namespace mllm::xnnpack
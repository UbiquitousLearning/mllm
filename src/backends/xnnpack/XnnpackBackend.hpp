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
#include <memory>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

class XnnpackModelRuntime {
public:
    explicit XnnpackModelRuntime(int32_t num_threads);
    ~XnnpackModelRuntime();

    bool createModel(const xnn_subgraph_t &model_factory);

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

class XnnpackTensorSymbolTable {
public:
private:
    // user side tensor:
    // user should free these tensor manually
    std::unordered_map<std::string, uint32_t> user_side_tensor_;

    // xnnpack will free this tensor when necessary
    std::unordered_map<std::string, uint32_t> xnn_side_tensor_;

    // all tensors
    std::unordered_map<std::string, uint32_t> all_tensor_;
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

    std::shared_ptr<XnnpackModelRuntime> getModelRuntime();

    std::shared_ptr<XnnpackModelRuntime> recreateModelRuntime(int thread_count = 1);

    xnn_subgraph_t getXnnSubgraph();

private:
    XnnpackBackendOpts opts_;

    // xnn stuff
    xnn_subgraph_t subgraph_ = nullptr;
    std::shared_ptr<XnnpackModelRuntime> model_runtime_ = nullptr;

    std::map<OpType, XnnpackBackend::Creator *> map_op_creator_;
    std::map<TensorFuncType, TensorFunction *> map_tensor_function_;
};

} // namespace mllm::xnnpack
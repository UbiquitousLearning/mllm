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

    void resetUuidExternalValuesMap(const std::unordered_map<uint32_t, xnn_external_value> &ext_vals);

public:
    std::unordered_map<uint32_t, xnn_external_value> &__uuidToExternalsV();

private:
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
        model_;
    pthreadpool_t threadpool_ = nullptr;
    xnn_runtime_t runtime_ = nullptr;
    std::vector<xnn_external_value> external_values_;
    std::unordered_map<uint32_t, xnn_external_value> uuid_2_externals_v_;
    int32_t num_threads_;
};

struct XnnpackBackendOpts {
    int32_t num_threads = 4;
};

class XnnpackBackend : public Backend {
public:
    explicit XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts = XnnpackBackendOpts{.num_threads = 4});
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

    void createSubgraph(int32_t external_nums = 16);

    void recreateSubgraph(int32_t external_nums = 16);

    void registerExternalValue(uint32_t uuid, const xnn_external_value &ext_v);

    void registerUuidTensor(uint32_t uuid, Tensor *t);

    void *getExternalValueptr(uint32_t uuid);

    static xnn_datatype mllmDType2XnnDType(DataType mllm_dtype);

    uint32_t getNewEXternalId();

    void assignPtrToTensor();

private:
    XnnpackBackendOpts opts_;

    // external values
    std::unordered_map<uint32_t, Tensor *> uuid_2_mllm_tensor_;
    std::unordered_map<uint32_t, xnn_external_value> uuid_2_externals_v_;

    // xnn stuff
    xnn_subgraph_t subgraph_ = nullptr;

    std::shared_ptr<XnnpackModelRuntime> model_runtime_ = nullptr;

    std::map<OpType, XnnpackBackend::Creator *> map_op_creator_;
    std::map<TensorFuncType, TensorFunction *> map_tensor_function_;
};

} // namespace mllm::xnnpack
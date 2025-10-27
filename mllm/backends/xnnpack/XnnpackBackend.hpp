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
#include "pthreadpool.h"
#include "xnnpack.h"
namespace mllm {
class Module;
class Layer;
} // namespace mllm
namespace mllm::xnnpack {

class XnnpackModelRuntime {
public:
    explicit XnnpackModelRuntime(pthreadpool_t threadpool);

    ~XnnpackModelRuntime();

    bool createModel(const xnn_subgraph_t &model_factory);

    bool createRuntime(uint32_t flags);

    bool reshapeRuntime();

    bool setupRuntime();

    bool invoke();

    void resetUuidExternalValuesMap(const std::unordered_map<uint32_t, xnn_external_value> &ext_vals);

    void setWeightCache(xnn_weights_cache_t weight_cache);

    xnn_runtime_t getXnnRt();

public:
    std::unordered_map<uint32_t, xnn_external_value> &__uuidToExternalsV();

private:
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model_;
    pthreadpool_t threadpool_ = nullptr;
    xnn_runtime_t runtime_ = nullptr;
    std::vector<xnn_external_value> external_values_;
    std::unordered_map<uint32_t, xnn_external_value> uuid_2_externals_v_;
    int32_t num_threads_;
    xnn_weights_cache_t weight_cache_ = nullptr;
};

struct XnnpackBackendOpts {
    int32_t num_threads = 4;
};

class XnnpackBackend;

class XnnpackCargo {
    XnnpackBackend *backend_ = nullptr;
    xnn_subgraph_t graph_ = nullptr;
    std::unordered_map<uint32_t, Tensor *> uuid_2_mllm_tensor_;
    std::unordered_map<uint32_t, xnn_external_value> uuid_2_externals_v_;
    std::unordered_map<uint32_t, Tensor *> uuid_2_mllm_weight_tensor_;
    std::unordered_map<uint32_t, bool> uuid_2_normal_tensor_;
    std::shared_ptr<XnnpackModelRuntime> model_runtime_ = nullptr;
    pthreadpool_t threadpool_ = nullptr;
    xnn_weights_cache_t weight_cache_ = nullptr;
    bool weight_cache_finalized_ = false;
    bool subgraph_dispatched_ = false;
    std::unordered_map<std::string, uint32_t> activation_name_2_uuid_;
    uint32_t exec_cnt_ = 0;

public:
    uint32_t getExecCnt();

    uint32_t incExecCnt();

    void setThreadPool(pthreadpool_t tp);

    std::shared_ptr<XnnpackModelRuntime> getModelRuntime();

    std::shared_ptr<XnnpackModelRuntime> recreateModelRuntime();

    xnn_subgraph_t getXnnSubgraph();

    void createSubgraph(int32_t external_nums = 4096);

    void recreateSubgraph(int32_t external_nums = 4096);

    void registerExternalValue(uint32_t uuid, const xnn_external_value &ext_v);

    void updateExternalValue(uint32_t uuid, const xnn_external_value &ext_v);

    void registerNormalValue(uint32_t uuid);

    void registerUuidTensor(uint32_t uuid, Tensor *t);

    void updateUuidTensor(uint32_t uuid, Tensor *t);

    void registerUuidWeightTensor(uint32_t uuid, Tensor *t);

    void *getExternalValueptr(uint32_t uuid);

    bool hasExternalValue(uint32_t uuid);

    bool hasNormalValue(uint32_t uuid);

    bool hasWeightValue(uint32_t uuid);

    uint32_t getNewEXternalId();

    void assignPtrToTensor();

    void setSubgraphDispatched(bool b);

    xnn_weights_cache_t getWeightCache();

    bool isWeightCacheFinalized() const;

    void setWeightCacheFinalized(bool b);

    bool inActivationName(const std::string &name);

    uint32_t getUUIDByActivationName(const std::string &name);

    void registerActivationNameAndUUID(const std::string &name, uint32_t uuid);
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

    static xnn_datatype mllmDType2XnnDType(DataType mllm_dtype);

    void createNewGraph(const std::string &name);

    std::shared_ptr<XnnpackCargo> getGraph(const std::string &name);

    bool hasGraph(const std::string &name);

    void onSetUpStart(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs, std::string graph_name) override;

    void onSetUpEnd(std::vector<shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs, std::string graph_name) override;

    void onExecuteStart(std::vector<shared_ptr<Tensor>> &inputs, std::vector<shared_ptr<Tensor>> &outputs, std::string graph_name) override;

    void onExecuteEnd(std::vector<std::shared_ptr<Tensor>> &outputs, const string &graph_name) override;

    std::vector<Tensor> runFunc(
        std::vector<std::string> out_names,
        TensorFuncType type,
        std::vector<float> float_args,
        std::vector<Tensor> input_tensors,
        bool in_place) override;
    std::vector<Tensor> runLayer(Layer *layer, std::vector<Tensor> inputs, int N) override;

    std::vector<Tensor> runOp(Op *op, std::vector<Tensor> input, std::vector<std::string> out_names, bool in_place) override;
    std::vector<Tensor> runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) override;

    XnnpackCargo *getCurProcessingGraph();

    static int xnn_threads;

    static bool enable_dynamic_shape;

    static bool enable_legacy_wrapper;

private:
    pthreadpool_t threadpool_ = nullptr;
    XnnpackBackendOpts opts_;
    std::unordered_map<std::string, std::shared_ptr<XnnpackCargo>> graphs_;

    std::string cur_processing_graph_name_;

    std::map<OpType, XnnpackBackend::Creator *> map_op_creator_;
    std::map<TensorFuncType, TensorFunction *> map_tensor_function_;
};

} // namespace mllm::xnnpack
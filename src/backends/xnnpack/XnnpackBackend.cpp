#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "pthreadpool.h"
#include "xnnpack.h"
#include "backends/xnnpack/Functions/XpBinaryFunc.hpp"
#include "backends/xnnpack/Ops/XpBinary.hpp"
#include "backends/xnnpack/XpMemoryManager.hpp"
#include "backends/xnnpack/Ops/XpDirect.hpp"
#include "backends/xnnpack/Ops/XpDispatch.hpp"
#include "backends/xnnpack/Ops/XpLinear.hpp"
#include "backends/xnnpack/Ops/XpMatmul.hpp"
#include "backends/xnnpack/Ops/XpRoPE.hpp"
#include "backends/xnnpack/Ops/XpSubGraphStart.hpp"
#include "backends/xnnpack/Ops/XpSubGraphFinalize.hpp"
#include "backends/xnnpack/Ops/XpD2H.hpp"
#include "backends/xnnpack/Ops/XpReLU.hpp"
#include "backends/xnnpack/Ops/XpSoftmax.hpp"
#include "backends/xnnpack/Ops/XpGeLU.hpp"
#include "backends/xnnpack/Ops/XpSiLU.hpp"
#include "backends/xnnpack/Ops/XpTranspose.hpp"
#include "backends/xnnpack/Functions/XpTransposeFunc.hpp"
#include "backends/xnnpack/Ops/XpRMSNorm.hpp"
#include "backends/xnnpack/Ops/XpKVCache.hpp"
#include "backends/xnnpack/Ops/XpCausalMask.hpp"
#include "backends/xnnpack/Ops/XpSDPA.hpp"
#include "backends/xnnpack/Functions/XpViewFunc.hpp"
#include "backends/xnnpack/Functions/XpMatmulFunc.hpp"
#include "backends/xnnpack/Ops/XpEmbedding.hpp"
#include "backends/xnnpack/Ops/XpParameter.hpp"
#include "xnnpack/allocator.h"
#include "xnnpack/memory.h"
#include "xnnpack/subgraph.h"
#include <cstdlib>
#include <memory>

namespace mllm {

class XpBackendCreator : public BackendCreator {
    Backend *create(BackendConfig config) override {
        // initialize xnnpack
        if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
            ::mllm::xnnpack::Log::error("failed to initialize XNNPACK");
            return nullptr;
        }

        auto mm = std::make_shared<::mllm::xnnpack::XpMemoryManager>();
        return new ::mllm::xnnpack::XnnpackBackend(mm);
    };
};

void registerXNNBackendCreator() {
    ::mllm::xnnpack::Log::info("xnnpack backend registered");
    InsertBackendCreatorMap(MLLM_XNNPACK, std::make_shared<XpBackendCreator>());
}
} // namespace mllm

namespace mllm::xnnpack {

XnnpackModelRuntime::XnnpackModelRuntime(pthreadpool_t threadpool) :
    threadpool_(threadpool), model_(nullptr, xnn_delete_subgraph) {
    num_threads_ = pthreadpool_get_threads_count(threadpool);
}

XnnpackModelRuntime::~XnnpackModelRuntime() {
    if (runtime_) {
        xnn_delete_runtime(runtime_);
    }

    // not release all
    // FIXME: explicit memory leak.
    // NOTE: explicit memory leak.
    // NOTE: explicit memory leak.
    // NOTE: explicit memory leak.
    // NOTE: explicit memory leak.
    // NOTE: explicit memory leak.
    // NOTE: explicit memory leak.
    //
    // for (auto i = 0; i < external_values_.size(); ++i) {
    //     if ((model_->values[i].flags & ((uint32_t)XNN_VALUE_FLAG_EXTERNAL_INPUT)) == 1) {
    //         xnn_release_simd_memory(uuid_2_externals_v_[i].data);
    //     }
    // }
}

bool XnnpackModelRuntime::createModel(const xnn_subgraph_t &model_factory) {
    model_.reset(model_factory);
    if (!model_) {
        Log::error("failed to create model");
        return false;
    }

    for (uint32_t i = 0; i < model_->num_values; ++i) {
        // if not external values. ignore alloc memory
        if ((model_->values[i].flags & ((uint32_t)XNN_VALUE_FLAG_EXTERNAL_INPUT | (uint32_t)XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
            continue;
        }

        // The prepared external_num > actually external_num, ignore redundant part.
        if (uuid_2_externals_v_.count(i)) {
            // if already alloced by user, ignore alloc memory
            if (uuid_2_externals_v_[i].data) {
                external_values_.push_back(xnn_external_value{i, uuid_2_externals_v_[i].data});
                continue;
            }

            // Make a buffer for this external value.
            size_t size = xnn_tensor_get_size(&model_->values[i]) + XNN_EXTRA_BYTES;
            auto ev = xnn_external_value{i, xnn_allocate_zero_simd_memory(size)};
            uuid_2_externals_v_[i] = ev;
            external_values_.push_back(ev);
        }
    }

    return model_ != nullptr;
}

bool XnnpackModelRuntime::createRuntime(uint32_t flags) {
    assert(!runtime_);
    // flags |= XNN_FLAG_NO_OPERATOR_FUSION;
    return xnn_status_success == xnn_create_runtime_v4(model_.get(), weight_cache_, nullptr, threadpool_, flags, &runtime_);
}

bool XnnpackModelRuntime::reshapeRuntime() {
    return xnn_status_success == xnn_reshape_runtime(runtime_);
}

bool XnnpackModelRuntime::setupRuntime() {
    return xnn_status_success == xnn_setup_runtime_v2(runtime_, external_values_.size(), external_values_.data());
}

bool XnnpackModelRuntime::invoke() {
    return xnn_status_success == xnn_invoke_runtime(runtime_);
}

void XnnpackModelRuntime::resetUuidExternalValuesMap(const std::unordered_map<uint32_t, xnn_external_value> &ext_vals) {
    uuid_2_externals_v_ = ext_vals;
}

void XnnpackModelRuntime::setWeightCache(xnn_weights_cache_t weight_cache) {
    weight_cache_ = weight_cache;
}

xnn_runtime_t XnnpackModelRuntime::getXnnRt() {
    return runtime_;
}

std::unordered_map<uint32_t, xnn_external_value> &XnnpackModelRuntime::__uuidToExternalsV() {
    return uuid_2_externals_v_;
}

XnnpackBackend::XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts) :
    Backend(mm), opts_(opts) {
    // init weight_cache_
    // xnn_create_weights_cache(&weight_cache_);

    // register ops
    type_ = BackendType::MLLM_XNNPACK;
    registerOps();
    registerFuncs();
}

XnnpackBackend::~XnnpackBackend() {
}

bool XnnpackBackend::addCreator(OpType t, Creator *c) {
    if (map_op_creator_.count(t)) {
        Log::error("{} op has been added to this backend.", (int32_t)t);
        return false;
    }
    map_op_creator_.insert({t, c});
    return true;
}

Op *XnnpackBackend::opCreate(const OpParam &op_param, string name, int thread_count) {
    auto op_type = OpType(op_param.find("type")->second);
    auto iter = map_op_creator_.find(op_type);

    if (thread_count) {
        Log::warn("Xnnpack use global thread pool. thread_count is set to {}, but not used.", thread_count);
    }

    if (iter == map_op_creator_.end()) {
        Log::error("OpType={}, Name={} is not supported yet.", int(op_param.find("type")->second), name);
        return nullptr;
    }
    auto op = iter->second->create(op_param, this, name, thread_count);
    return op;
}

TensorFunction *XnnpackBackend::funcCreate(TensorFuncType type) {
    auto iter = map_tensor_function_.find(type);
    if (iter == map_tensor_function_.end()) {
        Log::error("Xnnpack backend don't support func type {}", (int32_t)type);
        return nullptr;
    }
    return iter->second;
}

void XnnpackBackend::registerOps() {
    addCreator(D2H, new XpD2HCreator());
    addCreator(ADD, new XpAddCreator());
    addCreator(DIRECT, new XpDirectCreator());
    addCreator(DISPATCH, new XpDispatchCreator());
    addCreator(SUBGRAPHSTART, new XpSubGraphStartCreator());
    addCreator(SUBGRAPHFINALIZE, new XpSubGraphFinalizeCreator());
    addCreator(LINEAR, new XpLinearCreator());
    addCreator(MATMUL, new XpMatMulCreator());
    addCreator(ROPE, new XpRoPECreator());
    addCreator(RELU, new XpReLUCreator());
    addCreator(SOFTMAX, new XpSoftmaxCreator());
    addCreator(OP_GELU, new XpGeLUCreator());
    addCreator(SILU, new XpSiLUCreator());
    addCreator(TRANSPOSE, new XpTransposeCreator());
    addCreator(RMSNORM, new XpRMSNormCreator());
    addCreator(XP_KVCACHE, new XpKVCacheCreator());
    addCreator(CAUSALMASK, new XpCausalMaskCreator());
    addCreator(SDPA, new XpSDPACreator());
    addCreator(EMBEDDING, new XpEmbeddingCreator());
    addCreator(PARAMETER, new XpParameterCreator());
}

void XnnpackBackend::registerFuncs() {
    // broadcast element wise tensor func
    map_tensor_function_[TensorFuncType::FUNC_ADD] = new XpBroadcastAddFunction();
    map_tensor_function_[TensorFuncType::FUNC_SUB] = new XpBroadcastSubFunction();
    map_tensor_function_[TensorFuncType::FUNC_MUL] = new XpBroadcastMulFunction();
    map_tensor_function_[TensorFuncType::FUNC_DIV] = new XpBroadcastDivFunction();

    // element wise tensor func
    map_tensor_function_[TensorFuncType::FUNC_TTADD] = new XpTTAddFunction();
    map_tensor_function_[TensorFuncType::FUNC_TTSUB] = new XpTTSubFunction();
    map_tensor_function_[TensorFuncType::FUNC_TTMUL] = new XpTTMulFunction();
    map_tensor_function_[TensorFuncType::FUNC_TTDIV] = new XpTTDivFunction();

    // others
    map_tensor_function_[TensorFuncType::FUNC_TRANPOSE] = new XpTransposeFunction();
    map_tensor_function_[TensorFuncType::FUNC_VIEW] = new XpViewFunction();
    map_tensor_function_[TensorFuncType::FUNC_MM] = new XpMatmulFunction();
}

uint32_t XnnpackCargo::getExecCnt() {
    return exec_cnt_;
}

uint32_t XnnpackCargo::incExecCnt() {
    exec_cnt_++;
    return exec_cnt_;
}

void XnnpackCargo::setThreadPool(pthreadpool_t tp) {
    threadpool_ = tp;
}

std::shared_ptr<XnnpackModelRuntime> XnnpackCargo::getModelRuntime() {
    return model_runtime_;
}

std::shared_ptr<XnnpackModelRuntime> XnnpackCargo::recreateModelRuntime() {
    model_runtime_ = std::make_shared<XnnpackModelRuntime>(threadpool_);

    // set external values
    model_runtime_->resetUuidExternalValuesMap(uuid_2_externals_v_);
    model_runtime_->setWeightCache(weight_cache_);

    return model_runtime_;
}

xnn_subgraph_t XnnpackCargo::getXnnSubgraph() {
    return graph_;
}

void XnnpackCargo::createSubgraph(int32_t external_nums) {
    if (graph_) {
        Log::error("The subgraph has already been created. Use recreateSubGraph instead.");
        exit(-1);
    }

    uuid_2_externals_v_.clear();
    uuid_2_mllm_tensor_.clear();
    uuid_2_mllm_weight_tensor_.clear();
    uuid_2_normal_tensor_.clear();
    activation_name_2_uuid_.clear();
    auto status = xnn_create_subgraph(external_nums, 0, &graph_);
    if (status != xnn_status_success) {
        Log::error("Failed to create subgrpah");
        exit(-1);
    }
}

void XnnpackCargo::recreateSubgraph(int32_t external_nums) {
    if (graph_) {
        // no need to delete this, the previous xnnpack runtime will manage it.
        // xnn_delete_subgraph(subgraph_);
        uuid_2_mllm_tensor_.clear();
        uuid_2_mllm_weight_tensor_.clear();
        uuid_2_externals_v_.clear();
        uuid_2_normal_tensor_.clear();
        activation_name_2_uuid_.clear();
    }

    auto status = xnn_create_subgraph(external_nums, 0, &graph_);
    if (status != xnn_status_success) {
        Log::error("Failed to create subgrpah");
        exit(-1);
    }
}

void XnnpackCargo::registerExternalValue(uint32_t uuid, const xnn_external_value &ext_v) {
    if (uuid_2_externals_v_.count(uuid)) {
        Log::error("when reigster a external value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_externals_v_.insert({uuid, ext_v});
}

void XnnpackCargo::updateExternalValue(uint32_t uuid, const xnn_external_value &ext_v) {
    if (!uuid_2_externals_v_.count(uuid)) {
        Log::error("when update a external value, found exists uuid: {}", uuid);
        exit(-1);
    }
    uuid_2_externals_v_[uuid] = ext_v;
}

void XnnpackCargo::registerNormalValue(uint32_t uuid) {
    if (uuid_2_normal_tensor_.count(uuid)) {
        Log::error("when reigster a normal value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_normal_tensor_.insert({uuid, true});
}

void XnnpackCargo::registerUuidTensor(uint32_t uuid, Tensor *t) {
    if (uuid_2_mllm_tensor_.count(uuid)) {
        Log::error("when reigster a tensor value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_mllm_tensor_.insert({uuid, t});
}

void XnnpackCargo::updateUuidTensor(uint32_t uuid, Tensor *t) {
    if (!uuid_2_mllm_tensor_.count(uuid)) {
        Log::error("XnnpackCargo::updateUuidTensor failed. {} is not exists", uuid);
        exit(-1);
    }

    uuid_2_mllm_tensor_[uuid] = t;
}

void XnnpackCargo::registerUuidWeightTensor(uint32_t uuid, Tensor *t) {
    if (uuid_2_mllm_weight_tensor_.count(uuid)) {
        Log::error("when reigster a weight tensor value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_mllm_weight_tensor_.insert({uuid, t});
}

void *XnnpackCargo::getExternalValueptr(uint32_t uuid) {
    if (uuid_2_externals_v_.count(uuid)) {
        return uuid_2_externals_v_[uuid].data;
    }
    Log::error("getExternalValueptr return nullptr for uuid: {}", uuid);
    return nullptr;
}

bool XnnpackCargo::hasExternalValue(uint32_t uuid) {
    return uuid_2_externals_v_.count(uuid);
}

bool XnnpackCargo::hasNormalValue(uint32_t uuid) {
    return uuid_2_normal_tensor_.count(uuid);
}

bool XnnpackCargo::hasWeightValue(uint32_t uuid) {
    return uuid_2_mllm_weight_tensor_.count(uuid);
}

xnn_datatype XnnpackBackend::mllmDType2XnnDType(DataType mllm_dtype) {
    switch (mllm_dtype) {
    case MLLM_TYPE_F32:
        return xnn_datatype_fp32;
    case MLLM_TYPE_F16:
        return xnn_datatype_fp16;
    case MLLM_TYPE_I32:
        return xnn_datatype_int32;
    default:
        return xnn_datatype_invalid;
    }
    return xnn_datatype_invalid;
}

uint32_t XnnpackCargo::getNewEXternalId() {
    return (uint32_t)uuid_2_externals_v_.size();
}

void XnnpackCargo::assignPtrToTensor() {
    // update from runtime
    uuid_2_externals_v_ = getModelRuntime()->__uuidToExternalsV();

    // for (auto &iter : uuid_2_mllm_tensor_) {
    //     auto t = iter.second;
    //     auto uuid = iter.first;
    //     auto ext_v = uuid_2_externals_v_[uuid];
    //     t->forceResetHostPointer(ext_v.data);
    // }

    for (auto &iter : uuid_2_mllm_weight_tensor_) {
        iter.second->uuid() = XNN_INVALID_VALUE_ID;
    }
}

void XnnpackCargo::setSubgraphDispatched(bool b) {
    subgraph_dispatched_ = b;
}

xnn_weights_cache_t XnnpackCargo::getWeightCache() {
    return weight_cache_;
}

bool XnnpackCargo::isWeightCacheFinalized() const {
    return weight_cache_finalized_;
}

void XnnpackCargo::setWeightCacheFinalized(bool b) {
    weight_cache_finalized_ = b;
}

bool XnnpackCargo::inActivationName(const std::string &name) {
    return activation_name_2_uuid_.count(name);
}

uint32_t XnnpackCargo::getUUIDByActivationName(const std::string &name) {
    if (inActivationName(name)) return activation_name_2_uuid_[name];
    Log::error("XnnpackCargo::getUUIDByActivationName, {} not in activation name", name);
    exit(-1);
}

void XnnpackCargo::registerActivationNameAndUUID(const std::string &name, uint32_t uuid) {
    if (inActivationName(name)) {
        Log::error("XnnpackCargo::registerActivationNameAndUUID, {} already exists", name);
        exit(-1);
    }
    activation_name_2_uuid_.insert({name, uuid});
}

void XnnpackBackend::createNewGraph(const std::string &name) {
    if (graphs_.count(name)) {
        if (enable_dynamic_shape) {
            Log::error("XnnpackBackend::createNewGraph, {} graph already exists", name);
            exit(-1);
        } else {
            graphs_.erase(graphs_.find(name));
        }
    }

    graphs_.insert({name, std::make_shared<XnnpackCargo>()});
    graphs_[name]->setThreadPool(threadpool_);
    graphs_[name]->createSubgraph();
}

std::shared_ptr<XnnpackCargo> XnnpackBackend::getGraph(const std::string &name) {
    if (!graphs_.count(name)) {
        Log::error("XnnpackBackend::getGraph, {} graph not exists");
        exit(-1);
    }
    return graphs_[name];
}

bool XnnpackBackend::hasGraph(const std::string &name) {
    return graphs_.count(name);
}

void XnnpackBackend::onSetUpStart(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs, std::string graph_name) {
    // 0. create graph
    cur_processing_graph_name_ = graph_name;

    if (!hasGraph(graph_name) || !XnnpackBackend::enable_dynamic_shape) {
        createNewGraph(graph_name);
        auto cargo = getGraph(graph_name);

        if (XnnpackBackend::enable_legacy_wrapper) Backend::onSetUpStart(inputs, outputs, graph_name);

        // 1. register all inputs
        for (auto &t : inputs) {
            auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

            xnn_status status;
            std::vector<size_t> dims;
            for (auto d : t->shape()) dims.push_back(d);

            uint32_t flags = XNN_VALUE_FLAG_EXTERNAL_INPUT;
            uint32_t external_id = cargo->getNewEXternalId();

            switch (xp_dtype) {
            case xnn_datatype_fp32: {
                status = xnn_define_tensor_value(
                    cargo->getXnnSubgraph(), xp_dtype,
                    dims.size(), dims.data(),
                    /*data=*/nullptr,
                    external_id, flags, &t->uuid());
                break;
            }
            default:
                Log::error("XnnpackBackend::onSetUpStart, Unsupported datatype.");
                break;
            }

            cargo->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            cargo->registerUuidTensor(t->uuid(), t.get());
            cargo->registerActivationNameAndUUID(t->name(), t->uuid());

            if (status != xnn_status_success) {
                Log::error("xnnpack backend defineXpTensor Error");
                exit(-1);
            }
        }
    } else {
        // do not create a new graph. Reuse already exists runtime
        auto cargo = getGraph(graph_name);

        for (auto &t : inputs) {
            t->uuid() = cargo->getUUIDByActivationName(t->name());
            cargo->updateExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            cargo->updateUuidTensor(t->uuid(), t.get());
            std::vector<size_t> dims;
            for (auto d : t->shape()) dims.push_back(d);
            xnn_reshape_external_value(cargo->getModelRuntime()->getXnnRt(), t->uuid(), dims.size(), dims.data());
        }
    }
}

void XnnpackBackend::onSetUpEnd(std::vector<shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs, std::string graph_name) {
    cur_processing_graph_name_ = graph_name;

    if (getGraph(graph_name)->getExecCnt() == 0 || !XnnpackBackend::enable_dynamic_shape) {
        if (XnnpackBackend::enable_legacy_wrapper) Backend::onSetUpEnd(inputs, outputs, graph_name);

        // 0. get graph
        auto cargo = getGraph(graph_name);

        // 1. register all outputs
        for (auto &t : outputs) {
            auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

            xnn_status status;
            std::vector<size_t> dims;
            for (auto d : t->shape()) dims.push_back(d);

            uint32_t flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
            uint32_t external_id = cargo->getNewEXternalId();

            switch (xp_dtype) {
            case xnn_datatype_fp32: {
                status = xnn_define_tensor_value(
                    cargo->getXnnSubgraph(), xp_dtype,
                    dims.size(), dims.data(),
                    /*data=*/nullptr,
                    external_id, flags, &t->uuid());
                break;
            }
            default:
                break;
            }

            cargo->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            cargo->registerUuidTensor(t->uuid(), t.get());
            cargo->registerActivationNameAndUUID(t->name(), t->uuid());

            if (status != xnn_status_success) {
                Log::error("xnnpack backend defineXpTensor Error");
                exit(-1);
            }
        }
    } else {
        // do not create a new graph. Reuse already exists runtime
        auto cargo = getGraph(graph_name);

        for (auto &t : outputs) {
            t->uuid() = cargo->getUUIDByActivationName(t->name());
            cargo->updateExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            cargo->updateUuidTensor(t->uuid(), t.get());
            std::vector<size_t> dims;
            for (auto d : t->shape()) dims.push_back(d);
            xnn_reshape_external_value(cargo->getModelRuntime()->getXnnRt(), t->uuid(), dims.size(), dims.data());
        }
    }
}

void XnnpackBackend::onExecuteStart(std::vector<shared_ptr<Tensor>> &inputs, std::vector<shared_ptr<Tensor>> &outputs, std::string graph_name) {
    cur_processing_graph_name_ = graph_name;
}

void XnnpackBackend::onExecuteEnd(std::vector<std::shared_ptr<Tensor>> &outputs, const string &graph_name) {
    cur_processing_graph_name_ = graph_name;
    auto cargo = getCurProcessingGraph();

    if (getGraph(graph_name)->getExecCnt() == 0 || !XnnpackBackend::enable_dynamic_shape) {
        // recreate runtime
        auto m_rt = cargo->recreateModelRuntime();

        // create Model
        m_rt->createModel(cargo->getXnnSubgraph());

        // create runtime
        m_rt->createRuntime(0);

        // auto wc = xnnbk->getWeightCache();
        // if (!xnnbk->isWeightCacheFinalized()) {
        //     xnn_finalize_weights_cache(wc, xnn_weights_cache_finalization_kind_hard);
        //     xnnbk->setWeightCacheFinalized(true);
        // }

        // reshape
        m_rt->reshapeRuntime();

        // setup
        m_rt->setupRuntime();

        // run
        if (!m_rt->invoke()) {
            Log::error("XnnpackBackend::onExecuteStart xnn invoke failed");
            return;
        }

        // update all output's ptr
        cargo->assignPtrToTensor();

        cargo->setSubgraphDispatched(true);

        cargo->incExecCnt();
    } else {
        // recreate runtime
        auto m_rt = cargo->getModelRuntime();

        // setup
        m_rt->setupRuntime();

        // run
        if (!m_rt->invoke()) {
            Log::error("XnnpackBackend::onExecuteStart xnn invoke failed");
            return;
        }

        // update all output's ptr
        cargo->assignPtrToTensor();

        cargo->setSubgraphDispatched(true);

        cargo->incExecCnt();
    }

    for (auto &o : outputs) {
        o->forceResetHostPointer(getCurProcessingGraph()->getExternalValueptr(o->uuid()));
        o->uuid() = XNN_INVALID_VALUE_ID;
    }
}

XnnpackCargo *XnnpackBackend::getCurProcessingGraph() {
    if (!graphs_.count(cur_processing_graph_name_)) {
        Log::error("XnnpackBackend::getCurProcessingGraph, {} graph not exists");
        exit(-1);
    }
    return graphs_[cur_processing_graph_name_].get();
}

int XnnpackBackend::xnn_threads = 4;

bool XnnpackBackend::enable_dynamic_shape = true;

bool XnnpackBackend::enable_legacy_wrapper = false;

} // namespace mllm::xnnpack
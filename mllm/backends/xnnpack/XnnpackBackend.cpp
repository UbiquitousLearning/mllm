#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Backend.hpp"
#include "Module.hpp"
#include "layer.hpp"
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
#include <regex>

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

// std::vector<Tensor> XnnpackBackend::runFunc(std::vector<std::string> out_names,
//                                             TensorFuncType type,
//                                             std::vector<float> float_args,
//                                             std::vector<Tensor> input_tensors,
//                                             bool in_place) {
//     Module *module = input_tensors.empty() ? mllm::Module::llm_model_ptr : input_tensors[0].module();
//     assert(module != nullptr);
//     auto &activation_tensors = module->activation_tensors;
//     auto &activation_tensors_num = module->activation_tensors_num;

//     std::vector<std::shared_ptr<Tensor>> output_ptrs;
//     for (const auto &out_name : out_names) {
//         if (activation_tensors.find(out_name) == activation_tensors.end()) {
//             Backend *backend_h = Backend::global_backends[MLLM_CPU].get();
//             if (!input_tensors.empty()) {
//                 backend_h = input_tensors[0].backend();
//             }
//             activation_tensors[out_name] = std::make_shared<Tensor>(backend_h);
//             activation_tensors[out_name]->setName(out_name);
//             activation_tensors[out_name]->setModule(module);
//             activation_tensors_num[out_name] = 0;
//         }
//         output_ptrs.push_back(activation_tensors[out_name]);
//     }

//     if (module->doLoad) {
//         std::vector<Tensor> results;
//         for (auto &out_tensor : output_ptrs) {
//             results.push_back(*activation_tensors[out_tensor->name()]);
//         }
//         return results;
//     }

//     Backend *backend_h = Backend::global_backends[MLLM_CPU].get();
//     if (!input_tensors.empty()) {
//         backend_h = input_tensors[0].backend();
//     }
//     TensorFunction *func = backend_h->funcCreate(type);

//     std::vector<std::shared_ptr<Tensor>> input_ptrs;
//     for (auto &tensor : input_tensors) {
//         input_ptrs.push_back(activation_tensors[tensor.name()]);
//     }
//     // if (in_place) {
//     //     for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
//     //         input_tensors[i].setName(out_names[i]);
//     //         output_ptrs.push_back(input_tensors[i]);
//     //     }
//     // }

// #ifdef DEBUGOPTIME
//     auto start_t = mllm_time_us();
// #endif

//     switch (Tensor::tensor_status) {
//     case TENSOR_STATIC_INIT:
//         func->reshape(output_ptrs, input_ptrs, float_args);
//         func->setUp(output_ptrs, input_ptrs, float_args);
//         break;
//     case TENSOR_STATIC_READY:
//         func->execute(output_ptrs, input_ptrs, float_args);
//         break;
//     case TENSOR_STATIC_TRACE:
//         if (backend_h->type() == BackendType::MLLM_CPU) {
//             Tracer::addTensorFunction(func, input_ptrs, output_ptrs, float_args);
//         }
//         break;
//     default:
//         break;
//     }

//     // if (Backend::global_backends.size() == 1) {
//     //     for (auto input_tensor : input_ptrs) {
//     //         auto it = activation_tensors_num.find(input_tensor->name());
//     //         if (it != activation_tensors_num.end()) {
//     //             switch (Tensor::tensor_status) {
//     //             case TENSOR_STATIC_INIT:
//     //                 it->second += 1;
//     //                 break;
//     //             case TENSOR_STATIC_READY:
//     //                 it->second -= 1;
//     //                 break;
//     //             default:
//     //                 break;
//     //             }
//     //             if (it->second == 0 && module_tensors[input_tensor->name()]->sequence() > 1 && module_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
//     //                 activation_tensors[input_tensor->name()]->free();
//     //             }
//     //         }
//     //     }
//     // }

// #ifdef DEBUGOPTIME
//     if (Tensor::tensor_status == TENSOR_STATIC_READY) {
//         auto end_t = mllm_time_us();
//         std::cout << (out_names.empty() ? "" : out_names[0]) << " | "
//                   << Tensor::tensor_status << " time: "
//                   << (end_t - start_t) / 1000.0F << "ms" << std::endl;
//     }
// #endif

// #ifdef DEBUGSAVETENSOR
//     for (auto &out_name : out_names) {
//         activation_tensors[out_name]->saveNData<float>();
//     }
// #endif

//     std::vector<Tensor> results;
//     for (auto &out_tensor : output_ptrs) {
//         results.emplace_back(*activation_tensors[out_tensor->name()]);
//     }
//     return results;
// }
std::string name_num_to_X(const std::string &input_string) {
    std::regex pattern(R"(\.\d{1,3}\.)"); // Matches any number between 1 and 100 between two dots
    std::string replacement = ".X.";      // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
std::string name_X_to_num(const std::string &input_string, int in_idx) {
    std::regex pattern(".X.");                                    // Matches any number between 1 and 100 between two dots
    std::string replacement = "." + std::to_string(in_idx) + "."; // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
void init_reset_KVCache(string input_name, Module *module, int saved_list_idx, map<string, string> layername_2_tensorname, Backend *backend_) {
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    vector<string> renameX_names;
    renameX_names.push_back(input_name);
    const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2", "-cat", "-split-0-48"};
    vector<string> new_names;
    bool can_break = true;
    auto in_x_name = renameX_names[0];
    while (can_break) {
        can_break = false;
        for (const auto &suffix : suffixs) {
            if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end() && std::find(new_names.begin(), new_names.end(), r_name) == new_names.end()) {
                    new_names.push_back(r_name);
                    in_x_name = r_name;
                    can_break = true;
                }
                break;
            }
        }
    }
    renameX_names.insert(renameX_names.end(), new_names.begin(), new_names.end());
    for (const auto x_name : renameX_names) {
        auto name = name_X_to_num(x_name, saved_list_idx);
        layername_2_tensorname[name] = name;
        activation_tensors[name] = std::make_shared<Tensor>(backend_);
        activation_tensors[name]->initFrom(*activation_tensors[x_name]);
        activation_tensors[name]->setName(name);
        activation_tensors[name]->setModule(module);
    }
}

std::vector<Tensor> XnnpackBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;
    // Module::runlistIdx = saved_list_idx;
    bool do_init = false;

    if (module->doLoad || !layer->inited_loaded) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        layer->backend_ = Backend::global_backends[Module::tmp_device];
        do_init = !layer->inited_loaded;
        // if (layer->op_ == nullptr) {
        //     layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
        // }
        if (layer->param_["type"] == SUBGRAPHFINALIZE) {
            for (auto &input : inputs) {
                activation_tensors[input.name()]->setTtype(GRAPH_OUTPUT);
            }
        }
        // if (module->doLoad) {
        //     layer->op_->load(*module->loader);
        //     layer->inited_loaded = true;
        // } else if (layer->loaded_param) {
        //     layer->inited_loaded = layer->loaded_param;
        // } else {
        //     if (!layer->inited_loaded) {
        //         // module->loader = new ParamLoader("");
        //         // op_->load(*module->loader);
        //         auto empty_loader = new ParamLoader("");
        //         layer->op_->load(*empty_loader);
        //         layer->inited_loaded = true;
        //     }
        // }
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + layer->op_->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + layer->op_->name()};
        }
        for (const auto &layer_next_name : layer_next_names) {
            string next_name;
            if (Layer::use_layername_2_tensorname) {
                if (Layer::layername_2_tensorname.find(layer_next_name) == Layer::layername_2_tensorname.end()) {
                    if (layer->param_["type"] == KVCACHE) {
                        Layer::layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, layer->saved_list_idx, Layer::layername_2_tensorname, layer->backend_);
                    } else {
                        Layer::layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = Layer::layername_2_tensorname[layer_next_name];
            } else if (layer_next_name.find("visual") != string::npos) {
                // QNN VLM trick: visual model use act tensor sharing
                if (Layer::layername_2_tensorname.find(layer_next_name) == Layer::layername_2_tensorname.end()) {
                    if (layer->param_["type"] == KVCACHE) {
                        Layer::layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, layer->saved_list_idx, Layer::layername_2_tensorname, layer->backend_);
                    } else {
                        Layer::layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = Layer::layername_2_tensorname[layer_next_name];
            } else {
                next_name = layer_next_name;
            }
            if (activation_tensors.find(next_name) == activation_tensors.end()) {
                activation_tensors[next_name] = std::make_shared<Tensor>(layer->backend_);
                activation_tensors[next_name]->setName(next_name);
                activation_tensors[next_name]->setModule(module);
                activation_tensors_num[next_name] = 0;
            }
        }
        if (module->doLoad) {
            vector<Tensor> output_result = {};
            for (const auto &layer_next_name : layer_next_names) {
                string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
                output_result.push_back(*activation_tensors[next_name]);
            }
            return output_result;
        }
    }
    // input_tensors
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        if (input.shouldInGraphs()) {
            auto input_name = input.name();
            if (layer->param_["type"] == KVCACHE && do_init && Layer::use_layername_2_tensorname) {
                input_name = name_X_to_num(input_name, layer->saved_list_idx);
            }
            input_tensors.push_back(activation_tensors[input_name]);
        } else {
            input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
    }
    // output_tensors
    vector<string> layer_next_names = {};
    if (N > 1) {
        for (int i = 0; i < N; ++i) {
            layer_next_names.push_back("out-" + layer->op_->name() + "-" + std::to_string(i));
        }
    } else {
        layer_next_names = {"out-" + layer->op_->name()};
    }
    vector<shared_ptr<Tensor>> output_tensors = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
        output_tensors.push_back(activation_tensors[next_name]);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        if (!Module::isFirstChunk && layer->backend_->type() == MLLM_QNN) {
        } else {
            layer->op_->reshape(input_tensors, output_tensors);
            layer->op_->setUp(input_tensors, output_tensors);
        }
        break;
    }
    case TENSOR_STATIC_READY: {
        if (!Module::isFirstChunk && layer->backend_->type() == MLLM_QNN && layer->param_["type"] != SUBGRAPHSTART) {
        } else {
            layer->op_->execute(input_tensors, output_tensors);
        }
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (layer->backend_->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(layer->op_, input_tensors, output_tensors);
        } else if (layer->param_["type"] == SUBGRAPHSTART) { // begin of QNN graph
            Tracer::addModule(input_tensors, {}, layer->op_->name());
        }
        break;
    }
    default: {
        break;
    }
    }
// if (Backend::global_backends.size() == 1) {
//     for (auto input_tensor : input_tensors) {
//         if ((activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end())) {
//             switch (Tensor::tensor_status) {
//             case TENSOR_STATIC_INIT: {
//                 activation_tensors_num[input_tensor->name()] += 1;
//                 break;
//             }
//             case TENSOR_STATIC_READY: {
//                 activation_tensors_num[input_tensor->name()] -= 1;
//                 break;
//             }
//             default: {
//             }
//             }
//             if (activation_tensors_num[input_tensor->name()] == 0 && activation_tensors[input_tensor->name()]->sequence() > 1
//                 && activation_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
//                 activation_tensors[input_tensor->name()]->free();
//                 // std::cout << input_tensor->name() << "|" << std::endl;
//             }
//         }
//     }
// }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << layer->op_->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
    vector<Tensor> output_result = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
#ifdef DEBUGSAVETENSOR
        activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
        output_result.push_back(*activation_tensors[next_name]);
    }
    return output_result;
}

std::vector<Tensor> XnnpackBackend::runOp(Op *op, std::vector<Tensor> inputs, std::vector<std::string> out_names, bool in_place) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;
    // Module::runlistIdx = saved_list_idx;
    bool do_init = false;

    if (module->doTrace) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        op->backend() = Backend::global_backends[Module::tmp_device];
        // do_init = !layer->inited_loaded;
        // if (layer->op_ == nullptr) {
        //     layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
        // }
        if (op->type() == SUBGRAPHFINALIZE) {
            for (auto &input : inputs) {
                activation_tensors[input.name()]->setTtype(GRAPH_OUTPUT);
            }
        }
        // if (module->doLoad) {
        //     layer->op_->load(*module->loader);
        //     layer->inited_loaded = true;
        // } else if (layer->loaded_param) {
        //     layer->inited_loaded = layer->loaded_param;
        // } else {
        //     if (!layer->inited_loaded) {
        //         // module->loader = new ParamLoader("");
        //         // op_->load(*module->loader);
        //         auto empty_loader = new ParamLoader("");
        //         layer->op_->load(*empty_loader);
        //         layer->inited_loaded = true;
        //     }
        // }
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + op->name()};
        }
        for (const auto &layer_next_name : layer_next_names) {
            string next_name;
            if (Layer::use_layername_2_tensorname) {
                if (Layer::layername_2_tensorname.find(layer_next_name) == Layer::layername_2_tensorname.end()) {
                    if (layer->param_["type"] == KVCACHE) {
                        Layer::layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, layer->saved_list_idx, Layer::layername_2_tensorname, layer->backend_);
                    } else {
                        Layer::layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = Layer::layername_2_tensorname[layer_next_name];
            } else if (layer_next_name.find("visual") != string::npos) {
                // QNN VLM trick: visual model use act tensor sharing
                if (Layer::layername_2_tensorname.find(layer_next_name) == Layer::layername_2_tensorname.end()) {
                    if (layer->param_["type"] == KVCACHE) {
                        Layer::layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, layer->saved_list_idx, Layer::layername_2_tensorname, layer->backend_);
                    } else {
                        Layer::layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = Layer::layername_2_tensorname[layer_next_name];
            } else {
                next_name = layer_next_name;
            }
            if (activation_tensors.find(next_name) == activation_tensors.end()) {
                activation_tensors[next_name] = std::make_shared<Tensor>(op->backend());
                activation_tensors[next_name]->setName(next_name);
                activation_tensors[next_name]->setModule(module);
                activation_tensors_num[next_name] = 0;
            }
        }
        vector<Tensor> output_result = {};
        for (const auto &layer_next_name : layer_next_names) {
            string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
            output_result.push_back(*activation_tensors[next_name]);
        }
        return output_result;
    }
    // input_tensors
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        if (input.shouldInGraphs()) {
            auto input_name = input.name();
            if (layer->param_["type"] == KVCACHE && do_init && Layer::use_layername_2_tensorname) {
                input_name = name_X_to_num(input_name, layer->saved_list_idx);
            }
            input_tensors.push_back(activation_tensors[input_name]);
        } else {
            input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
    }
    // output_tensors
    vector<string> layer_next_names = {};
    if (N > 1) {
        for (int i = 0; i < N; ++i) {
            layer_next_names.push_back("out-" + op->name() + "-" + std::to_string(i));
        }
    } else {
        layer_next_names = {"out-" + op->name()};
    }
    vector<shared_ptr<Tensor>> output_tensors = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
        output_tensors.push_back(activation_tensors[next_name]);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        op->reshape(input_tensors, output_tensors);
        op->setUp(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_READY: {
        op->execute(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (op->backend()->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(op, input_tensors, output_tensors);
        } else if (op->type() == SUBGRAPHSTART) { // begin of QNN graph
            Tracer::addModule(input_tensors, {}, op->name());
        }
        break;
    }
    default: {
        break;
    }
    }
    if (Backend::global_backends.size() == 1) {
        for (auto input_tensor : input_tensors) {
            if ((activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end())) {
                switch (Tensor::tensor_status) {
                case TENSOR_STATIC_INIT: {
                    activation_tensors_num[input_tensor->name()] += 1;
                    break;
                }
                case TENSOR_STATIC_READY: {
                    activation_tensors_num[input_tensor->name()] -= 1;
                    break;
                }
                default: {
                }
                }
                if (activation_tensors_num[input_tensor->name()] == 0 && activation_tensors[input_tensor->name()]->sequence() > 1
                    && activation_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
                    activation_tensors[input_tensor->name()]->free();
                    // std::cout << input_tensor->name() << "|" << std::endl;
                }
            }
        }
    }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << op->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
    vector<Tensor> output_result = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : (layer_next_name.find("visual") != string::npos ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name);
#ifdef DEBUGSAVETENSOR
        activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
        output_result.push_back(*activation_tensors[next_name]);
    }
    return output_result;
}
std::vector<Tensor> XnnpackBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    // set static tmp_device to device_ to init layers' op
    // auto previoud_device = Module::tmp_device;
    // Module::tmp_device = module->device_;
    // Module Loading
    if (Module::llm_model_ptr && Module::llm_model_ptr->doLoad) {
        auto outputs = module->Forward(inputs, args);
        // for inner module, set output tensors to GRAPH_OUTPUT
        // if (inputs[0].ttype() != TensorType::INPUT_TENSOR) { // XPUs' module should not be the outermost input tensor
        //     for (auto &output : outputs) {
        //         inputs[0].module()->activation_tensors[output.name()]->setTtype(GRAPH_OUTPUT);
        //     }
        // }
        // // set Module::tmp_device to previous device
        // Module::tmp_device = previoud_device;
        return outputs;
    }
    // if (false) {
    //     inputs[0].setTtype(TensorType::INPUT_TENSOR);
    // }
    // Module setUp & execute
    if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
        if (module->prefilling_token_size_ == 0) { // first time init
            module->prefilling_token_size_ = inputs[0].sequence() * inputs[0].batch();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence() * inputs[0].batch();
        }
        for (int i = 0; i < inputs.size(); i++) {
            auto &input = inputs[i];
            input.setName("input" + std::to_string(i));
            input.setTtype(TensorType::NORMAL_TENSOR);
            module->activation_tensors[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
            module->activation_tensors[input.name()]->setName(input.name());
            module->activation_tensors[input.name()]->setModule(module);
        }
        Module::llm_model_ptr = module;
        Tensor::tensor_status = TENSOR_STATIC_INIT;

        uint64_t time_start = mllm_time_us();
        module->Forward(inputs, args);
        Tensor::tensor_status = TENSOR_STATIC_READY; // change to EAGER

        auto output = module->Forward(inputs, args);
        uint64_t time_end = mllm_time_us();

        double inference_time_ = (time_end - time_start) / 1000.0F; // ms
        module->inference_times_.push_back(inference_time_);

        Module::llm_model_ptr->op_transposed_flag = true;
        return output;
    } else { // inner Modules
        // offload according to the backends' info inited during loading
        if (Tensor::tensor_status == TENSOR_STATIC_INIT && module->device_ != MLLM_CPU) { // backend specific module reshape & setup
            if (Module::isMultiChunkPrefilling && !Module::isFirstChunk) {                // set to TENSOR_UNDEFINED and SKIP executing qnn layers
                Tensor::tensor_status = TENSOR_UNDEFINED;
                auto outputs = module->Forward(inputs, args);
                Tensor::tensor_status = TENSOR_STATIC_INIT;
                return outputs;
            }
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            Backend::global_backends[module->device_]->onSetUpStart(inputs_vec, outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &i : inputs) {
                i.uuid() = inputs[0].module()->activation_tensors[i.name()]->uuid();
            }

            auto outputs = module->Forward(inputs, args);
            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Backend::global_backends[module->device_]->onSetUpEnd(inputs_vec, outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &o : outputs) {
                o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
            }

            return outputs;
        } else if (Tensor::tensor_status == TENSOR_STATIC_READY && module->device_ != MLLM_CPU) { // backend specific module execute
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            auto outputs = module->Forward(inputs, args);

            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Backend::global_backends[module->device_]->onExecuteStart(inputs_vec, outputs_vec, module->getUniqueName());

            Backend::global_backends[module->device_]->onExecuteEnd(outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &o : outputs) {
                o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
                o.forceResetHostPointer(outputs[0].module()->activation_tensors[o.name()]->rawHostPtr());
            }

            return outputs;
        } else if (Tensor::tensor_status == TENSOR_STATIC_TRACE && module->device_ != MLLM_CPU) {
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            auto outputs = module->Forward(inputs, args);

            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Tracer::addModule(inputs_vec, outputs_vec, module->getUniqueName());
            return outputs;
        }
        return module->Forward(inputs, args);
    }
}

} // namespace mllm::xnnpack
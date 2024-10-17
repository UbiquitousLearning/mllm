#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
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
#include "xnnpack/allocator.h"
#include "xnnpack/subgraph.h"

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

XnnpackModelRuntime::XnnpackModelRuntime(int32_t num_threads) :
    num_threads_(num_threads), model_(nullptr, xnn_delete_subgraph) {
    xnn_delete_runtime(runtime_);
    threadpool_ = pthreadpool_create(num_threads_);
}

XnnpackModelRuntime::~XnnpackModelRuntime() {
    if (runtime_) {
        xnn_delete_runtime(runtime_);
    }
    if (threadpool_) {
        pthreadpool_destroy(threadpool_);
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
    return xnn_status_success == xnn_create_runtime_v4(model_.get(), nullptr, nullptr, threadpool_, flags, &runtime_);
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

std::unordered_map<uint32_t, xnn_external_value> &XnnpackModelRuntime::__uuidToExternalsV() {
    return uuid_2_externals_v_;
}

XnnpackBackend::XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts) :
    Backend(mm), opts_(opts) {
    // runtime
    model_runtime_ = std::make_shared<XnnpackModelRuntime>(opts_.num_threads);

    // subgraph
    createSubgraph();

    // register ops
    type_ = BackendType::MLLM_XNNPACK;
    registerOps();
    registerFuncs();
}

XnnpackBackend::~XnnpackBackend() {
    if (subgraph_) {
        xnn_delete_subgraph(subgraph_);
    }
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
        Log::error("Op is not supported yet.");
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
}

std::shared_ptr<XnnpackModelRuntime> XnnpackBackend::getModelRuntime() {
    return model_runtime_;
}

std::shared_ptr<XnnpackModelRuntime> XnnpackBackend::recreateModelRuntime(int thread_count) {
    model_runtime_ = std::make_shared<XnnpackModelRuntime>(thread_count);

    // set external values
    model_runtime_->resetUuidExternalValuesMap(uuid_2_externals_v_);

    return model_runtime_;
}

xnn_subgraph_t XnnpackBackend::getXnnSubgraph() {
    return subgraph_;
}

void XnnpackBackend::createSubgraph(int32_t external_nums) {
    if (subgraph_) {
        Log::error("The subgraph has already been created. Use recreateSubGraph instead.");
        exit(-1);
    }

    uuid_2_externals_v_.clear();
    uuid_2_mllm_tensor_.clear();
    uuid_2_mllm_weight_tensor_.clear();
    auto status = xnn_create_subgraph(external_nums, 0, &subgraph_);
    if (status != xnn_status_success) {
        Log::error("Failed to create subgrpah");
        exit(-1);
    }
}

void XnnpackBackend::recreateSubgraph(int32_t external_nums) {
    if (subgraph_) {
        // no need to delete this, the previous xnnpack runtime will manage it.
        // xnn_delete_subgraph(subgraph_);
        uuid_2_mllm_tensor_.clear();
        uuid_2_mllm_weight_tensor_.clear();
        uuid_2_externals_v_.clear();
    }

    auto status = xnn_create_subgraph(external_nums, 0, &subgraph_);
    if (status != xnn_status_success) {
        Log::error("Failed to create subgrpah");
        exit(-1);
    }
}

void XnnpackBackend::registerExternalValue(uint32_t uuid, const xnn_external_value &ext_v) {
    if (uuid_2_externals_v_.count(uuid)) {
        Log::error("when reigster a external value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_externals_v_.insert({uuid, ext_v});
}

void XnnpackBackend::registerUuidTensor(uint32_t uuid, Tensor *t) {
    if (uuid_2_mllm_tensor_.count(uuid)) {
        Log::error("when reigster a tensor value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_mllm_tensor_.insert({uuid, t});
}

void XnnpackBackend::registerUuidWeightTensor(uint32_t uuid, Tensor *t) {
    if (uuid_2_mllm_weight_tensor_.count(uuid)) {
        Log::error("when reigster a weight tensor value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_mllm_weight_tensor_.insert({uuid, t});
}

void *XnnpackBackend::getExternalValueptr(uint32_t uuid) {
    if (uuid_2_externals_v_.count(uuid)) {
        return uuid_2_externals_v_[uuid].data;
    }
    Log::error("getExternalValueptr return nullptr for uuid: {}", uuid);
    return nullptr;
}

bool XnnpackBackend::hasExternalValue(uint32_t uuid) {
    return uuid_2_externals_v_.count(uuid);
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

uint32_t XnnpackBackend::getNewEXternalId() {
    return (uint32_t)uuid_2_externals_v_.size();
}

void XnnpackBackend::assignPtrToTensor() {
    // update from runtime
    uuid_2_externals_v_ = getModelRuntime()->__uuidToExternalsV();

    for (auto &iter : uuid_2_mllm_tensor_) {
        auto t = iter.second;
        auto uuid = iter.first;
        auto ext_v = uuid_2_externals_v_[uuid];
        t->forceResetHostPointer(ext_v.data);
    }

    for (auto &iter : uuid_2_mllm_weight_tensor_) {
        iter.second->uuid() = XNN_INVALID_VALUE_ID;
    }
}

void XnnpackBackend::setSubgraphDispatched(bool b) {
    subgraph_dispatched_ = b;
}

} // namespace mllm::xnnpack
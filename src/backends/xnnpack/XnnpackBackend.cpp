#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Backend.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "xnnpack/allocator.h"
#include "xnnpack/subgraph.h"

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
    for (auto &i : external_values_) {
        xnn_release_simd_memory(i.data);
    }
}

bool XnnpackModelRuntime::createModel(const std::function<xnn_subgraph_t()> &model_factory) {
    model_.reset(model_factory());
    if (!model_) {
        Log::error("failed to create model");
        return false;
    }

    for (uint32_t i = 0; i < model_->num_values; ++i) {
        if ((model_->values[i].flags & ((uint32_t)XNN_VALUE_FLAG_EXTERNAL_INPUT | (uint32_t)XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
            continue;
        }

        // Make a buffer for this external value.
        size_t size = xnn_tensor_get_size(&model_->values[i]) + XNN_EXTRA_BYTES;
        external_values_.push_back(
            xnn_external_value{i, xnn_allocate_zero_simd_memory(size)});
    }

    return model_ != nullptr;
}

bool XnnpackModelRuntime::createRuntime(uint32_t flags) {
    assert(!runtime);
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

XnnpackBackend::XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts) :
    Backend(mm), opts_(opts) {
}

XnnpackBackend::~XnnpackBackend() {
    // TODO
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
    // TODO
}

void XnnpackBackend::registerOps() {
    // TODO
}

void XnnpackBackend::registerFuncs() {
    // TODO
}

} // namespace mllm::xnnpack
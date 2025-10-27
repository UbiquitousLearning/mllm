#include "Tensor.hpp"

#include <cassert>
#include <cstdlib>
// #include <exception>
#include "Backend.hpp"
#include "Op.hpp"
#include "OpDefined.hpp"
#include "TensorImpl.hpp"
// #include "Timing.hpp"
#include "Types.hpp"
#include <Module.hpp>
#include <iostream>
#include <memory>
// #include <regex>
#include <string>
#include <vector>

namespace mllm {

Tensor::Tensor(int batch, int head, int sequence, int dimension, Backend *bn, bool do_alloc) :
    impl_(std::make_shared<TensorImpl>(bn)) { // 使用带Backend的TensorImpl构造函数
    impl_->dtype_ = MLLM_TYPE_F32;
    reshape(batch, head, sequence, dimension);
    if (do_alloc) {
        alloc();
    }
}

Tensor::Tensor(int batch, int head, int sequence, int dimension, BackendType bn_type, bool do_alloc) :
    impl_(std::make_shared<TensorImpl>()) {
    if (Backend::global_backends.find(bn_type) == Backend::global_backends.end()) {
        Module::initBackend(bn_type);
    }
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type].get();
    reshape(batch, head, sequence, dimension);
    if (do_alloc) {
        alloc();
    }
}

Tensor::Tensor(const std::vector<int> &shape) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->private_reshape(shape);
}

Tensor::Tensor(int value, Backend *bn) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[MLLM_CPU].get();
    reshape(1, 1, 1, 1);
    alloc();
    impl_->should_in_graphs_ = false;
    setDataAt<float>(0, 0, 0, 0, static_cast<float>(value));
    to(bn->type());
}

Tensor::Tensor(int value, BackendType bn_type) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type].get();
    reshape(1, 1, 1, 1);
    alloc();
    impl_->should_in_graphs_ = false;
    setDataAt<float>(0, 0, 0, 0, static_cast<float>(value));
}

Tensor::Tensor(std::vector<float> values, BackendType bn_type) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type].get();
    reshape(1, 1, 1, values.size());
    alloc();
    impl_->should_in_graphs_ = false;
    for (size_t i = 0; i < values.size(); ++i) {
        setDataAt<float>(0, 0, 0, i, values[i]);
    }
}

bool Tensor::reshape(const int batch, const int head, const int sequence, const int dimension) {
    return impl_->reshape(batch, head, sequence, dimension);
    // vector<int> shape(4);
    // shape[chls()[BATCH]] = batch;
    // shape[chls()[HEAD]] = head;
    // shape[chls()[SEQUENCE]] = sequence;
    // shape[chls()[DIMENSION]] = dimension;
    // return reshape(shape);
}

// Tensor.cpp
void Tensor::alloc() {
    // if ("out-model.embed_tokens" == name())
    //     std::cout << "alloc " << name() << std::endl;
    if (aggregated_) return;
    assert(impl_->backend_ != nullptr);
    if (!master_tensor_.expired()) return;
    if (!shape_offset_.empty() && !shape_master_.empty()) return;

    impl_->alloc();
}

bool Tensor::reshape(int batch, int channel, int time, int height, int width) {
    if (impl_->ctype_ != BTHWC) {
        impl_->ctype_ = BCTHW;
        impl_->chls_[BATCH] = 0;
        impl_->chls_[CHANNLE] = 1;
        impl_->chls_[TIME] = 2;
        impl_->chls_[HEIGHT] = 3;
        impl_->chls_[WIDTH] = 4;
    } else {
        impl_->chls_[BATCH] = 0;
        impl_->chls_[TIME] = 1;
        impl_->chls_[HEIGHT] = 2;
        impl_->chls_[WIDTH] = 3;
        impl_->chls_[CHANNLE] = 4;
    }

    std::vector<int> shape(5);
    const auto &chls = impl_->chls_; // 从TensorImpl获取维度映射

    shape[chls.at(BATCH)] = batch;
    shape[chls.at(CHANNLE)] = channel;
    shape[chls.at(TIME)] = time;
    shape[chls.at(HEIGHT)] = height;
    shape[chls.at(WIDTH)] = width;

    return impl_->private_reshape(shape);
}

TensorStatus Tensor::tensor_status;

uint32_t &Tensor::uuid() {
    return uuid_;
}

TensorType &Tensor::xnnTensorType() {
    return xnn_tensor_type_;
}

void Tensor::forceResetHostPointer(void *ptr) {
    impl_->host_ptr_ = ptr;
}

Tensor &Tensor::to(BackendType backend_type) {
    // TODO: check if the data is shared between devices
    // if so, return the origin tensor
    // if not, return the new tensor
    // TODO: if need copy, should implement copyDataCrossBn and do copy when Tensor::TENSOR_STATIC_READY
    // if (Module::llm_model_ptr->doChangeBn) {
    //     Module::llm_model_ptr->device() = backend_type;
    // }
    /**
     * Currently, there are following cases:
     * CPU -> QNN, QNN -> CPU
     * if it is CPU -> QNN, the buffer should be realloced
     * (NOTE: not handling data copy as the tensor.to() shoudld be called before the data is set and tensor.device() should be checked in frontend)
     * if it is QNN -> CPU, the data is sharable between CPU and QNN, no need to copy or realloc
     */
    if (device() == backend_type) {
        return *this;
    }
    if (backend_type == MLLM_CPU && device() == MLLM_QNN) {
        // data is sharable between CPU and QNN
        return *this;
    }
    // realloc the tensor
    // realloc the tensor
    if (backend_type == MLLM_QNN && device() == MLLM_CPU) {
        if (this->masterTensor() != nullptr) {
            auto master_tensor = this->masterTensor();
            master_tensor->free();
            master_tensor->to(MLLM_QNN);
            master_tensor->alloc();
            for (auto &child_wp : master_tensor->childTensors()) {
                // Lock the weak_ptr to get a shared_ptr
                if (auto child_sp = child_wp.lock()) {
                    child_sp->forceResetHostPointer(this->impl_->host_ptr_);
                }
            }
        } else {
            this->free();
            module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type].get());
            this->setBackend(Backend::global_backends[backend_type].get());
        }
        return *this;
    }
    if (backend_type == MLLM_CPU && device() == MLLM_XNNPACK) {
        module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type].get());
        this->setBackend(Backend::global_backends[backend_type].get());
        return *this;
    }
    if (backend_type == MLLM_XNNPACK && device() == MLLM_CPU) {
        module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type].get());
        this->setBackend(Backend::global_backends[backend_type].get());
        return *this;
    }
    Backend *target_backend = Backend::global_backends[backend_type].get();
    if (target_backend == nullptr) {
        Module::initBackend(backend_type);
        target_backend = Backend::global_backends[backend_type].get();
        assert(target_backend != nullptr && "Target backend is not initialized.");
    }
    // {
    //     std::cout << name() << ", changing backend from " << device() << " to " << backend_type << std::endl; // debug log
    // }
    impl_->to(target_backend);
    return *this;
};

bool is_kvcached_tensor(const std::shared_ptr<Tensor> &tensor) {
    if (tensor == nullptr) return false;
    if (auto master = tensor->masterTensor()) { // 调用新的 masterTensor()
        return master->name().find("Cache") != std::string::npos;
    }
    return false;
}

/**
 * @brief Allocates a single, non-aggregated tensor, deciding between KVCache or standard allocation.
 * @param module The current module.
 * @param backend The current backend.
 * @param standard_alloc_func The function to call for standard allocation.
 */
void Tensor::_allocate_final_tensor(
    const std::shared_ptr<Tensor> &template_tensor,
    Backend *backend) {
    if (is_kvcached_tensor(template_tensor)) {
        if (auto master_tensor_sp = template_tensor->masterTensor()) {
            if (master_tensor_sp->name().find(".Cache") != std::string::npos && (master_tensor_sp->batch() != batch())) {
                KVCache_batch = batch();
                master_tensor_sp->reshape(KVCache_batch, master_tensor_sp->head(),
                                          master_tensor_sp->sequence(), master_tensor_sp->dimension());
                master_tensor_sp->setName(name() + ".Cache");
                master_tensor_sp->alloc();

                switch (master_tensor_sp->dtype()) {
                case MLLM_TYPE_F32:
                    memset(master_tensor_sp->hostPtr<float>(), 0, master_tensor_sp->count() * sizeof(float));
                    break;
                case MLLM_TYPE_F16:
                    memset(master_tensor_sp->hostPtr<mllm_fp16_t>(), 0, master_tensor_sp->count() * sizeof(mllm_fp16_t));
                    break;
                case MLLM_TYPE_Q8_0:
                    memset((char *)master_tensor_sp->rawHostPtr(), 0,
                           master_tensor_sp->count() * sizeof(block_q8_0) / QK8_0);
                    break;
                default:
                    break;
                };
            }
            auto cache_seq_len_ = template_tensor->shapeOffset()[2];

            if (name().find("cache") == std::string::npos) {
                cache_seq_len_ = master_tensor_sp->cache_seq_len_;
                auto cpu_backend = dynamic_cast<CPUBackend *>(backend);
                if (cpu_backend && cpu_backend->isUsingDraft()) {
                    unsigned int last_draft_length = cpu_backend->getLastDraftLength();
                    const auto &last_verified_position_ids = cpu_backend->getLastVerifiedPositionIds();
                    cache_seq_len_ = cache_seq_len_ - last_draft_length + last_verified_position_ids.size();
                }
            }
            setDtype(master_tensor_sp->dtype());
            shallowCopyFrom(master_tensor_sp, false, {0, 0, (int)cache_seq_len_, 0});
        } else {
            setDtype(template_tensor->dtype());
            alloc();
        }
    } else {
        if (template_tensor != nullptr) {
            setDtype(template_tensor->dtype());
        }
        alloc();
    }
}
/**
 * @brief Handles the allocation and setup for an output tensor that is part of an aggregated tensor structure.
 * @param template_tensor The corresponding tensor from the activation map, which holds aggregation info.
 * @param module The current module.
 * @param backend The current backend.
 */
void Tensor::_allocate_aggregated_tensor(
    const std::shared_ptr<Tensor> &template_tensor,
    Module *module,
    Backend *backend) {
    bool keep_aggregated_structure = false;
    if (template_tensor->aggregatedDim() > 3) {
        keep_aggregated_structure = true; // Cannot handle dimensions > 3
    } else {
        for (const auto &ag_tensor : template_tensor->aggregatedTensors()) {
            if (ag_tensor->ctype() != template_tensor->aggregatedTensors()[0]->ctype()) { //???我什么这么写？因为quant
                keep_aggregated_structure = true;
                break;
            }
        }
    }
    if (keep_aggregated_structure) {
        vector<shared_ptr<Tensor>> shared_outputs;
        auto split_dim = template_tensor->aggregatedDim();
        const auto &ag_tensor = template_tensor->aggregatedTensors();
        for (int id = 0; id < ag_tensor.size(); ++id) {
            const auto &child_tt = ag_tensor[id];
            auto shared_ot = std::make_shared<Tensor>(backend);
            // shared_ot->setName(out_tensor->name() + ".split-" + std::to_string(id));
            assert(child_tt->name() == name() + ".split-" + std::to_string(id));
            shared_ot->setName(child_tt->name());
            shared_ot->setModule(module);
            shared_ot->setCtype(child_tt->ctype());
            // Reshape based on the split dimension and the template tensor
            switch (split_dim) {
            case Chl::HEAD:
                shared_ot->reshape(batch(), child_tt->head(), sequence(), dimension());
                break;
            case Chl::SEQUENCE:
                shared_ot->reshape(this->batch(), head(), child_tt->sequence(), dimension());
                break;
            case Chl::DIMENSION:
                shared_ot->reshape(batch(), head(), sequence(), child_tt->dimension());
                break;
            case Chl::D_HD:
            case Chl::HD:
                shared_ot->reshape(batch(), child_tt->head(), sequence(), child_tt->dimension());
                break;
            default:
                break; // Should not happen
            }
            shared_ot->_allocate_final_tensor(child_tt, backend);
            shared_outputs.push_back(shared_ot);
        }
        addTensors(shared_outputs, split_dim);
    } else {
        allowAggregated() = false;
        alloc();
    }
}

/**
 * @brief Allocates memory for a tensor based on a template tensor.
 * If the template tensor is aggregated, it allocates an aggregated tensor.
 * Otherwise, it allocates a final tensor.
 * @param template_tensor The template tensor to base the allocation on.
 */
void Tensor::allocFromTemplate(shared_ptr<Tensor> template_tensor) {
    assert(backend() != nullptr);
    if (template_tensor != nullptr && !template_tensor->aggregatedTensors().empty()) {
        _allocate_aggregated_tensor(template_tensor, module(), backend());
    } else {
        _allocate_final_tensor(template_tensor, backend());
    }
}

/**
 * @brief Runs a tensor function with the specified parameters.
 * @param out_names The names for the output tensors.
 * @param type The type of the tensor function to run.
 * @param param The parameters for the tensor function.
 * @param input_tensors The input tensors to the function.
 * @param in_place Whether to run the function in-place.
 * @return A vector of output tensors.
 */
std::vector<Tensor> Tensor::runFunc(std::vector<std::string> out_names,
                                    OpType type,
                                    OpParam param,
                                    std::vector<Tensor> input_tensors,
                                    bool in_place) {
    // auto start_time = mllm_time_us();
    // ==================== [开始] Op 缓存 ====================
    if (!input_tensors.empty()) {
        for (auto &input : input_tensors) {
            assert(input.backend() == input_tensors[0].backend() && "All inputs must have the same backend.");
        }
    }
    auto backend = input_tensors.empty() ? Backend::global_backends[MLLM_CPU].get() : input_tensors[0].backend();
    //////////==============QNN only====================///////////
    if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) { // 针对QNN的特殊处理
        // backend = Backend::global_backends[MLLM_QNN].get();
        backend = Backend::global_backends[MLLM_CPU].get(); // 想不到吧
    }
    //////////==============QNN only====================///////////
    // 1. 使用更高效的键生成方式
    static std::unordered_map<size_t, std::shared_ptr<Op>> op_cache; // 改用size_t作为键类型
    param["type"] = type;
    std::shared_ptr<Op> op_to_run;
    // 2. 使用更高效的哈希键生成
    static auto hash_combine = [](size_t seed, const auto &v) {
        seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    };
    size_t key = std::hash<int>{}(static_cast<int>(type));
    for (const auto &pair : param) {
        key = hash_combine(key, pair.first);
        key = hash_combine(key, pair.second);
    }
    // 3. 查找缓存 - 现在使用更快的size_t哈希查找
    auto it = op_cache.find(key);
    if (it != op_cache.end()) {
        op_to_run = it->second;
        if (op_to_run->backend() != backend) {
            backend = op_to_run->backend();
        }
    } else {
        std::unique_ptr<Op> op_new(backend->opCreate(param, ""));
        if (!op_new) {
            backend = Backend::global_backends[MLLM_CPU].get();
            op_new.reset(backend->opCreate(param, ""));
        }
        op_to_run = std::move(op_new);
        op_cache[key] = op_to_run;
    }
    // ==================== [结束] Op 缓存 ====================
    // Module *module = Module::llm_model_ptr;
    // if (module && !module->doTrace) {
    //     auto end_time = mllm_time_us();
    //     string name_o = out_names.empty() ? "out-" + input_tensors[0].name() : out_names[0];
    //     std::cout << name_o << " dispatch Func: " << type << " in " << (end_time - start_time) / 1000.0F << " ms" << std::endl;
    // }
    // 4. 使用缓存的或新创建的 Op 执行计算
    if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) { // 针对QNN的特殊处理
        backend = Backend::global_backends[MLLM_QNN].get();                                                                  // 想不到吧
    }
    return backend->runOp(op_to_run.get(), input_tensors, out_names, in_place);
}

Tensor Tensor::operator+(float data) {
    OpParam param;
    param["data"] = data;
    return runFunc({name() + "-add"}, F_ADD, param,
                   {*this})[0];
}

Tensor Tensor::operator-(float data) {
    OpParam param;
    param["data"] = data;
    return runFunc({name() + "-sub"}, F_SUB, param,
                   {*this})[0];
}

Tensor Tensor::operator*(float data) {
    OpParam param;
    param["data"] = data;
    return runFunc({name() + "-mul"}, F_MUL, param,
                   {*this})[0];
}

Tensor Tensor::operator/(float data) {
    OpParam param;
    param["data"] = data;
    return runFunc({name() + "-div"}, F_DIV, param,
                   {*this})[0];
}

Tensor Tensor::operator/(double data) {
    OpParam param;
    param["data"] = static_cast<float>(data);
    return runFunc({name() + "-div"}, F_DIV, param,
                   {*this})[0];
}

Tensor Tensor::operator/(int data) {
    OpParam param;
    param["data"] = (float)data;
    return runFunc({name() + "-div"}, F_DIVINT, param,
                   {*this})[0];
}

Tensor Tensor::operator+(Tensor other) {
    return runFunc({name() + "-TTadd"}, F_TTADD, {},
                   {*this, other})[0];
}

Tensor Tensor::operator-(Tensor other) {
    return runFunc({name() + "-TTsub"}, F_TTSUB, {},
                   {*this, other})[0];
}

Tensor Tensor::operator*(Tensor other) {
    return runFunc({name() + "-TTmul"}, F_TTMUL, {},
                   {*this, other})[0];
}

Tensor Tensor::operator/(Tensor other) {
    return runFunc({name() + "-TTdiv"}, F_TTDIV, {},
                   {*this, other})[0];
}
Tensor Tensor::operator~() {
    return runFunc({name() + "~"}, TILDE, {}, {*this})[0];
}

Tensor Tensor::mean(Chl axis) {
    OpParam param;
    param["axis"] = (float)axis;
    return runFunc({name() + "-mean"}, F_MEAN, param,
                   {*this})[0];
}

Tensor Tensor::view(int b, int h, int s, int d, bool in_place) {
    OpParam param;
    param["b"] = (float)b;
    param["h"] = (float)h;
    param["s"] = (float)s;
    param["d"] = (float)d;
    return runFunc({name() + "-view"}, F_VIEW, param,
                   {*this}, in_place)[0];
}

Tensor Tensor::flatten(Chl axis_start, Chl axis_end) {
    OpParam param;
    param["axis_start"] = (float)axis_start;
    param["axis_end"] = (float)axis_end;
    return runFunc({name() + "-flatten"}, F_FLATTEN, param,
                   {*this}, true)[0];
}

Tensor Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
    OpParam param;
    param["num_pairs"] = (float)axiss.size();
    int idx = 0;
    for (auto &axis : axiss) {
        param["axis1_" + std::to_string(idx)] = (float)axis.first;
        param["axis2_" + std::to_string(idx)] = (float)axis.second;
        idx++;
    }
    bool in_place = (master_tensor_.expired() || (master_tensor_.lock()->name().find("Cache") == std::string::npos && master_tensor_.lock()->name().find("weight") != std::string::npos));
    // for BSHD attention start
    if (Module::llm_model_ptr == nullptr || backend()->type() != MLLM_CPU || (axiss.size() == 1 && axiss[0].first == HEAD && axiss[0].second == SEQUENCE)) {
        in_place = false; // in-place transpose
    }
    // for BSHD attention end
    return runFunc({name() + "-transpose"}, F_TRANPOSE, param,
                   {*this}, in_place)[0];
}

Tensor Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    OpParam param;
    param["b_size"] = (float)b.size();
    param["h_size"] = (float)h.size();
    param["s_size"] = (float)s.size();
    param["d_size"] = (float)d.size();
    for (int i = 0; i < b.size(); ++i) param["b_" + std::to_string(i)] = (float)b[i];
    for (int i = 0; i < h.size(); ++i) param["h_" + std::to_string(i)] = (float)h[i];
    for (int i = 0; i < s.size(); ++i) param["s_" + std::to_string(i)] = (float)s[i];
    for (int i = 0; i < d.size(); ++i) param["d_" + std::to_string(i)] = (float)d[i];
    string name_su = "-clip-";
    if (!(d.empty() && b.empty() && h.empty() && s.empty())) {
        for (auto as : param) {
            name_su += std::to_string(int(as.second)) + "_";
        }
    }
    return runFunc({name() + name_su}, F_CLIP, param,
                   {*this})[0];
}

Tensor Tensor::clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    OpParam param;
    param["axis"] = (float)keep_axis;
    param["b_size"] = (float)b.size();
    param["h_size"] = (float)h.size();
    param["s_size"] = (float)s.size();
    param["d_size"] = (float)d.size();
    for (int i = 0; i < b.size(); ++i) param["b_" + std::to_string(i)] = (float)b[i];
    for (int i = 0; i < h.size(); ++i) param["h_" + std::to_string(i)] = (float)h[i];
    for (int i = 0; i < s.size(); ++i) param["s_" + std::to_string(i)] = (float)s[i];
    for (int i = 0; i < d.size(); ++i) param["d_" + std::to_string(i)] = (float)d[i];
    return runFunc({name() + "-clipaxis"}, F_CLIPAXIS, param,
                   {*this})[0];
}

Tensor Tensor::clip(vector<int> index, Chl dim) {
    Tensor index_tensor(1, 1, 1, index.size(), impl_->backend_, false);
    index_tensor.alloc();
    for (size_t i = 0; i < index.size(); ++i) {
        index_tensor.setDataAt<float>(0, 0, 0, i, static_cast<float>(index[i]));
    }
    index_tensor.setName(name() + "-cliptensor-index");
    OpParam param;
    param["dim"] = (float)dim;
    return runFunc({name() + "-cliptensor"}, F_CLIPTENSOR, param,
                   {*this, index_tensor})[0];
}
Tensor Tensor::clip(Tensor index, Chl dim) {
    OpParam param;
    param["dim"] = (float)dim;
    return runFunc({name() + "-cliptensor"}, F_CLIPTENSOR, param,
                   {*this, index})[0];
}
Tensor Tensor::expand(int b, int h, int s, int d) {
    OpParam param;
    param["b"] = (float)b;
    param["h"] = (float)h;
    param["s"] = (float)s;
    param["d"] = (float)d;
    return runFunc({name() + "-expand"}, F_EXPPAND, param,
                   {*this})[0];
}

Tensor Tensor::norm(int L_n) {
    OpParam param;
    param["L_n"] = (float)L_n;
    return runFunc({name() + "-norm"}, F_NORM, param,
                   {*this})[0];
}

Tensor Tensor::where(float value, Chl axis) {
    OpParam param;
    param["value"] = value;
    param["axis"] = axis;
    return runFunc({name() + "-where"}, F_WHERE, param,
                   {*this})[0];
}

Tensor Tensor::index_put(Tensor value, Tensor indices, bool accumulate) {
    OpParam param;
    param["accumulate"] = (float)accumulate;
    return runFunc({name() + "-index_put"}, F_INDEX_PUT, param,
                   {*this, value, indices},
                   !accumulate)[0];
}
void Tensor::scatter_add(Tensor value, Tensor indices, Chl dim) {
    OpParam param;
    runFunc({}, F_SCATTERRADD, param,
            {*this, value, indices})[0];
}
void Tensor::scatter_(Chl dim, Tensor index, float src) {
    OpParam param;
    param["dim"] = (float)dim;
    param["value"] = src;
    runFunc({}, SCATTER, param,
            {*this, index})[0];
}
Tensor Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    OpParam param;
    param["axis"] = (float)axis;
    Module *module = input_tensors[0].module();
    vector<Tensor> inputs = {};
    for (auto &input_tensor : input_tensors) {
        inputs.push_back(input_tensor);
    }
    return runFunc({input_tensors[0].name() + "-cat"}, F_CAT, param, inputs)[0];
}

Tensor Tensor::mm(Tensor input0, Tensor input1) {
    Module *module = input0.module();
    string nname = input0.name() + "-mm-" + input1.name();
    return runFunc(
        {nname}, F_MM, {},
        {input0, input1})[0];
}

Tensor Tensor::range(int start, int end) {
    OpParam param;
    param["start"] = (float)start;
    param["end"] = (float)end;
    return runFunc({"range-" + std::to_string(start) + "-" + std::to_string(end)}, F_RANGE,
                   param, {})[0];
}

vector<Tensor> Tensor::split(Tensor input, std::vector<int> each_dims,
                             Chl split_dim, int same_dim_size) {
    OpParam param;
    vector<std::string> next_names;
    param["num_splits"] = (float)each_dims.size();
    for (int i = 0; i < each_dims.size(); ++i) {
        param["dim_" + std::to_string(i)] = (float)each_dims[i];
        next_names.push_back(input.name() + ".split-" + std::to_string(i));
    }
    param["split_dim"] = (float)split_dim;
    param["head_size"] = (float)same_dim_size;
    Module *module = input.module();
    return runFunc(next_names, F_SPLIT, param,
                   {input});
}

vector<Tensor> Tensor::topk(Tensor input, int k, Chl dim) {
    Module *module = input.module();
    OpParam param;
    param["k"] = (float)k;
    param["dim"] = (float)dim;
    return runFunc({input.name() + "-top" + std::to_string(k) + "-value",
                    input.name() + "-top" + std::to_string(k) + "-idx"},
                   F_TOPK,
                   param,
                   {input});
}
Tensor Tensor::sum(Chl dim) {
    OpParam param;
    param["dim"] = (float)dim;
    return runFunc({name() + "-sum"}, F_SUM, param,
                   {*this})[0];
}
Tensor Tensor::argsort() {
    return runFunc({name() + "-argsort"}, F_ARGSORT, {},
                   {*this})[0];
}
Tensor Tensor::bincount() {
    return runFunc({name() + "-bincount"}, F_BINCOUNT, {},
                   {*this})[0];
}
Tensor Tensor::repeat(Chl dim, int dim_size) {
    OpParam param;
    param["dim"] = (float)dim;
    param["dim_size"] = (float)dim_size;
    return runFunc({name() + "-repeat"}, F_REPEAT, param,
                   {*this})[0];
}
Tensor Tensor::masked_fill(Tensor mask_index, float value) {
    OpParam param;
    param["value"] = value;
    return runFunc({name() + "-masked_fill"}, MASKEDFILL, param, {*this, mask_index})[0];
}
Tensor Tensor::gather(Tensor input, Tensor index, Chl dim) {
    OpParam param;
    param["dim"] = dim;
    return runFunc({input.name() + "-gather"}, GATHER, param, {input, index})[0];
}
Tensor Tensor::zero_like(Tensor input) {
    Module *module = input.module();
    OpParam param;
    param["like_value"] = 0.0f;
    return runFunc({input.name() + "-zero_like"}, F_LIKE, param,
                   {input})[0];
}
Tensor Tensor::flash_attention2_forward(Tensor q, Tensor k, Tensor v, bool causal_mask) {
    Module *module = q.module();
    OpParam param;
    param["causal_mask"] = causal_mask ? 1.0f : 0.0f;
    return runFunc({q.name() + "-" + k.name() + "-fa2"}, F_FA2, param,
                   {q, k, v})[0];
};
Tensor Tensor::sage_attention_forward(Tensor q, Tensor k, Tensor v, bool causal_mask) {
    Module *module = q.module();
    OpParam param;
    param["causal_mask"] = causal_mask ? 1.0f : 0.0f;
    return runFunc({q.name() + "-" + k.name() + "-sage_attn"}, F_SAGEATTN, param,
                   {q, k, v})[0];
};
Tensor Tensor::apply_rotary_pos_emb_vision(Tensor input, Tensor rotary_pos_emb) {
    Module *module = input.module();
    return runFunc({input.name() + "-apply_rotary_pos_emb"}, F_APPLY_VISIOROPE,
                   {},
                   {input, rotary_pos_emb})[0];
}

Tensor Tensor::fuyu_gather_embd(Tensor word, Tensor image_patches, Tensor image_patches_indices) {
    Module *module = word.module();
    return runFunc({word.name() + ".fuyu_gather_embd"}, F_FUYU_GATHER_EMBD,
                   {},
                   {word, image_patches, image_patches_indices},
                   true)[0];
}

Tensor Tensor::phi3v_hd_merge(Tensor input, int h_crop, int w_crop) {
    Module *module = input.module();
    OpParam param;
    param["h_crop"] = (float)h_crop;
    param["w_crop"] = (float)w_crop;
    // The input tensor should be in BTHWC format
    return runFunc({input.name() + ".phi3v_hd_merge"}, F_PHI3V_HD_MERGE,
                   param,
                   {input})[0];
}

} // namespace mllm
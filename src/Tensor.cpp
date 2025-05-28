#include "Tensor.hpp"

#include <cassert>
#include <cstdlib>
#include <exception>
#include <express/ExpressBase.hpp>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include <Module.hpp>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace mllm {

/* Tensor类构造函数实现（对应头文件中的声明）*/
Tensor::Tensor(const int batch, const int head, const int sequence, const int dimension) :
    impl_(std::make_shared<TensorImpl>()) { // 初始化impl_
    reshape(batch, head, sequence, dimension);
}

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
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type];
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
    impl_->backend_ = bn;
    reshape(1, 1, 1, 1);
    alloc();
    impl_->should_in_graphs_ = false;
    setDataAt<float>(0, 0, 0, 0, static_cast<float>(value));
}

Tensor::Tensor(int value, BackendType bn_type) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type];
    reshape(1, 1, 1, 1);
    alloc();
    impl_->should_in_graphs_ = false;
    setDataAt<float>(0, 0, 0, 0, static_cast<float>(value));
}

Tensor::Tensor(std::vector<float> values, BackendType bn_type) :
    impl_(std::make_shared<TensorImpl>()) {
    impl_->dtype_ = MLLM_TYPE_F32;
    impl_->backend_ = Backend::global_backends[bn_type];
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
    if (master_tensor_ != nullptr) return;
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
    if (backend_type == MLLM_QNN && device() == MLLM_CPU) {
        this->free();
    }
    if (backend_type == MLLM_CPU && device() == MLLM_XNNPACK) {
        module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type]);
        this->setBackend(Backend::global_backends[backend_type]);
        return *this;
    }
    if (backend_type == MLLM_XNNPACK && device() == MLLM_CPU) {
        module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type]);
        this->setBackend(Backend::global_backends[backend_type]);
        return *this;
    }
    module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type]);
    this->alloc();
    return *this;
};

// TensorFuctions
std::vector<Tensor> Tensor::runFunc(std::vector<std::string> out_names,
                                    TensorFuncType type,
                                    std::vector<float> float_args,
                                    std::vector<std::shared_ptr<Tensor>> input_tensors,
                                    bool in_place) {
    auto backend = input_tensors.empty() ? Backend::global_backends[MLLM_CPU] : input_tensors[0]->backend();
    if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) {
        backend = Backend::global_backends[MLLM_QNN];
    }
    return backend->runFunc(out_names, type, float_args, input_tensors, in_place);
}

/*
Tensor &Tensor::getFunc(const std::string &suffix, const TensorFuncType type,
                        vector<float> float_args, vector<Tensor *> other_tensors) {
    assert(module() != nullptr);
    auto &module_tensors = module()->activation_tensors;
    auto &activation_tensors_num = module()->activation_tensors_num;
    const std::string next_name = impl_->name_ + "-" + suffix;
    // if (module_tensors.find(name_) == module_tensors.end()) {
    //     module_tensors[name_] = std::shared_ptr<Tensor>(this, [](Tensor *) {});
    // }
    if (module_tensors.find(next_name) == module_tensors.end()) {
        module_tensors[next_name] = std::make_shared<Tensor>(impl_->backend_);
        module_tensors[next_name]->setName(next_name);
        module_tensors[next_name]->setModule(module());
        activation_tensors_num[next_name] = 0;
    }
    if (module()->doLoad) { return *module_tensors[next_name]; }
    TensorFunction *func = impl_->backend_->funcCreate(type);
    std::vector<Tensor *> tensorPtrs = {module_tensors[impl_->name_].get()};
    for (auto &other_tensor : other_tensors) { tensorPtrs.push_back(other_tensor); }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup({module_tensors[next_name].get()}, tensorPtrs, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute({module_tensors[next_name].get()}, tensorPtrs, float_args);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (impl_->backend_->type() == BackendType::MLLM_CPU) {
            Tracer::addTensorFunction(func, tensorPtrs, {module_tensors[next_name].get()}, float_args);
        }
        break;
    }
    default: {
    }
    }
    if (Backend::global_backends.size() == 1) {
        for (auto input_tensor : tensorPtrs) {
            if (activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end()) {
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
                if (activation_tensors_num[input_tensor->name()] == 0 && module_tensors[input_tensor->name()]->sequence() > 1
                    && module_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
                    module_tensors[input_tensor->name()]->free();
                    // std::cout << input_tensor->name() << " |F" << std::endl;
                }
            }
        }
    }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << next_name << " | " << Tensor::tensor_status
                  << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
#ifdef DEBUGSAVETENSOR
    module_tensors[next_name]->saveNData<float>();
#endif
    return *module_tensors[next_name];
}

void Tensor::getFunc(const TensorFuncType type,
                     vector<float> float_args, vector<Tensor *> other_tensors) {
    assert(module() != nullptr);
    auto &module_tensors = module()->activation_tensors;
    auto &activation_tensors_num = module()->activation_tensors_num;
    if (module()->doLoad) { return; }
    TensorFunction *func = impl_->backend_->funcCreate(type);
    std::vector<Tensor *> tensorPtrs = {module_tensors[impl_->name_].get()};
    for (auto &other_tensor : other_tensors) { tensorPtrs.push_back(other_tensor); }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup({}, tensorPtrs, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute({}, tensorPtrs, float_args);
        break;
    }
    default: {
    }
    }
    if (Backend::global_backends.size() == 1) {
        for (auto input_tensor : tensorPtrs) {
            if (activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end()
                // && input_tensor->dimension() * input_tensor->sequence() > 0
            ) {
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
                if (activation_tensors_num[input_tensor->name()] == 0 && module_tensors[input_tensor->name()]->sequence() > 1
                    && module_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
                    module_tensors[input_tensor->name()]->free();
                    // std::cout << input_tensor->name() << " |F" << std::endl;
                }
            }
        }
    }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << " | " << Tensor::tensor_status
                  << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
}

std::vector<std::reference_wrapper<Tensor>> Tensor::getStaticFunc(vector<std::string> out_names,
                                                                  const TensorFuncType type,
                                                                  vector<float> float_args,
                                                                  vector<Tensor *> input_tensors) {
    Module *module;
    if (!input_tensors.empty()) {
        module = input_tensors[0]->module();
    } else {
        module = Module::llm_model_ptr;
    }
    assert(module != nullptr);
    auto &module_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;
    auto *backend_h = Backend::global_backends[MLLM_CPU];
    if (!input_tensors.empty() && input_tensors[0]->impl_->backend_ != nullptr) {
        backend_h = input_tensors[0]->backend();
    }
    for (auto out_name : out_names) {
        if (module_tensors.find(out_name) == module_tensors.end()) {
            module_tensors[out_name] = std::make_shared<Tensor>(backend_h);
            module_tensors[out_name]->setName(out_name);
            module_tensors[out_name]->setModule(module);
            activation_tensors_num[out_name] = 0;
        }
    }
    if (module->doLoad) {
        std::vector<std::reference_wrapper<Tensor>> results;
        for (auto out_name : out_names) { results.push_back(*module_tensors[out_name]); }
        return results;
    }
    TensorFunction *func = backend_h->funcCreate(type);
    // std::vector<Tensor *> tensorPtrs;
    // for (auto input_tensor : input_tensors){ tensorPtrs.push_back(module_tensors[input_tensor->name()].get()); }
    std::vector<Tensor *> outPtrs;
    for (auto out_name : out_names) { outPtrs.push_back(module_tensors[out_name].get()); }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup(outPtrs, input_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute(outPtrs, input_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (backend_h->type() == BackendType::MLLM_CPU) {
            Tracer::addTensorFunction(func, input_tensors, outPtrs, float_args);
        }
        break;
    }
    default: {
    }
    }
    if (Backend::global_backends.size() == 1) {
        for (auto input_tensor : input_tensors) {
            if (activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end()) {
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
                if (activation_tensors_num[input_tensor->name()] == 0 && module_tensors[input_tensor->name()]->sequence() > 1
                    && module_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
                    module_tensors[input_tensor->name()]->free();
                    // std::cout << input_tensor->name() << " |S "<< std::endl;// << out_names[0] << std::endl;
                }
            }
        }
    }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << out_names[0] << " | " << Tensor::tensor_status
                  << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
#ifdef DEBUGSAVETENSOR
    for (auto out_name : out_names) { module_tensors[out_name]->saveNData<float>(); }
#endif
    std::vector<std::reference_wrapper<Tensor>> results;
    for (auto out_name : out_names) { results.push_back(*module_tensors[out_name]); }
    return results;
}
*/

Tensor Tensor::operator+(float data) {
    return runFunc({name() + "-add"}, FUNC_ADD, {data},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator-(float data) {
    return runFunc({name() + "-sub"}, FUNC_SUB, {data},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator*(float data) {
    return runFunc({name() + "-mul"}, FUNC_MUL, {data},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator/(float data) {
    return runFunc({name() + "-div"}, FUNC_DIV, {data},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator/(double data) {
    return runFunc({name() + "-div"}, FUNC_DIV, {static_cast<float>(data)},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator/(int data) {
    return runFunc({name() + "-div"}, FUNC_DIVINT, {static_cast<float>(data)},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::operator+(Tensor other) {
    return runFunc({name() + "-TTadd"}, FUNC_TTADD, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&other, [](Tensor *) {})})[0];
}

Tensor Tensor::operator-(Tensor other) {
    return runFunc({name() + "-TTsub"}, FUNC_TTSUB, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&other, [](Tensor *) {})})[0];
}

Tensor Tensor::operator*(Tensor other) {
    return runFunc({name() + "-TTmul"}, FUNC_TTMUL, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&other, [](Tensor *) {})})[0];
}

Tensor Tensor::operator/(Tensor other) {
    return runFunc({name() + "-TTdiv"}, FUNC_TTDIV, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&other, [](Tensor *) {})})[0];
}

Tensor Tensor::mean(Chl axis) {
    return runFunc({name() + "-mean"}, FUNC_MEAN, {(float)axis},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::view(int b, int h, int s, int d) {
    return runFunc({name() + "-view"}, FUNC_VIEW, {(float)b, (float)h, (float)s, (float)d},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})}, true)[0];
}

Tensor Tensor::flatten(Chl axis_start, Chl axis_end) {
    return runFunc({name() + "-flatten"}, FUNC_FLATTEN, {(float)axis_start, (float)axis_end},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})}, true)[0];
}

Tensor Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
    vector<float> axis_s;
    for (auto &axis : axiss) {
        axis_s.push_back((float)axis.first);
        axis_s.push_back((float)axis.second);
    }
    return runFunc({name() + "-transpose"}, FUNC_TRANPOSE, axis_s,
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})}, master_tensor_ == nullptr)[0];
}

Tensor Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    vector<float> axis_s;
    axis_s.push_back(b.size());
    axis_s.push_back(h.size());
    axis_s.push_back(s.size());
    axis_s.push_back(d.size());
    for (auto &axis : b) { axis_s.push_back((float)axis); }
    for (auto &axis : h) { axis_s.push_back((float)axis); }
    for (auto &axis : s) { axis_s.push_back((float)axis); }
    for (auto &axis : d) { axis_s.push_back((float)axis); }
    string name_su = "clip-";
    if (!(d.size() == 2 && b.empty() && h.empty() && s.empty())) {
        for (auto as : axis_s) {
            name_su += std::to_string(int(as)) + "_";
        }
    }
    return runFunc({name() + name_su}, FUNC_CLIP, axis_s,
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    vector<float> axis_s = {(float)keep_axis};
    axis_s.push_back(b.size());
    axis_s.push_back(h.size());
    axis_s.push_back(s.size());
    axis_s.push_back(d.size());
    for (auto &axis : b) { axis_s.push_back((float)axis); }
    for (auto &axis : h) { axis_s.push_back((float)axis); }
    for (auto &axis : s) { axis_s.push_back((float)axis); }
    for (auto &axis : d) { axis_s.push_back((float)axis); }
    return runFunc({name() + "-clipaxis"}, FUNC_CLIPAXIS, axis_s,
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::clip(Tensor index, Chl dim) {
    return runFunc({name() + "-cliptensor"}, FUNC_CLIPTENSOR, {(float)dim},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&index, [](Tensor *) {})})[0];
}
Tensor Tensor::expand(int b, int h, int s, int d) {
    return runFunc({name() + "-expand"}, FUNC_EXPPAND, {(float)b, (float)h, (float)s, (float)d},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::norm(int L_n) {
    return runFunc({name() + "-norm"}, FUNC_NORM, {(float)L_n},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::where(float value, Chl axis) {
    return runFunc({name() + "-where"}, FUNC_WHERE, {(float)value, (float)axis},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}

Tensor Tensor::index_put(Tensor value, Tensor indices, bool accumulate) {
    return runFunc({name() + "-index_put"}, FUNC_INDEX_PUT, {(float)accumulate},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&value, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&indices, [](Tensor *) {})},
                   !accumulate)[0];
}
void Tensor::scatter_reduce(Tensor value, Tensor indices) {
    runFunc({name()}, FUNC_SCATTERREDUCE, {},
            {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
             std::shared_ptr<Tensor>(&value, [](Tensor *) {}),
             std::shared_ptr<Tensor>(&indices, [](Tensor *) {})})[0];
}

Tensor Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    Module *module = input_tensors[0].module();
    vector<shared_ptr<Tensor>> inputs = {};
    for (auto &input_tensor : input_tensors) {
        inputs.push_back(std::shared_ptr<Tensor>(&input_tensor, [](Tensor *) {}));
    }
    return runFunc({input_tensors[0].name() + "-cat"}, FUNC_CAT, {(float)axis}, inputs)[0];
}

Tensor Tensor::mm(Tensor input0, Tensor input1) {
    Module *module = input0.module();
    string nname = input0.name() + "-mm-" + input1.name();
    return runFunc(
        {nname}, FUNC_MM, {},
        {std::shared_ptr<Tensor>(&input0, [](Tensor *) {}),
         std::shared_ptr<Tensor>(&input1, [](Tensor *) {})})[0];
}

Tensor Tensor::range(int start, int end) {
    return runFunc({"range-" + std::to_string(start) + "-" + std::to_string(end)}, FUNC_RANGE,
                   {(float)start, (float)end}, {})[0];
}

vector<Tensor> Tensor::split(Tensor input, std::vector<int> each_dims,
                             Chl split_dim, int same_dim_size) {
    vector<std::string> next_names;
    std::vector<float> args;
    for (int i = 0; i < each_dims.size(); ++i) {
        args.push_back(each_dims[i]);
        next_names.push_back(input.name() + ".split-" + std::to_string(i));
    }
    args.push_back(split_dim);
    args.push_back(same_dim_size);
    Module *module = input.module();
    return runFunc(next_names, FUNC_SPLIT, args,
                   {std::shared_ptr<Tensor>(&input, [](Tensor *) {})});
}

vector<Tensor> Tensor::topk(Tensor input, int k, Chl dim) {
    Module *module = input.module();
    return runFunc({input.name() + "-top" + std::to_string(k) + "-value",
                    input.name() + "-top" + std::to_string(k) + "-idx"},
                   FUNC_TOPK,
                   {(float)k, (float)dim},
                   {std::shared_ptr<Tensor>(&input, [](Tensor *) {})});
}
Tensor Tensor::sum(Chl dim) {
    return runFunc({name() + "sum"}, FUNC_SUM, {(float)dim},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}
Tensor Tensor::argsort() {
    return runFunc({name() + "argsort"}, FUNC_ARGSORT, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}
Tensor Tensor::bincount() {
    return runFunc({name() + "bincount"}, FUNC_BINCOUNT, {},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}
Tensor Tensor::repeat(Chl dim, int dim_size) {
    return runFunc({name() + "repeat"}, FUNC_REPEAT, {(float)dim, (float)dim_size},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})})[0];
}
Tensor Tensor::zero_like(Tensor input) {
    Module *module = input.module();
    return runFunc({input.name() + "-zero_like"}, FUNC_LIKE, {0.0},
                   {std::shared_ptr<Tensor>(&input, [](Tensor *) {})})[0];
}
Tensor Tensor::apply_rotary_pos_emb_vision(Tensor input, Tensor rotary_pos_emb) {
    Module *module = input.module();
    return runFunc({input.name() + "-apply_rotary_pos_emb"}, FUNC_APPLY_VISIOROPE,
                   {},
                   {std::shared_ptr<Tensor>(&input, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&rotary_pos_emb, [](Tensor *) {})})[0];
}

Tensor Tensor::fuyu_gather_embd(Tensor word, Tensor image_patches, Tensor image_patches_indices) {
    Module *module = word.module();
    return runFunc({word.name() + ".fuyu_gather_embd"}, FUNC_FUYU_GATHER_EMBD,
                   {},
                   {std::shared_ptr<Tensor>(&word, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&image_patches, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&image_patches_indices, [](Tensor *) {})},
                   true)[0];
}

Tensor Tensor::phi3v_hd_merge(Tensor input, int h_crop, int w_crop) {
    Module *module = input.module();
    return runFunc({input.name() + ".phi3v_hd_merge"}, FUNC_PHI3V_HD_MERGE,
                   {(float)h_crop, (float)w_crop},
                   {std::shared_ptr<Tensor>(&input, [](Tensor *) {})})[0];
}

} // namespace mllm
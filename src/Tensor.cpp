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
        module()->activation_tensors[name()]->setBackend(Backend::global_backends[backend_type]);
        this->setBackend(Backend::global_backends[backend_type]);
        return *this;
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
    bool in_place = (master_tensor_ == nullptr);
    // for BSHD attention start
    if(axiss.size() == 1 && axiss[0].first == HEAD && axiss[0].second == SEQUENCE) {
        in_place = false; // in-place transpose
    }
    // for BSHD attention end
    return runFunc({name() + "-transpose"}, FUNC_TRANPOSE, axis_s,
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {})}, in_place)[0];
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

Tensor Tensor::clip(vector<int> index, Chl dim) {
    Tensor index_tensor(1, 1, 1, index.size(), impl_->backend_, false);
    index_tensor.alloc();
    for (size_t i = 0; i < index.size(); ++i) {
        index_tensor.setDataAt<float>(0, 0, 0, i, static_cast<float>(index[i]));
    }
    index_tensor.setName(name() + "-cliptensor-index");
    return runFunc({name() + "-cliptensor"}, FUNC_CLIPTENSOR, {(float)dim},
                   {std::shared_ptr<Tensor>(this, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&index_tensor, [](Tensor *) {})})[0];
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
Tensor Tensor::flash_attention2_forward(Tensor q, Tensor k, Tensor v, bool causal_mask) {
    Module *module = q.module();
    return runFunc({q.name() + "-" + k.name() + "-fa2"}, FUNC_FA2, {causal_mask ? 1.0f : 0.0f},
                   {std::shared_ptr<Tensor>(&q, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&k, [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&v, [](Tensor *) {})})[0];
};
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
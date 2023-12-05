
#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H
#include <climits>
#include "Backend.hpp"
#include "Check.hpp"
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include "Timing.hpp"

#include <assert.h>

const auto KMaxAxes = 32;

namespace mllm {
class Backend;

class Tensor {
public:
    // Tensor():data_(), diff_(), capacity_(0){}
    Tensor() :
        host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    Tensor(Backend *bn) :
        backend_(bn), host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    ~Tensor() {
        if (host_ptr_ != nullptr && masterTensor() == nullptr && !aggregated_) {
//            std::cout << "free " << name() << std::endl;
            backend_->free(host_ptr_);
            //allocated_ = false;
//            ::free(host_ptr_);
            host_ptr_ = nullptr;
        }
    }
    explicit Tensor(const int num, const int channels, const int height, const int width); // N C H W like Caffe //TODO add param: HostMemory; NCHW_Type?
    explicit Tensor(const vector<int> &shape);

    Backend *backend() const {
        return backend_;
    }
    void setBackend(Backend *bn) {
        backend_ = bn;
    };
    void setDtype(DataType dtype) {
        dtype_ = dtype;
    }

    inline const vector<int> &shape() const {
        return shape_;
    }

    //    bool reshape(const int num, const int channels, const int height, const int width);
    bool reshape(const int batch, const int head, const int sequence, const int dimension);

//    bool reshape_unsafe(const vector<int> &shape){
//        shape_ = shape;
//    }

    void alloc();
    void alloc(DataType dtype) {
        dtype_ = dtype;
        alloc();
    }

    void free(){
        if(aggregated_){return;}
        if (host_ptr_ != nullptr && masterTensor() == nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
            allocated_ = 0;
        }
    }

    void update();
    
    size_t size() const {
        return capacity_ * dtypeSize();
    }
    /*
    // Deprecated legacy shape accessor num: use batch() instead.
    inline int num() const {
        return legacyShape(0);
    }
    // Deprecated legacy shape accessor channels: use head() instead.
    inline int channels() const {
        return legacyShape(1);
    }
    // Deprecated legacy shape accessor height: use sequence() instead.
    inline int height() const {
        return legacyShape(2);
    }
    // Deprecated legacy shape accessor width: use dimension() instead.
    inline int width() const {
        return legacyShape(3);
    }
    */

    inline int batch() const {
        return legacyShape(0);
    }
    inline int head() const {
        switch (ctype_) {
        // case BHSD:
        //     return legacyShape(1);
        case BSHD:
            return legacyShape(2);
        case BHDS:
            return legacyShape(1);
        default:
            return -1;
        }
    }
    inline int sequence() const {
        switch (ctype_) {
        // case BHSD:
        //     return legacyShape(2);
        case BSHD:
            return legacyShape(1);
        case BHDS:
            return legacyShape(3);
        default:
            return -1;
        }
    }
    inline int dimension() const {
        switch (ctype_) {
        // case BHSD:
        //     return legacyShape(3);
        case BSHD:
            return legacyShape(3);
        case BHDS:
            return legacyShape(2);
        default:
            return -1;
        }
    }

    inline int count() const {
        return count_;
    }
    inline int numAxes() const {
        return shape_.size();
    }
    inline string ShapeString() const {
        std::ostringstream stream;
        for (int i : shape_) {
            stream << i << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
    }
    inline int canonicalAxisIndex(int axis_index) const {
        CHECK_GE(axis_index, -numAxes())
            << "axis " << axis_index << " out of range for " << numAxes()
            << "-D Tensor with shape " << ShapeString();
        CHECK_LT(axis_index, numAxes())
            << "axis " << axis_index << " out of range for " << numAxes()
            << "-D Tensor with shape " << ShapeString();
        if (axis_index < 0) {
            return axis_index + numAxes();
        }
        return axis_index;
    }
    inline int legacyShape(int index) const {
        CHECK_LE(numAxes(), 4)
            << "Cannot use legacy accessors on Tensors with > 4 axes.";
        CHECK_LT(index, 4);
        CHECK_GE(index, -4);
        if (index >= numAxes() || index < -numAxes()) {
            // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
            // indexing) -- this special case simulates the one-padding used to fill
            // extraneous axes of legacy Tensors.
            return 1;
        }
        return shape(index);
    }

    //    inline int offset(const int n, const int c = 0, const int h = 0,
    //                      const int w = 0) const {
    //        CHECK_GE(n, 0);
    //        CHECK_LE(n, batch());
    //        CHECK_GE(head(), 0);
    //        CHECK_LE(c, head());
    //        CHECK_GE(sequence(), 0);
    //        CHECK_LE(h, sequence());
    //        CHECK_GE(dimension(), 0);
    //        CHECK_LE(w, dimension());
    //        return ((n * head() + c) * sequence() + h) * dimension() + w;
    //    }
    inline int offset(const int b, const int h = 0, const int s = 0,
                      const int d = 0) const {
        // batch, head, sequence, dimension
        // CHECK_GE(b, 0);
        // CHECK_LE(b, batch());
        // CHECK_GE(head(), 0);
        // CHECK_LE(h, head());
        // CHECK_GE(sequence(), 0);
        // CHECK_LE(s, sequence());
        // CHECK_GE(dimension(), 0);
        // CHECK_LE(d, dimension());
        if (shape_offset_.size() == 4 & shape_master_.size() == 4) {
            const int base_batch_ = shape_master_[0];
            const int base_head_ = shape_master_[1];
            const int base_sequence_ = shape_master_[2];
            const int base_dimension_ = shape_master_[3];
            const int b_ = (b + shape_offset_[0])%base_batch_;
            const int h_ = (h + shape_offset_[1])%base_head_;
            const int s_ = (s + shape_offset_[2])%base_sequence_;
            const int d_ = (d + shape_offset_[3])%base_dimension_;
            switch (ctype_) {
            // case BHSD:
            //     return ((b_ * base_head_ + h_) * base_sequence_ + s_) * base_dimension_ + d_;
            case BSHD:
                return ((b_ * base_sequence_ + s_) * base_head_ + h_) * base_dimension_ + d_;
            case BHDS:
                return ((b_ * base_head_ + h_) * base_dimension_ + d_) * base_sequence_ + s_;
            default:
                break;
            }
        } else {
            switch (ctype_) {
            // case BHSD:
            //     return ((b * head() + h) * sequence() + s) * dimension() + d;
            case BSHD:
                return ((b * shape_[1] + s) * shape_[2] + h) * shape_[3] + d;
            case BHDS:
                return ((b * shape_[1] + h) * shape_[2] + d) * shape_[3] + s;
            default:
                break;
            }
        }
    }

    inline int offset(const vector<int> &indices) const {
        if (shape_offset_.size() == 4 & shape_master_.size() == 4) {
            return offset(indices[0], indices[1], indices[2], indices[3]);
        } else {
            CHECK_LE(indices.size(), numAxes());
            int offset = 0;
            for (int i = 0; i < numAxes(); ++i) {
                offset *= shape(i);
                if (indices.size() > i) {
                    CHECK_GE(indices[i], 0);
                    CHECK_LT(indices[i], shape(i));
                    offset += indices[i];
                }
            }
            return offset;
        }
    }
    /**
     * @brief Copy from a source Tensor.
     * @param source the Tensor to copy from
     * @param copy_diff if false, copy the data; if true, copy the diff
     * @param reshape if false, require this Tensor to be pre-shaped to the shape
     *        of other (and die otherwise); if true, reshape this Tensor to other's
     *        shape if necessary
     */
    void copyFrom(const Tensor &source, bool copy_diff = false,
                  bool reshape = false);
    void copyFrom(const shared_ptr<Tensor> &source, bool reshape = false);

    void deepCopyFrom(const shared_ptr<Tensor> &source, bool copyshape = true) {
        setMasterTensor(source.get());
        if(ctype_ != master_tensor_->ctype()) {
            if (transed_) {
                auto b = master_tensor_->batch();
                auto h = master_tensor_->head();
                auto d = master_tensor_->dimension();
                auto s = master_tensor_->sequence();
                master_tensor_->ctype_ = ctype_;
                master_tensor_->reshape(b, h, s,d);
            }else {
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                ctype_ = master_tensor_->ctype_;
                reshape(b, h, s,d);
                // ctype_ = source.ctype_;
            }
        }
        // deep Copy
        host_ptr_ = source->hostPtr<void>();
        capacity_ = source->capacity_;
        count_ = source->count_;
        if (copyshape) {
            shape_ = source->shape_;
        }
        allocated_ = source->allocated_;
        dtype_ = source->dtype_;
        // ctype_ = source->ctype_;
        //
        for (auto &child_tensor: child_tensors_) {
            child_tensor->deepCopyFrom(source, false);
            //remove child_temsor from child_tensors_:
            child_tensors_.erase(std::remove(child_tensors_.begin(), child_tensors_.end(), child_tensor), child_tensors_.end());
        }
        //
        source->addChildTensor(this);
    }
    /**
     * \brief this Tensor is a DEEP COPY of source
     * \param source
     * \param shape_offset
     */
    void deepCopyFrom(Tensor &source, const vector<int> &shape_offset) {
        //
        setMasterTensor(&source);
        if(ctype_ != master_tensor_->ctype()) {
            if (transed_) {
                auto b = master_tensor_->batch();
                auto h = master_tensor_->head();
                auto d = master_tensor_->dimension();
                auto s = master_tensor_->sequence();
                master_tensor_->ctype_ = ctype_;
                master_tensor_->reshape(b, h, s,d);
            }else {
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                ctype_ = master_tensor_->ctype_;
                reshape(b, h, s,d);
                // ctype_ = source.ctype_;
            }
        }
        assert(source.allocted());
        // don't need alloc()
        shape_offset_ = shape_offset;
        shape_master_ = {source.batch(), source.head(), source.sequence(), source.dimension()};
        if(source.head() != head()) {
            shape_master_ = {source.batch(), head(), source.sequence(), source.dimension() * source.head() / head()};
        }
        // deep Copy
        host_ptr_ = source.hostPtr<void>();
        allocated_ = source.allocated_;
        dtype_ = source.dtype_;
        //
        for (auto &child_tensor: child_tensors_) {
            child_tensor->deepCopyFrom(source, shape_offset);
            //remove child_temsor from child_tensors_:
            child_tensors_.erase(std::remove(child_tensors_.begin(), child_tensors_.end(), child_tensor), child_tensors_.end());
        }
        source.addChildTensor(this);
    }

    template <typename Dtype>
    Dtype *hostPtr() const {
        return (Dtype *)host_ptr_;
    }

    //    template <typename Dtype>
    //    Dtype dataAt(const int n, const int c, const int h, const int w) const {
    //        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
    //        return ((Dtype *)host_ptr_)[offset(n, c, h, w)];
    //    }
    template <typename Dtype>
    Dtype dataAt(const int batch, const int head, const int sequence, const int dimension) const {
        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
        if(!aggregated_) {
            return ((Dtype *)host_ptr_)[offset(batch, head, sequence, dimension)];
        }else{
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->dataAt<Dtype>(b, h, s, d);
        }
    }
    template <typename Dtype>
    Dtype dataAtDangerously(const int offset) const {
        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
        return ((Dtype *)host_ptr_)[offset];
    }

    template <typename Dtype>
    Dtype dataAt(const vector<int> &index) const {
        return dataAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    template <typename Dtype>
    Dtype* ptrAt(const int batch, const int head, const int sequence, const int dimension) {
        if(!aggregated_){
            return ((Dtype *)host_ptr_ + offset(batch, head, sequence, dimension));
        }else{
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->ptrAt<Dtype>(b, h, s, d);
        }
    }

    template <typename Dtype>
    Dtype* ptrAt(const vector<int> &index) const {
        return ptrAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    template <typename Dtype>
    void setDataAt(const int batch, const int head, const int sequence, const int dimension, Dtype value) {
        if(!aggregated_) {
            Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
            typed_ptr[offset(batch, head, sequence, dimension)] = value;
        }else{
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            aggregated_tensors_[tensor_id]->setDataAt<Dtype>(b, h, s, d, value);
        }
    }
    template <typename Dtype>
    void setDataAtDangerously(const int offset, Dtype value) const {
        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
        ((Dtype *)host_ptr_)[offset] = value;
    }

    template <typename Dtype>
    void setDataAt(const vector<int> &index, Dtype value) {
        setDataAt(index[0], index[1], index[2], index[3], value);
    }

    DataType dtype() const {
        return dtype_;
    }
    ChlType ctype() const {
        return ctype_;
    }
    void transShape(Chl dim_a, Chl dim_b) {
        if(dim_a == SEQUENCE && dim_b == DIMENSION) {
            if(ctype() == BSHD) {
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                ctype_ = BHDS;
                reshape(b, h, s,d);
                transed_ = true;
            }else if (transed_) {

            }
            else {
                std::cout<<"TODO, need support!"<<std::endl;
            }
        } else {
            std::cout<<"TODO, need support!"<<std::endl;
        }
    }

    size_t cntSize() {
        return DataTypeSize(dtype_, count_);
    }

    int dtypeSize() const {
        return DataTypeSize(dtype_, 1);
    }
    int dtypeSize(int size) {
        return DataTypeSize(dtype_, size);
    }

    void setName(string name) {
        name_ = name;
    }

    string name() const {
        return name_;
    }

    int allocted() const {
        return allocated_;
    }

    vector<int> shape_offset() const {
        return shape_offset_;
    }
    vector<int> shape_master() const {
        return shape_master_;
    }

    Tensor *masterTensor() const {
        return master_tensor_;
    }
    void setMasterTensor(Tensor* master_tensor) {
        master_tensor_ = master_tensor;
    }

    vector<Tensor *> childTensors() {
        return child_tensors_;
    }
    void addChildTensor(Tensor* child) {
        child_tensors_.push_back(child);
    }


    void addTensors(vector<shared_ptr<Tensor>> ts, Chl dim) {
        aggregated_ = true;
        aggregated_dim_ = dim;
        aggregated_dims_ = {};
        switch (dim) {
        case HEAD:{
            auto sum = 0;
            for (auto &t : ts) {
                CHECK_EQ(t->batch(), batch());
                CHECK_EQ(t->sequence(), sequence());
                CHECK_EQ(t->dimension(), dimension());
                sum += t->head();
                aggregated_dims_.push_back(sum);
            }
            CHECK_EQ(sum, head());
            break;
        }
        case SEQUENCE: {
            auto sum = 0;
            for (auto &t : ts) {
                CHECK_EQ(t->batch(), batch());
                CHECK_EQ(t->head(), head());
                CHECK_EQ(t->dimension(), dimension());
                sum += t->sequence();
                aggregated_dims_.push_back(sum);
            }
            CHECK_EQ(sum, sequence());
            break;
        }
        case DIMENSION: {
            auto sum = 0;
            for (auto &t : ts) {
                CHECK_EQ(t->batch(), batch());
                CHECK_EQ(t->head(), head());
                CHECK_EQ(t->sequence(), sequence());
                sum += t->dimension();
                aggregated_dims_.push_back(sum);
            }
            CHECK_EQ(sum, dimension());
            break;
        }
        default:
            break;
        }
        aggregated_tensors_ = ts;
    }

public:
    /*TEST*/

    template <typename Dtype>
    void checkData() {
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        bool ck = false;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float value = dataAt<Dtype>(n, c, h, w);
                        if (std::isnan(value) || std::isnan(-value)) {
                            // std::cout<<"["<<n<<","<<c<<","<<h<<","<<w<<"] ";//<<std::flush;
                            ck = true;
                        }
                    }
                }
            }
        }
        if(ck) {
            std::cout<<"\n[ERROR]:" << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] has Nan" << std::endl;
            //printData<Dtype>();
            assert(ck == false);
        }
    }

    void printShape(){
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
    }

    template <typename Dtype>
    void printData() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(0, 0, h, c) << " ";
                }
                std::cout << std::endl;
                std::cout << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(0, c, h, 0) << " ";
                }
                std::cout << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }

    template <typename Dtype>
    void printMem() {
        for (int i = 0; i < count_; ++i) {
            auto *typed_ptr = static_cast<Dtype *>(host_ptr_);
            std::cout << std::fixed << std::setprecision(7) << typed_ptr[i] << " ";
        }
    }

    template <typename Dtype>
    void printAVG() {
        float sum = 0;
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        bool ck = false;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float value = dataAt<Dtype>(n, c, h, w);
                        sum += value;
                    }
                }
            }
        }
        std::cout << name() << " " << sum / count() << std::endl;
//        std::cout << name() << ": shape:[" << num() << " " << channels() << " " << height() << " " << width() << "] AVG:" << sum / count() << std::endl;
    }

    shared_ptr<Tensor> view(int batch, int head, int sequence, int dimension) {
        auto t = std::make_shared<Tensor>();
        t->setBackend(backend_);
        t->setDtype(dtype_);
        t->reshape(batch, head, sequence, dimension);
        t->host_ptr_ = host_ptr_;
        return t;
    }

    shared_ptr<Tensor> unfold(int axis, int size, int step) {
        CHECK_GE(axis, 0);
        CHECK_LT(axis, numAxes());
        CHECK_GE(size, 0);
        CHECK_GE(step, 0);
        CHECK_LE(size, shape(axis));
        CHECK_LE(step, shape(axis));
        CHECK_EQ(shape(axis) % step, 0);
        auto t = std::make_shared<Tensor>();
        t->setBackend(backend_);
        t->setDtype(dtype_);
        vector<int> shape = shape_;
        shape[axis] = size;
        shape.insert(shape.begin() + axis + 1, shape[axis] / step);
        shape[axis + 1] = step;
        t->reshape(shape);
        t->host_ptr_ = host_ptr_;
        return t;
    }

    template <class Dtype>
    void fullData(Dtype value) {
        for (int n = 0; n < batch(); ++n) {
            for (int c = 0; c < head(); ++c) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int w = 0; w < dimension(); ++w) {
                        setDataAt<Dtype>(n, c, h, w, value);
                    }
                }
            }
        }
    }

    void fullDataTest() {
        for (int n = 0; n < batch(); ++n) {
            for (int c = 0; c < head(); ++c) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int w = 0; w < dimension(); ++w) {
                        setDataAt<float>(n, c, h, w, offset(n, c, h, w));
                    }
                }
            }
        }
    }
    void fullDataTest2() {
        for (int i = 0; i < count_; ++i) {
            float *typed_ptr = static_cast<float *>(host_ptr_);
            typed_ptr[i] = i;
        }
    }

    void permute(int axis0, int axis1, int axis2, int axis3, bool copy = true);

private:
    bool reshape(const vector<int> &shape);
    inline int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
    }

    int checkDim(int &b, int &h, int &s, int &d) const {
        if (!aggregated_){
            return -1;
        }
        int tensor_id = -1;
        switch (aggregated_dim_) {
        case HEAD: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (h < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            h = h - aggregated_dims_[tensor_id-1];
            break;
        }
        case SEQUENCE: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (s < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            s = s - aggregated_dims_[tensor_id-1];
            break;
        }
        case DIMENSION: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (d < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            d = d - aggregated_dims_[tensor_id-1];
            break;
        }
        default:
            break;
        }
        return tensor_id;
    }

private:
    string name_;
    DataType dtype_;
    ChlType ctype_ = BSHD;
    Backend *backend_;
    void *host_ptr_;
    void *device_ptr_;
    vector<int> shape_; // 保存 N K H W
    int capacity_;      // 元素个数 申请内存的总长度相关
    int count_;         // 当前元素数
    int allocated_ = 0;


    // shadow tensor if;
    vector<int> shape_offset_;
    vector<int> shape_master_;
    Tensor* master_tensor_ = nullptr;
    vector<Tensor *> child_tensors_;
    bool transed_ = false;

    //aggregated
    bool aggregated_ = false;
    vector<shared_ptr<Tensor>> aggregated_tensors_;
    Chl aggregated_dim_;
    vector<int> aggregated_dims_ ;

};
} // namespace mllm
#endif // MLLM_TENSOR_H
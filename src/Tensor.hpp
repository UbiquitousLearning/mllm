
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
        if (host_ptr_ != nullptr) {
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

    //    bool reshape(const int num, const int channels, const int height, const int width);
    bool reshape(const int batch, const int head, const int sequence, const int dimension);
    bool reshape(const vector<int> &shape);

    void alloc();
    void alloc(DataType dtype) {
        dtype_ = dtype;
        alloc();
    }

    void free(){
        if (host_ptr_ != nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
            allocated_ = 0;
        }
    }

    void update();
    
    size_t size() const {
        return capacity_ * dtypeSize();
    }

    // Deprecated legacy shape accessor num: use shape(0) instead.
    inline int num() const {
        return legacyShape(0);
    }
    // Deprecated legacy shape accessor channels: use shape(1) instead.
    inline int channels() const {
        return legacyShape(1);
    }
    // Deprecated legacy shape accessor height: use shape(2) instead.
    inline int height() const {
        return legacyShape(2);
    }
    // Deprecated legacy shape accessor width: use shape(3) instead.
    inline int width() const {
        return legacyShape(3);
    }

    inline int batch() const {
        return legacyShape(0);
    }
    inline int head() const {
        return legacyShape(1);
    }
    inline int sequence() const {
        return legacyShape(2);
    }
    inline int dimension() const {
        return legacyShape(3);
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
    inline const vector<int> &shape() const {
        return shape_;
    }
    inline int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
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
    //        CHECK_LE(n, num());
    //        CHECK_GE(channels(), 0);
    //        CHECK_LE(c, channels());
    //        CHECK_GE(height(), 0);
    //        CHECK_LE(h, height());
    //        CHECK_GE(width(), 0);
    //        CHECK_LE(w, width());
    //        return ((n * channels() + c) * height() + h) * width() + w;
    //    }
    inline int offset(const int b, const int h = 0, const int s = 0,
                      const int d = 0) const {
        // batch, head, sequence, dimension
        CHECK_GE(b, 0);
        CHECK_LE(b, batch());
        CHECK_GE(head(), 0);
        CHECK_LE(h, head());
        CHECK_GE(sequence(), 0);
        CHECK_LE(s, sequence());
        CHECK_GE(dimension(), 0);
        CHECK_LE(d, dimension());
        if (shape_offset_.size() == 4 & shape_base_.size() == 4) {
            const int base_head_ = shape_base_[1];
            const int base_sequence_ = shape_base_[2];
            const int base_dimension_ = shape_base_[3];
            const int b_ = b + shape_offset_[0];
            const int h_ = h + shape_offset_[1];
            const int s_ = s + shape_offset_[2];
            const int d_ = d + shape_offset_[3];
            return ((b_ * base_head_ + h_) * base_sequence_ + s_) * base_dimension_ + d_;
        } else {
            return ((b * head() + h) * sequence() + s) * dimension() + d;
        }
    }

    inline int offset(const vector<int> &indices) const {
        if (shape_offset_.size() == 4 & shape_base_.size() == 4) {
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

    void deepCopyFrom(const shared_ptr<Tensor> &source, bool reshape = false) {
        // deep Copy
        host_ptr_ = source->hostPtr<void>();
        capacity_ = source->capacity_;
        count_ = source->count_;
        shape_ = source->shape_;
        allocated_ = source->allocated_;
        dtype_ = source->dtype_;
    }
    /**
     * \brief this Tensor is a DEEP COPY of source
     * \param source
     * \param shape_offset
     */
    void deepCopyOffsetFrom(Tensor &source, const vector<int> &shape_offset) {
        assert(source.allocted());
        // don't need alloc()
        shape_offset_ = shape_offset;
        shape_base_ = source.shape_;
        // deep Copy
        host_ptr_ = source.hostPtr<void>();
        allocated_ = source.allocated_;
        dtype_ = source.dtype_;
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
        return ((Dtype *)host_ptr_)[offset(batch, head, sequence, dimension)];
    }
    template <typename Dtype>
    Dtype dataAtDangerously(const int offset) const {
        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
        return ((Dtype *)host_ptr_)[offset];
    }

    template <typename Dtype>
    Dtype dataAt(const vector<int> &index) const {
            //        return hostPtr<Dtype>()[offset(index)];
        return ((Dtype *)host_ptr_)[offset(index)];
    }

    template <typename Dtype>
    Dtype* ptrAt(const vector<int> &index) const {
        return ((Dtype *)host_ptr_ + offset(index));
    }

    template <typename Dtype>
    Dtype* ptrAt(const int batch, const int head, const int sequence, const int dimension) const {
        return ((Dtype *)host_ptr_ + offset(batch, head, sequence, dimension));
    }

    //    template <typename Dtype>
    //    void setDataAt(const int n, const int c, const int h, const int w, Dtype value) {
    //        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
    //        typed_ptr[offset(n, c, h, w)] = value;
    //    }
    template <typename Dtype>
    void setDataAt(const int batch, const int head, const int sequence, const int dimension, Dtype value) {
        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
        typed_ptr[offset(batch, head, sequence, dimension)] = value;
    }
    template <typename Dtype>
    void setDataAtDangerously(const int offset, Dtype value) const {
        //        return hostPtr<Dtype>()[offset(n, c, h, w)];
        ((Dtype *)host_ptr_)[offset] = value;
    }

    template <typename Dtype>
    void setDataAt(const vector<int> &index, Dtype value) {
        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
        typed_ptr[offset(index)] = value;
    }

    DataType dtype() const {
        return dtype_;
    }

    int cntSize() {
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
    vector<int> shape_base() const {
        return shape_base_;
    }

    template <typename Dtype>
    void checkData() {
        // n c h w
        int N = num();
        int C = channels();
        int H = height();
        int W = width();
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
            std::cout<<"\n[ERROR]:" << name() << ": shape:[" << num() << " " << channels() << " " << height() << " " << width() << "] has Nan" << std::endl;
            //printData<Dtype>();
        }
    }





    /*TEST*/
    void printShape(){
        std::cout << name() << ": shape:[" << num() << " " << channels() << " " << height() << " " << width() << "]" << std::endl;
    }

    template <typename Dtype>
    void printData() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << num() << " " << channels() << " " << height() << " " << width() << "]" << std::endl;
        // n c h w
        int N = num();
        int C = channels();
        int H = height();
        int W = width();
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
    void printAVG() {
        float sum = 0;
        // n c h w
        int N = num();
        int C = channels();
        int H = height();
        int W = width();
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

    template <class Dtype>
    void fullData(Dtype value) {
        for (int n = 0; n < num(); ++n) {
            for (int c = 0; c < channels(); ++c) {
                for (int h = 0; h < height(); ++h) {
                    for (int w = 0; w < width(); ++w) {
                        setDataAt<Dtype>(n, c, h, w, value);
                    }
                }
            }
        }
    }

    void fullDataTest() {
        for (int n = 0; n < num(); ++n) {
            for (int c = 0; c < channels(); ++c) {
                for (int h = 0; h < height(); ++h) {
                    for (int w = 0; w < width(); ++w) {
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
    string name_;
//    int byte_width_; // 32/16/8/4 //enum
    DataType dtype_;
    Backend *backend_;
    void *host_ptr_;
    void *device_ptr_;

    // shared_ptr<HostMemory> data_; //存放数据
    // shared_ptr<HostMemory> diff_; //存放梯度  //TODO: not need for "inference"; only define; do not use. DELITE
    // shared_ptr<HostMemory> shape_data_; //Tensor形状，N K H W //4*sizeofint

    // TODO device_data_?

    vector<int> shape_; // 保存 N K H W
    int capacity_;      // 元素个数 申请内存的总长度相关
    int count_;         // 当前元素数

    int allocated_ = 0;


    // shadow tensor if;
    vector<int> shape_offset_;
    vector<int> shape_base_;
};
} // namespace mllm
#endif // MLLM_TENSOR_H
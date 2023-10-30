
#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H

#include <climits>
#include <string>
#include "MemoryManager.hpp"
#include "Backend.hpp"
#include <iostream>
#include <cstdio>
#include <iomanip>

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
        if (host_ptr_ != nullptr  && allocated_) {
            backend_->free(host_ptr_);
            allocated_ = false;
        }
    }
    explicit Tensor(const int num, const int channels, const int height, const int width); // N C H W like Caffe //TODO add param: HostMemory; NCHW_Type?
    explicit Tensor(const vector<int> &shape);
    // void SetBackend(shared_ptr<Backend> bn){
    //     backend_= bn;
    // };
    Backend *backend() const {
        return backend_;
    }
    void setBackend(Backend *bn) {
        backend_ = bn;
    };
    void setDtype(mllm_dtype dtype) {
        dtype_ = dtype;
    }

    //    bool reshape(const int num, const int channels, const int height, const int width);
    bool reshape(const int batch, const int head, const int sequence, const int dimension);
    bool reshape(const vector<int> &shape);

    void alloc();
    void alloc(mllm_dtype dtype) {
        dtype_ = dtype;
        alloc();
    }

    void free(){
        if (host_ptr_ != nullptr && allocated_) {
            backend_->free(host_ptr_);
            allocated_ = false;
        }
    }

    void update();

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
    inline string shapeString() const {
        ostringstream stream;
        for (int i : shape_) {
            stream << i << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
    }
    inline int canonicalAxisIndex(int axis_index) const {
        CHECK_GE(axis_index, -numAxes())
            << "axis " << axis_index << " out of range for " << numAxes()
            << "-D Tensor with shape " << shapeString();
        CHECK_LT(axis_index, numAxes())
            << "axis " << axis_index << " out of range for " << numAxes()
            << "-D Tensor with shape " << shapeString();
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
    inline int offset(const int batch, const int head = 0, const int sequence = 0,
                      const int dimension = 0) const {
        // batch, head, sequence, dimension
        CHECK_GE(batch, 0);
        CHECK_LE(batch, num());
        CHECK_GE(channels(), 0);
        CHECK_LE(head, channels());
        CHECK_GE(height(), 0);
        CHECK_LE(sequence, height());
        CHECK_GE(width(), 0);
        CHECK_LE(dimension, width());
        return ((batch * channels() + head) * height() + sequence) * width() + dimension;
    }

    inline int offset(const vector<int> &indices) const {
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

    template <typename Dtype>
    Dtype *hostPtr() {
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
    void setDataAt(const vector<int> &index, Dtype value) {
        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
        typed_ptr[offset(index)] = value;
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

    mllm_dtype dtype() const {
        return dtype_;
    }

    float dtypeSize() {
        switch (dtype_) {
        case MLLM_TYPE_F32:
            return sizeof(float);
        case MLLM_TYPE_F16:
            return sizeof(short);
        case MLLM_TYPE_I32:
            return sizeof(int);
        case MLLM_TYPE_I16:
            return sizeof(short);
        case MLLM_TYPE_I8:
            return sizeof(char);
            // TODO WRONG?
        case MLLM_TYPE_Q4_0:
            return (sizeof(block_q4_0)) / (QK4_0 / 2);
        case MLLM_TYPE_Q4_K:
            return (sizeof(block_q4_K)) / (QK_K / 2);
        case MLLM_TYPE_Q8_0:
            return (sizeof(block_q8_0)) / (QK8_0);
        case MLLM_TYPE_Q8_K:
            return (sizeof(block_q8_K)) / (QK_K);
        }
    }
//
//    void setByteWidth(int bw) {
//        byte_width_ = bw;
//    }
    // TODO:Name?

    void setName(string name) {
        name_ = name;
    }

    string name() const {
        return name_;
    }

    bool allocted() const {
        return allocated_;
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
    // shared_ptr<Backend> backend_;
//    int byte_width_; // 32/16/8/4 //enum
    mllm_dtype dtype_;
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

    bool allocated_ = false;
    // bn
};
} // namespace mllm
#endif // MLLM_TENSOR_H
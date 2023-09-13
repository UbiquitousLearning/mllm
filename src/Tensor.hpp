
#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H



#include <climits>
#include "MemoryManager.hpp"
#include "Backend.hpp"

const int kMaxAxes = 32;

namespace mllm {

    class Tensor {
    public:
        // Tensor():data_(), diff_(), capacity_(0){}
        Tensor():capacity_(0), bytewidth_(sizeof(float)){}
        Tensor(Backend* bn):backend_(bn), capacity_(0), bytewidth_(sizeof(float)){}
        explicit Tensor(const int num, const int channels, const int height, const int width); //N C H W like Caffe //TODO add param: HostMemory; NCHW_Type?
        explicit Tensor(const vector<int>& shape);
        // void SetBackend(shared_ptr<Backend> bn){
        //     backend_= bn;
        // };
        void SetBackend(Backend* bn){
            backend_= bn;
        };

        bool Reshape(const int num, const int channels, const int height,const int width);
        bool Reshape(const vector<int>& shape);

        void Alloc();

        // const float* cpu_data() const; //静态访问
        // const Dtype* cpu_diff() const;

        template <typename Dtype>
        const Dtype* cpu_data() const{
            return (const Dtype*)host_ptr_;
        }

        // void set_cpu_data(Dtype* data);
        // void set_cpu_diff(Dtype* diff);

        void Update();

/*
        //TODO 针对data的计算，或者加一个参数is_diff来支持data&diff
	    Tensor& operator[] (int i);//??
        Tensor& operator= (double val);
        Tensor& operator*= (const double i);
        friend Tensor operator*(Tensor A, Tensor B);
        friend Tensor operator/(Tensor A, Tensor B);
        friend Tensor operator/(Tensor A, double val);
        friend Tensor operator*(double num, Tensor B);
        friend Tensor operator+(Tensor A, Tensor B);
        friend Tensor operator+(Tensor A, double val);
        friend Tensor sqrt(Tensor A);
        friend Tensor square(Tensor A);
*/

        // Deprecated legacy shape accessor num: use shape(0) instead.
        inline int num() const { return LegacyShape(0); }
        // Deprecated legacy shape accessor channels: use shape(1) instead.
        inline int channels() const { return LegacyShape(1); }
        // Deprecated legacy shape accessor height: use shape(2) instead.
        inline int height() const { return LegacyShape(2); }
        // Deprecated legacy shape accessor width: use shape(3) instead.
        inline int width() const { return LegacyShape(3); }

        inline int count() const { return count_; }
        inline int num_axes() const { return shape_.size(); }
        inline string shape_string() const {
            ostringstream stream;
            for (int i : shape_) {
                stream << i << " ";
            }
            stream << "(" << count_ << ")";
            return stream.str();
        }
        inline int CanonicalAxisIndex(int axis_index) const {
            CHECK_GE(axis_index, -num_axes())
                << "axis " << axis_index << " out of range for " << num_axes()
                << "-D Tensor with shape " << shape_string();
            CHECK_LT(axis_index, num_axes())
                << "axis " << axis_index << " out of range for " << num_axes()
                << "-D Tensor with shape " << shape_string();
            if (axis_index < 0) {
                return axis_index + num_axes();
            }
            return axis_index;
        }
        inline const vector<int>& shape() const { return shape_; }
        inline int shape(int index) const {
            return shape_[CanonicalAxisIndex(index)];
        }
        inline int LegacyShape(int index) const {
            CHECK_LE(num_axes(), 4)
                << "Cannot use legacy accessors on Tensors with > 4 axes.";
            CHECK_LT(index, 4);
            CHECK_GE(index, -4);
            if (index >= num_axes() || index < -num_axes()) {
                // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
                // indexing) -- this special case simulates the one-padding used to fill
                // extraneous axes of legacy Tensors.
                return 1;
            }
            return shape(index);
        }

        inline int offset(const int n, const int c = 0, const int h = 0,
                          const int w = 0) const {
            CHECK_GE(n, 0);
            CHECK_LE(n, num());
            CHECK_GE(channels(), 0);
            CHECK_LE(c, channels());
            CHECK_GE(height(), 0);
            CHECK_LE(h, height());
            CHECK_GE(width(), 0);
            CHECK_LE(w, width());
            return ((n * channels() + c) * height() + h) * width() + w;
        }

        inline int offset(const vector<int>& indices) const {
            CHECK_LE(indices.size(), num_axes());
            int offset = 0;
            for (int i = 0; i < num_axes(); ++i) {
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
         *        of other (and die otherwise); if true, Reshape this Tensor to other's
         *        shape if necessary
         */
        void CopyFrom(const Tensor& source, bool copy_diff = false,
                      bool reshape = false);

        template <typename Dtype>
        inline Dtype data_at(const int n, const int c, const int h,
                             const int w) const {
            return cpu_data<Dtype>()[offset(n, c, h, w)];
        }

        // inline Dtype diff_at(const int n, const int c, const int h,
        //                      const int w) const {
        //     return cpu_diff()[offset(n, c, h, w)];
        // }

        template <typename Dtype>
        inline Dtype data_at(const vector<int>& index) const {
            return cpu_data<Dtype>()[offset(index)];
        }

        // inline Dtype diff_at(const vector<int>& index) const {
        //     return cpu_diff()[offset(index)];
        // }


        void PrintData();

        int ByteWidth() const {
            return bytewidth_;
        }
        
    private:
        // shared_ptr<Backend> backend_;
        int bytewidth_;//32/16/8/4 //enum
        Backend* backend_;
        void* host_ptr_;
        void* device_ptr_;

        // shared_ptr<HostMemory> data_; //存放数据 
        // shared_ptr<HostMemory> diff_; //存放梯度  //TODO: not need for "inference"; only define; do not use. DELITE
        // shared_ptr<HostMemory> shape_data_; //Tensor形状，N K H W //4*sizeofint

        //TODO device_data_?

        vector<int> shape_; //保存 N K H W
        int capacity_; //元素个数 申请内存的总长度相关
        int count_; //当前元素数
        //bn
    };
}
#endif //MLLM_TENSOR_H
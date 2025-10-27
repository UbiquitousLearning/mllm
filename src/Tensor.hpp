#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H
// #include <climits>
#include "DataType.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <Types.hpp>
#include <assert.h>
// #include <sys/stat.h>

#include "TensorImpl.hpp"
namespace mllm {
class Backend;
class Module;

/* Tensor is the baseic data structure of mllm. It is used to store the data of the model's weights and activations(the intermediate data of the calculation).
 * The Tensor class contained 3 kinds of Tensors: BasicTensor, ChildTensor. AggregatedTensor.
 *
 * I）These are some basic attributes of Tensors:
 * - The data of Tensor is stored in the host memory started on 'host_ptr_'.
 * - The shape of Tensor is stored in 'shape_', which is a vector of integers, relay on private variable'ctype_'. e.g. [2, 3, 4, 5] for 4-D Tensor.
 * - Private variable 'ctype_' indicates the order of the dimensions in the memory.
 *   e.g. ctype_ == BSHD, the order of the dimensions in the memory is: batch, sequence, head, dimension.
 *        ctype_ == BHDS, the order of the dimensions in the memory is: batch, head, dimension, sequence.
 *        ctype_ == BCTHW, the order of the dimensions in the memory is: batch,  channel, time, height, width, which is used for 5-D Tensor.
 * - The data type of Tensor is 'dtype_', which can be MLLM_TYPE_FP32, MLLM_TYPE_FP16, MLLM_TYPE_Q4_K, etc.
 * - Private variable 'transed_' indicates whether the Tensor has been transposed. See method `transShape` below for more information.
 *   e.g. origin tensor's ctype_ == BSHD, transed_ == false,
 *        transed tensor's ctype_ == BHDS, transed_ == true.
 *
 * II）These are some attributes used for ChildTensor:
 * The ChildTensor is a Tensor which is a part of another Tensor(called 'MasterTensor'), and 'host_ptr_' of ChildTensor is the same as the 'host_ptr_' of MasterTensor.
 * Each ChlidTensor only have one MasterTensor, but each MasterTensor can have multiple ChildTensors.
 * - Private variable 'shape_master_' indicates the shape of MasterTensor.
 * - Private variable 'master_tensor_' indicates the MasterTensor of ChildTensor.
 * - Private variable 'shape_offset_' indicates the offset of each dimension of ChildTensor compared to MasterTensor.
 *   e.g. MasterTensor's shape is [2, 3, 4, 5], ChildTensor's shape is [1, 2, 3, 4], then shape_offset_ = [1, 0, 0, 0].
 * - Private variable 'child_tensors_' indicates the ChildTensors of MasterTensor.
 * = Private variable 'undiffusion_' indicates whether the 'transed_' of ChildTensor can be diffussion to it's MasterTensor.
 *
 * III）These are some attributes used for AggregatedTensor:
 * The AggregatedTensor is a Tensor which is a aggregation of multiple Tensors.
 * The 'host_ptr_' of AggregatedTensor is NULL nad not used.
 * - Private variable 'aggregated_tensors_' indicates the Tensors which are aggregated by AggregatedTensor.
 * - Private variable 'aggregated_dim_' indicates the dimension of AggregatedTensor. e.g. HEAD, SEQUENCE, DIMENSION.
 * - Private variable 'aggregated_dims_' indicates the sumed size of each dimension of each Tensors.
 *   e.g. aggregated_dim_ = SEQUENCE; aggregated_dims_ = [2, 3];
 *        then the size of SEQUENCE dimension of the first Tensor is 2, the size of SEQUENCE dimension of the second Tensor is 1.
 *
 */

class QuantParam {
public:
    QuantParam() :
        scale(0.0f), zero_point(0) {
    }
    QuantParam(float s, int zp) :
        scale(s), zero_point(zp) {
    }

    float scale;    // quantization scale
    int zero_point; // quantization zero point
};

class Tensor : public std::enable_shared_from_this<Tensor> {
protected:
    std::shared_ptr<TensorImpl> impl_; // 核心：使用shared_ptr管理实现
private:
    // used for ChildTensor
    vector<uint64_t> shape_offset_;
    vector<uint64_t> shape_master_;
    std::weak_ptr<Tensor> master_tensor_;
    vector<std::weak_ptr<Tensor>> child_tensors_;

    // AggregatedTensor相关
    bool aggregated_ = false;
    bool allow_aggregated_ = true;
    vector<shared_ptr<Tensor>> aggregated_tensors_;
    Tensor *deaggregated_tensor_ = nullptr;
    Chl aggregated_dim_;
    vector<int> aggregated_dims_;

    vector<float> seq_means_;

    TensorType ttype_ = NORMAL_TENSOR;
    uint32_t uuid_ = 4294967295U;
    TensorType xnn_tensor_type_ = TensorType::NORMAL_TENSOR;

public:
    int cache_seq_len_;
    QuantParam quant_param;
    bool inited() {
        return impl_ != nullptr && impl_->host_ptr_ != nullptr && impl_->name_ != "";
    }

public:
    // 拷贝语义：默认浅拷贝（共享实现）
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;

    // 移动语义
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    /* 构造系列 */
    Tensor() :
        impl_(std::make_shared<TensorImpl>()) {
    }
    explicit Tensor(Backend *bn) :
        impl_(std::make_shared<TensorImpl>(bn)) {
    }

    /**
     * \brief build 4-D Tensor with four dimensions: [batch, head, sequence, dimension].
     *        The four dimension designed for Transformer-based LLMs：
     * \param batch  batch size
     * \param head   multi-head number
     * \param sequence  tokens numbers in a sequence
     * \param dimension the hidden size
     */
    // explicit Tensor(int batch, int head, int sequence, int dimension);
    explicit Tensor(int batch, int head, int sequence, int dimension, Backend *bn, bool do_alloc = true);
    explicit Tensor(int batch, int head, int sequence, int dimension, BackendType bn_type = MLLM_CPU, bool do_alloc = true);
    /**
     * \brief build Tensor with shape.
     *        [ATTENTION] this function only used to build Tensor which other Tensor's shape !!!
     *        e.g. Tensor other_tensor(origin_tensor->shape());
     * \param shape
     */
    explicit Tensor(const vector<int> &shape);

    Tensor(int value, Backend *bn);

    Tensor(int value, BackendType bn_type = MLLM_CPU);

    Tensor(vector<float> values, BackendType bn_type = MLLM_CPU);

    ~Tensor() {
        if (auto master = master_tensor_.lock()) {
            auto &children = master->childTensors();
            children.erase(
                std::remove_if(children.begin(), children.end(),
                               [this](const std::weak_ptr<Tensor> &wp) {
                                   return wp.expired() || wp.lock().get() == this;
                               }),
                children.end());
        }
    }

public:
    static TensorStatus tensor_status;
    /**
     * \brief reshape 4-D Tensor with four dimensions: [batch, head, sequence, dimension].
     *        The four dimension designed for Transformer-based LLMs：
     * \param batch  batch size
     * \param head   multi-head number
     * \param sequence  tokens numbers in a sequence
     * \param dimension the hidden size
     * \return whether reshape success.
     */
    bool reshape(int batch, int head, int sequence, int dimension);

    /**
     * \brief alloc the memory of Tensor.
     * \param dtype the data type of this Tensor. e.g. MLLM_TYPE_F32, MLLM_TYPE_Q4_K
     */
    void alloc(DataType dtype) {
        impl_->dtype_ = dtype;
        alloc();
    }
    void alloc();
    void alloc(vector<unsigned int> alloc_size);

    /**
     * \brief free the memory of Tensor.
     */
    void free() {
        if (aggregated_) { return; }
        if (masterTensor() == nullptr) {
            impl_->free();
        }
    }
    void unload() {
        if (impl_) {
            impl_->unload();
        }
    }

    /**
     * \brief  get the number of bytes occupied by Tensor's data in memory.
     *         depends on the total dimension sizes and data type.
     * \return the number of bytes occupied by Tensor's data in memory
     */
    size_t size() const {
        return impl_->capacity_ * dtypeSize();
    }
    /**
     * \brief get the size of the corresponding dimension for 4-D Tensor, contains: batch, head, sequence, dimension.
     *        each Tensor has private variable 'ctype_', which indicates the order of the dimensions in the memory.
     *        e.g. ctype_ == BSHD, the order of the dimensions in the memory is: batch, sequence, head, dimension.
     *             ctype_ == BHDS, the order of the dimensions in the memory is: batch, head, dimension, sequence.
     *        so batch() is not equal to shape(0), it depends on the value of ctype_.
     *        no matter what the value of ctype_ is, these functions will return the size of the corresponding dimension.
     * \return the size of the corresponding dimension
     */
    std::map<Chl, int> &chls() {
        return impl_->chls_;
    }

    int batch() {
        return impl_->batch();
    }
    int head() {
        return impl_->head();
    }
    int sequence() {
        return impl_->sequence();
    }
    int dimension() {
        return impl_->dimension();
    }

    /**
     * \brief get the totol size of all dimensions.
     *        mostly, count() == batch() * head() * sequence() * dimension()
     * \return the totol size of all dimensions
     */
    int count() const {
        return impl_->count_;
    }
    string shapeString() const {
        std::ostringstream stream;
        for (int i : impl_->shape_) {
            stream << i << " ";
        }
        stream << "(" << impl_->count_ << ")";
        return stream.str();
    }
    int legacyShape(int index) const {
        return impl_->legacyShape(index);
    }
    /**
     * \brief get the offset compared to 'host_ptr_'.
     *        depends on the total dimension sizes and data type.
     *        if the Tensor has a "MasterTensor", the offset will be calculated based on the "MasterTensor".
     * \param b batch index
     * \param h head index
     * \param s sequence index
     * \param d deimension index
     * \return the offset compared to 'host_ptr_'.
     */
    uint64_t offset(const int b, const int h = 0, const int s = 0,
                    const int d = 0) {
        // batch, head, sequence, dimension
        if (shape_offset_.size() == 4 && shape_master_.size() == 4) {
            auto base_batch_ = shape_master_[0];
            auto base_head_ = shape_master_[1];
            auto base_sequence_ = shape_master_[2];
            auto base_dimension_ = shape_master_[3];
            auto b_ = (b + shape_offset_[0]) % base_batch_;
            auto h_ = (h + shape_offset_[1]) % base_head_;
            auto s_ = (s + shape_offset_[2]) % base_sequence_;
            auto d_ = (d + shape_offset_[3]) % base_dimension_;
            switch (impl_->ctype_) {
            case BHSD:
                return ((b_ * base_head_ + h_) * base_sequence_ + s_) * base_dimension_ + d_;
            case BSHD:
                return ((b_ * base_sequence_ + s_) * base_head_ + h_) * base_dimension_ + d_;
            case BHDS:
                return ((b_ * base_head_ + h_) * base_dimension_ + d_) * base_sequence_ + s_;
            case BDHS:
                return ((b_ * base_dimension_ + d_) * base_head_ + h_) * base_sequence_ + s_;
            case SBHD:
                return ((s_ * base_batch_ + b_) * base_head_ + h_) * base_dimension_ + d_;
            case DBHS:
                return ((d_ * base_batch_ + b_) * base_head_ + h_) * base_sequence_ + s_;
            default:
                break;
            }
        } else {
            switch (impl_->ctype_) {
            case BHSD:
                return ((b * impl_->shape_[1] + h) * impl_->shape_[2] + s) * impl_->shape_[3] + d;
            case BSHD:
                return ((b * impl_->shape_[1] + s) * impl_->shape_[2] + h) * impl_->shape_[3] + d;
            case BHDS:
                return ((b * impl_->shape_[1] + h) * impl_->shape_[2] + d) * impl_->shape_[3] + s;
            case BDHS:
                return ((b * impl_->shape_[1] + d) * impl_->shape_[2] + h) * impl_->shape_[3] + s;
            case SBHD:
                return ((s * impl_->shape_[1] + b) * impl_->shape_[2] + h) * impl_->shape_[3] + d;
            case DBHS:
                return ((d * impl_->shape_[1] + b) * impl_->shape_[2] + h) * impl_->shape_[3] + s;
            default:
                break;
            }
        }
        return -1;
    }
    /**
     * \brief get the offset compared to 'host_ptr_'.
     * \param indices the indexes of each dimension, must be {batch, head, sequence, dimension}
     * \return the offset compared to 'host_ptr_'.
     */
    int offset(const vector<int> &indices) {
        if (shape_offset_.size() == 4 && shape_master_.size() == 4) {
            return offset(indices[0], indices[1], indices[2], indices[3]);
        } else {
            int offset = 0;
            for (int i = 0; i < impl_->numAxes(); ++i) {
                offset *= impl_->shape(i);
                if (indices.size() > i) {
                    offset += indices[i];
                }
            }
            return offset;
        }
    }

    int sequenceSkipDim() const {
        if (!master_tensor_.expired()) {
            auto master = master_tensor_.lock();
            if (master && !master->master_tensor_.expired()) {
                auto grandmaster = master->master_tensor_.lock();
                auto shape = grandmaster->impl_->shape_;
                if (grandmaster->impl_->ctype_ == BSHD) {
                    return shape[3] * shape[2];
                } else if (grandmaster->impl_->ctype_ == BHSD) {
                    return shape[3];
                } else if (grandmaster->impl_->ctype_ == BHDS) {
                    return shape[3];
                } else if (grandmaster->impl_->ctype_ == BDHS) {
                    return shape[3] * impl_->shape_[2];
                } else {
                    std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                    return -1;
                }
            } else if (master) {
                auto shape = master->impl_->shape_;
                if (master->impl_->ctype_ == BSHD) {
                    return shape[3] * shape[2];
                } else if (master->impl_->ctype_ == BHSD) {
                    return shape[3];
                } else if (master->impl_->ctype_ == BHDS) {
                    return shape[3];
                } else if (master->impl_->ctype_ == BDHS) {
                    return shape[3] * impl_->shape_[2];
                } else {
                    std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                    return -1;
                }
            }
        } else {
            if (impl_->ctype_ == BSHD) {
                return impl_->shape_[3] * impl_->shape_[2];
            } else if (impl_->ctype_ == BHSD) {
                return impl_->shape_[3];
            } else if (impl_->ctype_ == BHDS) {
                return impl_->shape_[3];
            } else if (impl_->ctype_ == BDHS) {
                return impl_->shape_[3] * impl_->shape_[2];
            } else if (impl_->ctype_ == DBHS) {
                return impl_->shape_[3] * impl_->shape_[2];
            } else if (impl_->ctype_ == SBHD) {
                return impl_->shape_[3] * impl_->shape_[2];
            } else {
                std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                return -1;
            }
            // return shape_[3]*shape_[2];
        }
        return -1;
    }

    /**
     * \brief obtain the raw pointer to the first address where tensor stores data.
     * \return the pointer(void *) to the first address where tensor stores data.
     */
    void *rawHostPtr() const {
        return impl_->host_ptr_;
    }

    /**
     * \brief obtain the pointer to the first address where tensor stores data.
     * \tparam Dtype float, mllm_fp16_t, etc.
     * \return the pointer to the first address where tensor stores data.
     */
    template <typename Dtype>
    Dtype *hostPtr() const {
        return (Dtype *)impl_->host_ptr_;
    }

    /**
     * @brief 获取设备内存的通用描述符。
     * @return DeviceMemory& 对设备内存描述符的引用。
     */
    DeviceMemory &device_memory() {
        if (backend() == nullptr || backend()->type() == MLLM_CPU) {
            throw std::runtime_error("Device memory is not available for CPU backend.");
        }
        return impl_->device_memory_;
    }

    /**
     * @brief 获取设备内存的通用描述符 (const 版本)。
     * @return const DeviceMemory& 对设备内存描述符的常量引用。
     */
    const DeviceMemory &device_memory() const {
        if (backend() == nullptr || backend()->type() == MLLM_CPU) {
            throw std::runtime_error("Device memory is not available for CPU backend.");
        }
        return impl_->device_memory_;
    }
    /**
     * \brief Get the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param batch Batch index
     * \param head Head index
     * \param sequence Sequence index
     * \param dimension Dimension index
     * \return Returns the data at the specified position.
     */
    template <typename Dtype>
    Dtype dataAt(const int batch, const int head, const int sequence, const int dimension) {
        if (!aggregated_) {
            return ((Dtype *)impl_->host_ptr_)[offset(batch, head, sequence, dimension)];
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->dataAt<Dtype>(b, h, s, d);
        }
    }
    template <typename Dtype>
    Dtype &d(const int batch, const int sequence, const int head, const int dimension) {
        if (!aggregated_) {
            return ((Dtype *)impl_->host_ptr_)[offset(batch, head, sequence, dimension)];
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->d<Dtype>(b, s, h, d);
        }
    }
    /**
     * \brief Get the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param index A vector containing four elements, representing the indices of batch, head, sequence, and dimension respectively.
     * \return Returns the data at the specified position.
     */
    template <typename Dtype>
    Dtype dataAt(const vector<int> &index) {
        return dataAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    /**
     * \brief Get the pointer to the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param batch Batch index
     * \param head Head index
     * \param sequence Sequence index
     * \param dimension Dimension index
     * \return Returns the pointer to the data at the specified position.
     */
    template <typename Dtype>
    Dtype *ptrAt(const int batch, const int head, const int sequence, const int dimension) {
        if (!aggregated_) {
            return ((Dtype *)impl_->host_ptr_ + offset(batch, head, sequence, dimension));
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->ptrAt<Dtype>(b, h, s, d);
        }
    }
    template <typename Dtype>
    Dtype *p(const int batch, const int sequence, const int head, const int dimension) {
        if (!aggregated_) {
            return ((Dtype *)impl_->host_ptr_ + offset(batch, head, sequence, dimension));
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->p<Dtype>(b, s, h, d);
        }
    }
    /**
     * \brief Get the pointer to the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param index A vector containing four elements, representing the indices of batch, head, sequence, and dimension respectively.
     * \return Returns the pointer to the data at the specified position.
     */
    template <typename Dtype>
    Dtype *ptrAt(const vector<int> &index) const {
        return ptrAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    /**
     * \brief Set the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param batch Batch index
     * \param head Head index
     * \param sequence Sequence index
     * \param dimension Dimension index
     * \param value The value to be set
     */
    template <typename Dtype>
    void setDataAt(const int batch, const int head, const int sequence, const int dimension, Dtype value) {
        if (!aggregated_) {
            Dtype *typed_ptr = static_cast<Dtype *>(impl_->host_ptr_);
            typed_ptr[offset(batch, head, sequence, dimension)] = value;
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            aggregated_tensors_[tensor_id]->setDataAt<Dtype>(b, h, s, d, value);
        }
    }
    /**
     * \brief Set the data at the specified position.
     * \tparam Dtype Data type, such as float, mllm_fp16_t, etc.
     * \param index A vector containing four elements, representing the indices of batch, head, sequence, and dimension respectively.
     * \param value The value to be set
     */
    template <typename Dtype>
    void setDataAt(const vector<int> &index, Dtype value) {
        setDataAt(index[0], index[1], index[2], index[3], value);
    }
    /**
     * \brief Get the 'dtype' at the specified position.
     * \param batch Batch index
     * \param head Head index
     * \param sequence Sequence index
     * \param dimension Dimension index
     * \return data type , e.g. MLLM_TYPE_F32, MLLM_TYPE_Q4_K
     */
    DataType dtypeAt(const int batch, const int head, const int sequence, const int dimension) {
        if (!aggregated_) {
            return impl_->dtype_;
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->impl_->dtype_;
        }
    }

    Backend *backend() const {
        return impl_->backend_;
    }
    void setBackend(Backend *bn) {
        impl_->backend_ = bn;
    };

    DataType dtype() const {
        return impl_->dtype_;
    }
    void setDtype(DataType dtype) {
        impl_->dtype_ = dtype;
    }

    TensorType ttype() const {
        return ttype_;
    }
    void setTtype(TensorType ttype) {
        ttype_ = ttype;
    }

    std::vector<int> shape() const {
        std::vector<int> shape_int(impl_->shape_.size());
        std::transform(impl_->shape_.begin(), impl_->shape_.end(), shape_int.begin(), [](uint64_t val) {
            return static_cast<int>(val);
        });
        return shape_int;
    }
    int shape(Chl axis) const {
        switch (axis) {
        case Chl::BATCH:
            return impl_->shape_[impl_->chls_[BATCH]];
        case Chl::HEAD:
            return impl_->shape_[impl_->chls_[HEAD]];
        case Chl::SEQUENCE:
            return impl_->shape_[impl_->chls_[SEQUENCE]];
        case Chl::DIMENSION:
            return impl_->shape_[impl_->chls_[DIMENSION]];
            // case TIME:
            //     return impl_->shape_[impl_->chls_[TIME]];
            // case HEIGHT:
            return impl_->shape_[impl_->chls_[HEIGHT]];
        case Chl::WIDTH:
            return impl_->shape_[impl_->chls_[WIDTH]];
        // case CHANNLE:
        //     return impl_->shape_[impl_->chls_[CHANNLE]];
        default:
            throw std::invalid_argument("Invalid axis for shape retrieval");
        }
    }

    ChlType ctype() const {
        return impl_->ctype_;
    }
    void setCtype(ChlType type) {
        impl_->ctype_ = type;
        switch (impl_->ctype_) {
        case BHSD:
            impl_->chls()[BATCH] = 0;
            impl_->chls()[HEAD] = 1;
            impl_->chls()[SEQUENCE] = 2;
            impl_->chls()[DIMENSION] = 3;
            break;
        case BSHD:
            impl_->chls()[BATCH] = 0;
            impl_->chls()[SEQUENCE] = 1;
            impl_->chls()[HEAD] = 2;
            impl_->chls()[DIMENSION] = 3;
            break;
        case BHDS:
            impl_->chls()[BATCH] = 0;
            impl_->chls()[HEAD] = 1;
            impl_->chls()[DIMENSION] = 2;
            impl_->chls()[SEQUENCE] = 3;
            break;
        case SBHD:
            impl_->chls()[SEQUENCE] = 0;
            impl_->chls()[BATCH] = 1;
            impl_->chls()[HEAD] = 2;
            impl_->chls()[DIMENSION] = 3;
            break;
        case BTHWC:
            impl_->chls()[BATCH] = 0;
            impl_->chls()[TIME] = 1;
            impl_->chls()[HEIGHT] = 2;
            impl_->chls()[WIDTH] = 3;
            impl_->chls()[CHANNLE] = 3;
            break;
        case BCTHW:
            impl_->chls()[BATCH] = 0;
            impl_->chls()[CHANNLE] = 1;
            impl_->chls()[TIME] = 2;
            impl_->chls()[HEIGHT] = 3;
            impl_->chls()[WIDTH] = 3;
            break;
        default:
            break;
        }
    }
    size_t cntSize() {
        return DataTypeSize(impl_->dtype_, impl_->count_);
    }
    int dtypeSize() const {
        return DataTypeSize(impl_->dtype_, 1);
    }
    int dtypeSize(int size) {
        return DataTypeSize(impl_->dtype_, size);
    }
    void setName(string name) {
        impl_->name_ = name;
    }
    string name() const {
        return impl_->name_;
    }
    int allocted() const {
        return impl_->allocated_;
    }

    /**
     * \brief Transforms the shape of the Tensor based on the provided dimensions.
     * \param dim_a The first dimension to be transformed. Default is SEQUENCE.
     * \param dim_b The second dimension to be transformed. Default is DIMENSION.
     * \param undiffusion A boolean flag indicating whether the transformation should be undiffused. Default is false.
     *
     * This function checks the current shape of the Tensor (defined by the private variable 'ctype_') and the provided dimensions to be transformed.
     * If the current shape is BSHD (Batch, Sequence, Head, Dimension) and the dimensions to be transformed are SEQUENCE and DIMENSION,
     * it change 'ctype_' to BHDS (Batch, Head, Dimension, Sequence) format.
     * If the current shape is BCTHW (Batch, Channel, Time, Height, Width) and the dimensions to be transformed are THW and CHANNEL,
     * it rchange 'ctype_'  to BTHWC (Batch, Time, Height, Width, Channel) format.
     * If the current shape is BSHD (Batch, Sequence, Head, Dimension) and the dimensions to be transformed are BATCH and SEQUENCE,
     * it change 'ctype_' to SBHD (Sequence, Batch, Head, Dimension) format.
     * After reshaping, it sets the 'transed_' flag to true and the 'undiffusion_' flag to the provided value.
     *
     * TODO abanden
     */
    void transShape(Chl dim_a = SEQUENCE, Chl dim_b = DIMENSION, bool undiffusion = false) {
        if (dim_a == SEQUENCE && dim_b == DIMENSION && ctype() == BSHD) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            impl_->ctype_ = BHDS;
            auto ori_seq_idx = impl_->chls()[SEQUENCE];
            auto ori_head_idx = impl_->chls()[HEAD];
            auto ori_dim_idx = impl_->chls()[DIMENSION];
            impl_->chls()[HEAD] = ori_seq_idx;
            impl_->chls()[DIMENSION] = ori_head_idx;
            impl_->chls()[SEQUENCE] = ori_dim_idx;
            reshape(b, h, s, d);
            impl_->transed_ = true;
            impl_->undiffusion_ = undiffusion;
        } else if (dim_a == SEQUENCE && dim_b == DIMENSION && ctype() == BHDS) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            impl_->ctype_ = BSHD;
            auto ori_seq_idx = impl_->chls()[SEQUENCE];
            auto ori_head_idx = impl_->chls()[HEAD];
            auto ori_dim_idx = impl_->chls()[DIMENSION];
            impl_->chls()[SEQUENCE] = ori_head_idx;
            impl_->chls()[HEAD] = ori_dim_idx;
            impl_->chls()[DIMENSION] = ori_seq_idx;
            reshape(b, h, s, d);
            impl_->transed_ = false;
            impl_->undiffusion_ = undiffusion;
        } else if (THW == dim_a && dim_b == CHANNLE && ctype() == BCTHW) {
            auto b = batch();
            auto c = channel();
            auto t = time();
            auto h = height();
            auto w = width();
            impl_->ctype_ = BTHWC;
            auto ori_chl_idx = impl_->chls()[CHANNLE];
            auto ori_time_idx = impl_->chls()[TIME];
            auto ori_height_idx = impl_->chls()[HEIGHT];
            auto ori_width_idx = impl_->chls()[WIDTH];
            impl_->chls()[TIME] = ori_chl_idx;
            impl_->chls()[HEIGHT] = ori_time_idx;
            impl_->chls()[WIDTH] = ori_height_idx;
            impl_->chls()[CHANNLE] = ori_width_idx;
            reshape(b, c, t, h, w);
            impl_->transed_ = true;
            impl_->undiffusion_ = undiffusion;
        } else if (dim_a == BATCH && dim_b == SEQUENCE && ctype() == BSHD) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            impl_->ctype_ = SBHD;
            auto ori_batch_idx = impl_->chls()[BATCH];
            auto ori_seq_idx = impl_->chls()[SEQUENCE];
            impl_->chls()[SEQUENCE] = ori_batch_idx;
            impl_->chls()[BATCH] = ori_seq_idx;
            reshape(b, h, s, d);
            impl_->transed_ = true;
            impl_->undiffusion_ = undiffusion;
        }
    }

    /**
     * @brief Copy from a source Tensor.
     *        [ATTENTION] this function only support for Tensors without "MasterTensor".
     * @param source the Tensor to copy from
     */
    void copyFrom(const Tensor &source) {
        assert(masterTensor() == nullptr);
        assert(source.dtype() == dtype());
        assert(source.count() == count());
        memcpy(impl_->host_ptr_, source.impl_->host_ptr_, cntSize());
    }
    void initFrom(const Tensor &source) {
        impl_->dtype_ = source.impl_->dtype_;
        impl_->chls_ = source.impl_->chls_;
        impl_->ctype_ = source.impl_->ctype_;
        impl_->shape_ = source.impl_->shape_;
        impl_->count_ = source.impl_->count_;
        if (source.impl_->host_ptr_ != nullptr) {
            alloc();
        }
    }
    void copyFrom(const shared_ptr<Tensor> &source) {
        assert(masterTensor() == nullptr);
        assert(source->dtype() == dtype());
        assert(source->count() == count());
        memcpy(impl_->host_ptr_, source->impl_->host_ptr_, cntSize());
    }

    void changeCtype(int size = 0) {
        impl_->changeCtype(size);
    }

    bool &transed() {
        return impl_->transed_;
    }
    bool &undiffusion() {
        return impl_->undiffusion_;
    }
    void setUndiffusion(bool undiffusion) {
        impl_->undiffusion_ = undiffusion;
        for (auto &child_tensor : child_tensors_) {
            if (!child_tensor.expired()) {
                child_tensor.lock()->impl_->undiffusion_ = undiffusion;
            }
        }
    }

    vector<std::pair<Chl, Chl>> &transFrom() {
        return impl_->trans_from_;
    }

    bool &shouldInGraphs() {
        return impl_->should_in_graphs_;
    }

    vector<float> &seqMeans() {
        if (!master_tensor_.expired()) {
            return master_tensor_.lock()->seq_means_;
        }
        return seq_means_;
    }

    static Tensor zeros(int batch, int head, int sequence, int dimension, BackendType bn_type = MLLM_CPU) {
        Tensor tensor1(batch, head, sequence, dimension, bn_type, true);
        std::fill(tensor1.hostPtr<float>(), tensor1.hostPtr<float>() + tensor1.count(), 0);
        tensor1.shouldInGraphs() = false;
        return tensor1;
    }
    static Tensor ones(int batch, int head, int sequence, int dimension, BackendType bn_type = MLLM_CPU) {
        Tensor tensor1(batch, head, sequence, dimension, bn_type, true);
        std::fill(tensor1.hostPtr<float>(), tensor1.hostPtr<float>() + tensor1.count(), 1);
        tensor1.shouldInGraphs() = false;
        return tensor1;
    }
    static Tensor full(int batch, int head, int sequence, int dimension, float data, BackendType bn_type = MLLM_CPU) {
        Tensor tensor1(batch, head, sequence, dimension, bn_type, true);
        std::fill(tensor1.hostPtr<float>(), tensor1.hostPtr<float>() + tensor1.count(), data);
        tensor1.shouldInGraphs() = false;
        return tensor1;
    }

    /**
     * \brief Overload the operators.
     * \param data binary data
     * \return Tensor
     */
    Tensor operator+(float data);
    Tensor operator-(float data);
    Tensor operator*(float data);
    Tensor operator/(float data);
    Tensor operator/(double data);
    Tensor operator/(int data);
    Tensor operator~();

    /**
     * \brief Overload the operators.
     * \param other The Other Tensor
     * \return Tensor
     */
    Tensor operator+(Tensor other);
    Tensor operator-(Tensor other);
    Tensor operator*(Tensor other);
    Tensor operator/(Tensor other);

    Tensor mean(Chl axis);

    Tensor view(int b, int h, int s, int d, bool in_place = true);
    Tensor flatten(Chl axis_start, Chl axis_end);
    Tensor transpose(Chl axis0, Chl axis1) {
        return transpose({{axis0, axis1}});
    }
    Tensor transpose(vector<std::pair<Chl, Chl>> axiss);
    Tensor clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d);
    Tensor clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d);
    Tensor clip(vector<int> index, Chl dim);
    Tensor clip(Tensor index, Chl dim);
    Tensor expand(int b, int h, int s, int d);
    static Tensor cat(vector<Tensor> input_tensors, Chl dims);
    static Tensor mm(Tensor input0, Tensor input1);
    Tensor norm(int L_n);
    Tensor where(float value, Chl axis);
    static Tensor range(int start, int end);
    static vector<Tensor> split(Tensor input, std::vector<int> each_dims, Chl split_dim, int same_dim_size = -1);
    vector<Tensor> split(std::vector<int> each_dims, Chl split_dim, int same_dim_size = -1) {
        return split(*this, each_dims, split_dim, same_dim_size);
    }
    Tensor index_put(Tensor value, Tensor indices, bool accumulate);
    void scatter_add(Tensor value, Tensor indices, Chl dim = SEQUENCE);
    void scatter_(Chl dim, Tensor index, float src);
    static vector<Tensor> topk(Tensor input, int k, Chl dim);
    vector<Tensor> topk(int k, Chl dim) {
        return topk(*this, k, dim);
    }
    Tensor sum(Chl dim);
    Tensor argsort();
    Tensor bincount();
    Tensor repeat(Chl dim, int dim_size);
    Tensor masked_fill(Tensor mask, float value);
    static Tensor gather(Tensor input, Tensor index, Chl dim);
    static Tensor zero_like(Tensor input);
    static Tensor flash_attention2_forward(Tensor q, Tensor k, Tensor v, bool is_causal = true);
    static Tensor sage_attention_forward(Tensor q, Tensor k, Tensor v, bool causal_mask = false);
    static Tensor apply_rotary_pos_emb_vision(Tensor input, Tensor rotary_pos_emb);

    // models use only
    static Tensor fuyu_gather_embd(Tensor word, Tensor image_patches, Tensor image_patches_indices);
    static Tensor phi3v_hd_merge(Tensor input, int h_crop, int w_crop);

    /* Functions used for ChildTensor:
     * - shallowCopyFrom
     * - shape_offset
     * - shape_master
     * - masterTensor
     * - setMasterTensor
     * - childTensors
     * - addChildTensor
     */

    // 新增一个方法，用于强制设置指针并转移所有权句柄
    // 这是比将 ParamLoader 设为友元类更清晰的做法
    void setHostPtr(void *ptr, std::shared_ptr<void> memory_handle) {
        // 如果 Tensor 已经持有自己分配的内存，则先释放它
        if (impl_->host_ptr_ != nullptr && impl_->owns_host_ptr_) {
            impl_->free();
        }
        // 接管来自 mmap 的新指针和内存句柄
        impl_->host_ptr_ = ptr;
        impl_->owns_host_ptr_ = false;                    // 标记内存为外部管理
        impl_->memory_handle_ = std::move(memory_handle); // 持有 mmap 句柄
        impl_->allocated_ = count();                      // 标记为已分配状态
    }

    /**
     * @brief 使当前 Tensor 成为 source Tensor 的一个子 Tensor (Shallow Copy)。
     * 它不分配新内存，而是共享 source 的内存。
     * @param source 将要成为父 Tensor 的张量。
     * @param copyshape 如果为 true 且 shape_offset 为空，则直接复制 source 的形状。
     * @param shape_offset 定义子 Tensor 相对于父 Tensor 的维度偏移，用于创建切片(slice)。
     * @param head_rep 用于分组查询注意力(GQA)，表示K/V头的重复次数。
     */
    void shallowCopyFrom(std::shared_ptr<Tensor> source, bool copyshape = true, const vector<int> &shape_offset = {}, int head_rep = 1) {
        // 步骤 0: 初始设置
        // 如果提供了偏移量，则子张量有自己的独立形状，不应复制父张量的形状。
        if (!shape_offset.empty()) {
            copyshape = false;
        }
        setMasterTensor(source); // 建立父子关系的第一步

        // 步骤 1: 同步父子 Tensor 间的内存布局 (ctype)
        reconcileLayouts(source.get());

        // 步骤 2: 核心浅拷贝操作 - 共享数据指针和元数据
        impl_->host_ptr_ = source->hostPtr<void>();
        impl_->memory_handle_ = source->impl_->memory_handle_;
        impl_->owns_host_ptr_ = false;
        impl_->device_memory_ = source->impl_->device_memory_;
        impl_->owns_device_memory_ = false;
        impl_->capacity_ = source->impl_->capacity_;
        impl_->count_ = source->impl_->count_;
        impl_->allocated_ = source->impl_->allocated_;
        impl_->dtype_ = source->impl_->dtype_;
        if (copyshape) {
            impl_->shape_ = source->impl_->shape_;
        }

        // 步骤 3: 处理切片(offset)和GQA逻辑
        if (!shape_offset.empty()) {
            setupShapeForView(source.get(), shape_offset, head_rep);
        }

        // 步骤 4: 维护张量层级结构 (处理孙张量)
        reparentChildTensors(source, shape_offset, head_rep);
        source->addChildTensor(shared_from_this());
    }

    // void shallowCopyFrom(Tensor &source, bool copyshape, const vector<int> &shape_offset = {}, int head_rep = 1) {
    //     // 使用 source.shared_from_this() 从一个已经被 shared_ptr 管理的对象引用中，安全地获取其 shared_ptr。否则有use-after-free 的风险
    //     shallowCopyFrom(source.shared_from_this(), copyshape, shape_offset, head_rep);
    // }
    vector<int> shapeOffset() const {
        std::vector<int> shape_int(shape_offset_.size());
        std::transform(shape_offset_.begin(), shape_offset_.end(), shape_int.begin(), [](uint64_t val) {
            return static_cast<int>(val);
        });
        return shape_int;
    }
    vector<int> shapeMaster() const {
        std::vector<int> shape_int(shape_master_.size());
        std::transform(shape_master_.begin(), shape_master_.end(), shape_int.begin(), [](uint64_t val) {
            return static_cast<int>(val);
        });
        return shape_int;
    }
    vector<uint64_t> &shape_master() {
        return shape_master_;
    }

    std::shared_ptr<Tensor> masterTensor() const {
        return master_tensor_.lock();
    }
    void setMasterTensor(shared_ptr<Tensor> master_tensor) {
        master_tensor_ = master_tensor;
    }

    vector<std::weak_ptr<Tensor>> &childTensors() {
        return child_tensors_;
    }

    void addChildTensor(std::shared_ptr<Tensor> child) {
        auto it = std::find_if(child_tensors_.begin(), child_tensors_.end(),
                               [&](const std::weak_ptr<Tensor> &wp) {
                                   return !wp.expired() && wp.lock() == child;
                               });

        if (it == child_tensors_.end()) {
            child_tensors_.push_back(child);
        }
    }
    /* Functions used for AggregatedTensor:
     * - addTensors
     */
    /**
     * \brief aggregate multiple Tensors to AggregatedTensor, only used for AggregatedTensor.
     * \param ts tensors wanted to be aggregated in AggregatedTensor.
     * \param dim aggregated dimension, can be HEAD, SEQUENCE, DIMENSION.
     */
    void addTensors(vector<shared_ptr<Tensor>> ts, Chl dim) {
        aggregated_ = true;
        aggregated_dim_ = dim;
        aggregated_dims_ = {};
        switch (dim) {
        case HEAD: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->sequence() == sequence());
                assert(t->dimension() == dimension());
                sum += t->head();
                aggregated_dims_.push_back(sum);
            }
            // assert(sum == head());
            break;
        }
        case SEQUENCE: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->head() == head());
                assert(t->dimension() == dimension());
                sum += t->sequence();
                aggregated_dims_.push_back(sum);
            }
            // assert(sum == sequence());
            break;
        }
        case DIMENSION: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->head() == head());
                assert(t->sequence() == sequence());
                sum += t->dimension();
                aggregated_dims_.push_back(sum);
            }
            // assert(sum == dimension());
            break;
        }
        case D_HD:
        case HD: {
            auto sum = 0;
            for (auto &t : ts) {
                sum += t->dimension();
                aggregated_dims_.push_back(sum);
            }
            break;
        }
        case D_DH: {
            auto sum = 0;
            for (auto &t : ts) {
                sum += t->head();
                aggregated_dims_.push_back(sum);
            }
            break;
        }
        default:
            break;
        }
        aggregated_tensors_ = ts;
        for (auto t : aggregated_tensors_) {
            t->deaggregated_tensor_ = this;
        }
    }
    bool aggregated() const {
        return aggregated_;
    }

    vector<shared_ptr<Tensor>> &aggregatedTensors() {
        return aggregated_tensors_;
    }
    void removeAggregatedTensors() {
        aggregated_tensors_.clear();
        aggregated_ = false;
        aggregated_dim_ = BATCH;
        aggregated_dims_.clear();
        deaggregated_tensor_ = nullptr;
    }
    Tensor *deaggregatedTensor() const {
        return deaggregated_tensor_;
    }
    Chl aggregatedDim() const {
        return aggregated_dim_;
    }

    BackendType device() const {
        return impl_->backend_->type();
    }

    Tensor &to(BackendType backend_type);
    Tensor to(DataType dtype) {
        if (dtype == MLLM_TYPE_F16) {
            return half();
        } else if (dtype == MLLM_TYPE_F32) {
            return fp32();
        } else {
            throw std::runtime_error("Unsupported dtype conversion.");
        }
    }

    Tensor &cpu() {
        return to(MLLM_CPU);
    }
    Tensor &qnn() {
        return to(MLLM_QNN);
    }
    Tensor &cl() {
        return to(MLLM_OPENCL);
    }

    Tensor half() {
        if (dtype() == MLLM_TYPE_F16) {
            return *this;
        }
        assert(dtype() == MLLM_TYPE_F32 && "Tensor::half() can only be called on an FP32 tensor.");
        assert(master_tensor_.expired() && "Conversion not supported for child tensors.");
        if (allocted()) {
            Tensor half_tensor(backend());
            auto batch = this->batch();
            auto head = this->head();
            auto sequence = this->sequence();
            auto dimension = this->dimension();
            half_tensor.setDtype(MLLM_TYPE_F16);
            half_tensor.setName(impl_->name_);
            half_tensor.setCtype(impl_->ctype_);
            half_tensor.reshape(batch, head, sequence, dimension);
            half_tensor.alloc();
            backend()->convert_fp_data(this, &half_tensor);
            return half_tensor;
        } else {
            impl_->dtype_ = MLLM_TYPE_F16;
            return *this;
        }
    }
    Tensor fp16() {
        return half();
    }
    Tensor fp32() {
        if (dtype() == MLLM_TYPE_F32) {
            return *this;
        }
        assert(dtype() == MLLM_TYPE_F16 && "Tensor::fp32() can only be called on an FP16 tensor.");
        assert(master_tensor_.expired() && "Conversion not supported for child tensors.");
        if (allocted()) {
            Tensor fp32_tensor(backend());
            auto batch = this->batch();
            auto head = this->head();
            auto sequence = this->sequence();
            auto dimension = this->dimension();
            fp32_tensor.setDtype(MLLM_TYPE_F32);
            fp32_tensor.setName(impl_->name_);
            fp32_tensor.setCtype(impl_->ctype_);
            fp32_tensor.reshape(batch, head, sequence, dimension);
            fp32_tensor.alloc();
            backend()->convert_fp_data(this, &fp32_tensor);
            return fp32_tensor;
        } else {
            impl_->dtype_ = MLLM_TYPE_F16;
            return *this;
        }
    }

    static vector<Tensor> toDevice(vector<Tensor> inputs, BackendType backend_type) {
        for (auto &input : inputs) {
            if (input.device() != backend_type) {
                input.to(backend_type);
            }
        }
        return inputs;
    };
    static vector<Tensor> toCPU(vector<Tensor> inputs) {
        return toDevice(inputs, MLLM_CPU);
    }
    static vector<Tensor> toQNN(vector<Tensor> inputs) {
        return toDevice(inputs, MLLM_QNN);
    }

    static void reshapeAllocCrossBn(Tensor &src_t, Tensor &dst_t);
    static void copyDataCrossBn(Tensor &src_t, Tensor &dst_t);

public:
    uint32_t &uuid();

    TensorType &xnnTensorType();

    bool &allowAggregated() {
        return allow_aggregated_;
    }

    void forceResetHostPointer(void *ptr);

    float i8_scale = 1.f;

    void allocFromTemplate(shared_ptr<Tensor> template_tensor);

private:
    void _allocate_final_tensor(
        const std::shared_ptr<Tensor> &template_tensor,
        Backend *backend);
    void _allocate_aggregated_tensor(
        const std::shared_ptr<Tensor> &template_tensor,
        Module *module,
        Backend *backend);

public:
    /* Functions used for 5-D Tensor:
     * - reshape
     * - channel
     * - time
     * - height
     * - width
     * - offset
     * - dataAt
     * - ptrAt
     * - setDataAt
     */

    /**
     * \brief Reshape 5-D Tensor with five dimensions: [batch, channel, time, height, width].
     *        The five dimensions are designed for Convolutional Neural Networks (CNNs):
     * \param batch Batch size
     * \param channel Number of channels
     * \param time Time dimension (used for 3D convolutions)
     * \param height Height of the 2D grid
     * \param width Width of the 2D grid
     * \return Whether the reshape operation was successful.
     */
    bool reshape(int batch, int channel, int time, int height, int width);
    /**
     * \brief get the size of the corresponding dimension for 5-D Tensor, contains: batch, head, sequence, dimension.
     *        each Tensor has private variable 'ctype_', which indicates the order of the dimensions in the memory.
     *        e.g. ctype_ == BCTHW, the order of the dimensions in the memory is: batch, channel, time,height, width.
     *             ctype_ == BTHWC, the order of the dimensions in the memory is: batch, time, height, width, channel.
     *        so channel() is not equal to shape(1), it depends on the value of ctype_.
     *        no matter what the value of ctype_ is, these functions will return the size of the corresponding dimension.
     * \return the size of the corresponding dimension
     */
    int channel() {
        return impl_->channel();
    }
    int time() {
        return impl_->time();
    }
    int height() {
        return impl_->height();
    }
    int width() {
        return impl_->width();
    }
    int offset(const int b, const int c, const int t, const int h, const int w) {
        assert(impl_->ctype_ == BCTHW || impl_->ctype_ == BTHWC);
        switch (impl_->ctype_) {
        case BCTHW:
            return (((b * channel() + c) * time() + t) * height() + h) * width() + w;
        case BTHWC:
            return (((b * time() + t) * height() + h) * width() + w) * channel() + c;
        default: return -1;
        }
    }
    template <typename Dtype>
    Dtype dataAt(const int batch, const int channel, const int time, const int height, const int width) {
        assert(impl_->ctype_ == BCTHW || impl_->ctype_ == BTHWC);
        return ((Dtype *)impl_->host_ptr_)[offset(batch, channel, time, height, width)];
    }
    template <typename Dtype>
    Dtype *ptrAt(const int batch, const int channel, const int time, const int height, const int width) {
        assert(impl_->ctype_ == BCTHW || impl_->ctype_ == BTHWC);
        return ((Dtype *)impl_->host_ptr_ + offset(batch, channel, time, height, width));
    }
    template <typename Dtype>
    void setDataAt(const int batch, const int channel, const int time, const int height, const int width, Dtype value) {
        assert(impl_->ctype_ == BCTHW || impl_->ctype_ == BTHWC);
        Dtype *typed_ptr = static_cast<Dtype *>(impl_->host_ptr_);
        typed_ptr[offset(batch, channel, time, height, width)] = value;
    }
    Module *module() const {
        return impl_->module_;
    }
    void setModule(Module *module) {
        impl_->module_ = module;
    }
    void transCopyShape(const vector<int> &shape) {
        impl_->private_reshape(shape);
    }

private:
    /**
     * @brief (辅助函数) 处理父子 Tensor 之间的内存布局 (ctype) 同步。
     * 这段逻辑直接从原始的 shallowCopyFrom 中提取，保留了所有边缘情况的处理。
     * @param master_tensor 新的父 Tensor。
     */
    void reconcileLayouts(Tensor *master_tensor) {
        // 情况 1: 通用的4D张量布局同步 (非5D视觉张量)
        // 条件: 父子 ctype 不一致，且允许布局变化从子张量“扩散”到父张量。
        if (impl_->ctype_ != BCTHW && impl_->ctype_ != BTHWC
            && impl_->ctype_ != master_tensor->ctype() && !impl_->undiffusion_) {
            if (impl_->transed_) { // 如果子张量(this)已被转置，则强制父张量跟随子的布局
                auto b = master_tensor->batch();
                auto h = master_tensor->head();
                auto d = master_tensor->dimension();
                auto s = master_tensor->sequence();
                master_tensor->impl_->ctype_ = impl_->ctype_;
                master_tensor->impl_->chls_ = impl_->chls_;
                master_tensor->reshape(b, h, s, d);
            } else { // 否则，子张量跟随父张量的布局
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                impl_->ctype_ = master_tensor->impl_->ctype_;
                impl_->chls_ = master_tensor->impl_->chls_;
                reshape(b, h, s, d);
            }
        }
        // 情况 2 和 情况 3 都需要访问 child_tensors_，所以需要保护
        if (child_tensors_.empty()) {
            return;
        }
        // 情况 2: 处理三层张量结构中的布局冲突 (祖父 -> this -> 孙子)
        if (auto child_sp = child_tensors_[0].lock()) {
            Tensor *child = child_sp.get();

            if (child->ctype() == master_tensor->impl_->ctype_ && ctype() != master_tensor->impl_->ctype_) {
                auto b = child->batch();
                auto h = child->head();
                auto s = child->sequence();
                auto d = child->dimension();

                impl_->chls_ = master_tensor->impl_->chls_;
                child->impl_->chls_ = master_tensor->impl_->chls_;

                for (int i = impl_->trans_from_.size() - 1; i >= 0; --i) {
                    auto tf = impl_->trans_from_[i];
                    std::swap(child->impl_->chls()[tf.first], child->impl_->chls()[tf.second]);
                }
                changeCtype();
                child->changeCtype();
                child->reshape(b, h, s, d);
                transCopyShape(child->shape());
            }
            // 情况 3: 处理从4D (LLM) 到5D (Vision) 张量的特殊布局转换
            else if (child->ctype() == BCTHW && master_tensor->impl_->ctype_ == BSHD && ctype() != BCTHW) {
                auto b = child->batch();
                auto c = child->channel();
                auto t = child->time();
                auto h = child->height();
                auto w = child->width();

                impl_->chls_ = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
                child->impl_->chls_ = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};

                for (int i = impl_->trans_from_.size() - 1; i >= 0; --i) {
                    auto tf = impl_->trans_from_[i];
                    std::swap(child->impl_->chls()[tf.first], child->impl_->chls()[tf.second]);
                }
                changeCtype();
                child->changeCtype();
                child->reshape(b, c, t, h, w);
                transCopyShape(child->shape());
            }
        }
    }

    /**
     * @brief (辅助函数) 为作为 "View" 的子 Tensor 设置形状和偏移信息。
     * @param source 父 Tensor
     * @param shape_offset 维度偏移
     * @param head_rep GQA 的头重复次数
     */
    void setupShapeForView(Tensor *source, const vector<int> &shape_offset, int head_rep) {
        // 记录父张量的原始维度，用于后续计算 offset
        shape_master_ = {(uint64_t)source->batch(), (uint64_t)source->head(),
                         (uint64_t)source->sequence(), (uint64_t)source->dimension()};
        shape_offset_ = {(uint64_t)shape_offset[0], (uint64_t)shape_offset[1],
                         (uint64_t)shape_offset[2], (uint64_t)shape_offset[3]};

        // 如果父子布局转置了 (例如 BSHD vs BHDS), 需要同步调整 shape_master_ 和 shape_offset_ 的记录顺序
        if (!std::equal(source->impl_->chls_.begin(), source->impl_->chls_.end(), impl_->chls_.begin())
            && impl_->chls_[SEQUENCE] == source->impl_->chls_[DIMENSION]
            && source->impl_->chls_[SEQUENCE] == impl_->chls_[DIMENSION]) {
            std::swap(shape_master_[2], shape_master_[3]); // 交换 sequence 和 dimension
            std::swap(shape_offset_[2], shape_offset_[3]);
        }

        // 特殊处理 GQA (Grouped-Query Attention)
        // 当子张量的头数量与父张量不同时，通常意味着 K/V cache 的共享。
        // 我们需要调整 shape_master_ 的 dimension 来反映这一点，确保偏移计算正确。
        if (source->head() != head()) {
            if (head() == 1 && head_rep == 1) { // 可能是 MQA (Multi-Query Attention)
                shape_master_ = {(uint64_t)source->batch(), (uint64_t)head(), (uint64_t)source->sequence(), (uint64_t)source->dimension() * source->head() / head()};
            } else if (head() == 1 && head_rep > 1) { // GQA
                shape_master_ = {(uint64_t)source->batch(), (uint64_t)head(), (uint64_t)source->sequence(), (uint64_t)source->dimension() * source->head() / head_rep};
            }
        }
    }

    /**
     * @brief (辅助函数) 重新指定当前 Tensor 的子 Tensor (孙张量) 的父节点。
     * 将它们从 this 的子节点变为 source 的直接子节点。
     * @param source 新的父(祖父)节点
     * @param shape_offset
     * @param head_rep
     */
    void reparentChildTensors(std::shared_ptr<Tensor> source, const vector<int> &shape_offset, int head_rep) {
        auto it = child_tensors_.begin();
        while (it != child_tensors_.end()) {
            if (auto child_sp = it->lock()) {
                /*
                vector<int> final_offset;
                auto origin_shape_offset = child_sp->shapeOffset();
                if (!origin_shape_offset.empty()) {
                    final_offset = origin_shape_offset;
                } else if (!shape_offset.empty()) {
                    final_offset = shape_offset;
                }
                child_sp->shallowCopyFrom(source, false, final_offset, head_rep);
                */
                // merge qnn:
                vector<int> final_offset;
                auto origin_shape_offset = child_sp->shapeOffset();
                if (!origin_shape_offset.empty()) {
                    if (!shape_offset.empty()) {
                        // 修改 origin_shape_offset 的第三个元素（索引为2）
                        origin_shape_offset[2] = shape_offset[2];
                    }
                    final_offset = origin_shape_offset; // 使用修改后的 origin_shape_offset
                } else if (!shape_offset.empty()) {
                    final_offset = shape_offset;
                } else {
                    final_offset.clear(); // 或者保持默认的空 vector
                }
                child_sp->shallowCopyFrom(source, false, final_offset, head_rep);

                it = child_tensors_.erase(it);
            } else {
                it = child_tensors_.erase(it);
            }
        }
    }

    int checkDim(int &b, int &h, int &s, int &d) {
        if (!aggregated_) {
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
            if (tensor_id > 0)
                h = h - aggregated_dims_[tensor_id - 1];
            break;
        }
        case SEQUENCE: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (s < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            if (tensor_id > 0)
                s = s - aggregated_dims_[tensor_id - 1];
            break;
        }
        case DIMENSION: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (d < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            if (tensor_id > 0) {
                d = d - aggregated_dims_[tensor_id - 1];
            }
            break;
        }
        case D_HD: {
            if (aggregated_tensors_[0]->dimension() == aggregated_tensors_[1]->dimension()) {
                int dim_size = aggregated_tensors_[0]->dimension();
                int aggregated_size = aggregated_tensors_.size();
                h = d / (dim_size * aggregated_size);
                auto d_m = d % (dim_size * aggregated_size);
                tensor_id = d_m / dim_size;
                d = d_m % dim_size;
                // h = h_;
            } else {
                // TODO
                auto orin_d = d;
                int head_size = aggregated_tensors_[0]->head();
                int dim_t = d % (dimension() / head_size);
                int old_dim = 0;
                for (int a = 0; a < aggregated_dims_.size(); ++a) {
                    if (dim_t < aggregated_dims_[a]) {
                        tensor_id = a;
                        break;
                    }
                    old_dim += aggregated_tensors_[a]->dimension();
                }
                // int dim_size = aggregated_tensors_[tensor_id]->dimension();
                h = d * head_size / dimension();
                d = dim_t - old_dim;
                // std::cout<<tensor_id<<" "<<h<<" "<<d<<" , "<<orin_d<<std::endl;
            }
            break;
        }
        case HD: {
            auto orin_d = d;
            if (aggregated_tensors_[0]->dimension() == aggregated_tensors_[1]->dimension()) {
                int dim_size = aggregated_tensors_[0]->dimension();
                int head_size = aggregated_tensors_[0]->head();
                tensor_id = orin_d / (dim_size * head_size);
                h = (orin_d - tensor_id * (dim_size * head_size)) / dim_size;
                d = (orin_d - tensor_id * (dim_size * head_size)) % dim_size;
            } else {
                int head_size = aggregated_tensors_[0]->head();
                int old_dim = 0;
                for (int a = 0; a < aggregated_dims_.size(); ++a) {
                    if (d < aggregated_dims_[a] * head_size) {
                        tensor_id = a;
                        break;
                    }
                    old_dim += aggregated_tensors_[a]->dimension();
                }
                int dim_size = aggregated_tensors_[tensor_id]->dimension();
                h = (orin_d - old_dim * head_size) / dim_size;
                d = (orin_d - old_dim * head_size) % dim_size;
                // std::cout<<tensor_id<<" "<<h<<" "<<d<<" , "<<orin_d<<std::endl;
            }
            break;
        }
        case D_DH: {
            auto orin_d = d;
            int dim_size = aggregated_tensors_[0]->dimension();
            int total_head_idx = d / dim_size;
            d = d % dim_size;
            int old_head_idx = 0;
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                old_head_idx += aggregated_dims_[a];
                if (total_head_idx < old_head_idx) {
                    tensor_id = a;
                    old_head_idx -= aggregated_dims_[a];
                    break;
                }
            }
            h = total_head_idx - old_head_idx;
            break;
        }
        default:
            break;
        }
        return tensor_id;
    }

    // in_place=true: 只有输入, 输出==输入，返回输入
    static std::vector<Tensor> runFunc(std::vector<std::string> out_names,
                                       OpType type,
                                       OpParam param,
                                       std::vector<Tensor> input_tensors = {},
                                       bool in_place = false);

public:
    /* Functions used for TEST & DEBUG
     * - checkData
     * - printShape
     * - printData
     * - saveData
     * - printMem
     * - printAVG
     * - printCtype
     */

    template <typename Dtype>
    void checkData() {
        if (ctype() == BTHWC || ctype() == BCTHW || dtype() != MLLM_TYPE_F32) {
            return;
        }
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
        if (ck) {
            std::cout << "\n[ERROR]:" << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] has Nan" << std::endl;
            // printData<Dtype>();
            assert(ck == false);
        }
    }

    void printShape() {
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
    }

    template <typename Dtype>
    void printDataTorchLike() {
        if (ctype() == BTHWC || ctype() == BCTHW) {
            printData<Dtype>();
            return;
        }

        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;

        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();

        // 每行最多打印的元素数量
        const int max_elements_per_line = 6;
        // 最多打印的行数
        const int max_rows = 6;

        // 递归打印函数
        auto printRecursive = [&](auto &&self, int n, int c, int h, int w, int depth) -> void {
            if (depth == 0) {
                std::cout << "[";
            }

            if (depth == 3) {
                // 打印单个元素
                std::cout << std::fixed << std::setprecision(7) << std::setw(10) << static_cast<float>(dataAt<Dtype>(n, c, h, w)) << " ";
                if (w == W - 1) {
                    std::cout << "]";
                }
                return;
            }

            if (depth == 2) {
                std::cout << "[";
                if (W > max_elements_per_line) {
                    int half = max_elements_per_line / 2;
                    for (int w_idx = 0; w_idx < half; ++w_idx) {
                        self(self, n, c, h, w_idx, depth + 1);
                    }
                    std::cout << "... ";
                    for (int w_idx = W - half; w_idx < W; ++w_idx) {
                        self(self, n, c, h, w_idx, depth + 1);
                    }
                } else {
                    for (int w_idx = 0; w_idx < W; ++w_idx) {
                        self(self, n, c, h, w_idx, depth + 1);
                    }
                }
                std::cout << "]";
                return;
            }

            if (depth == 1) {
                std::cout << "[";
                if (H > max_rows) {
                    int half = max_rows / 2;
                    for (int h_idx = 0; h_idx < half; ++h_idx) {
                        self(self, n, c, h_idx, 0, depth + 1);
                        std::cout << std::endl
                                  << " ";
                    }
                    std::cout << "..." << std::endl
                              << " ";
                    for (int h_idx = H - half; h_idx < H; ++h_idx) {
                        self(self, n, c, h_idx, 0, depth + 1);
                        if (h_idx < H - 1) {
                            std::cout << std::endl
                                      << " ";
                        }
                    }
                } else {
                    for (int h_idx = 0; h_idx < H; ++h_idx) {
                        self(self, n, c, h_idx, 0, depth + 1);
                        if (h_idx < H - 1) {
                            std::cout << std::endl
                                      << " ";
                        }
                    }
                }
                std::cout << "]";
                return;
            }

            if (depth == 0) {
                for (int c_idx = 0; c_idx < C; ++c_idx) {
                    self(self, n, c_idx, 0, 0, depth + 1);
                    if (c_idx < C - 1) {
                        std::cout << std::endl
                                  << std::endl;
                    }
                }
                std::cout << "]";
                return;
            }
        };

        // 打印整个 Tensor
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            printRecursive(printRecursive, n_idx, 0, 0, 0, 0);
            if (n_idx < N - 1) {
                std::cout << std::endl
                          << std::endl;
            }
        }
    }

    template <typename Dtype>
    void printData() {
        assert(backend()->type() == MLLM_CPU && "printData only support CPU backend.");
        if (ctype() == BTHWC || ctype() == BCTHW) {
            printData<Dtype>();
            return;
        }
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << static_cast<float>(dataAt<Dtype>(0, 0, h, c)) << " ";
                }
                std::cout << std::endl;
                std::cout << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << static_cast<float>(dataAt<Dtype>(0, c, h, 0)) << " ";
                }
                std::cout << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            std::cout << std::fixed << std::setprecision(7) << static_cast<float>(dataAt<Dtype>(n, c, h, w)) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    template <typename Dtype>
    void saveData(string ex = "", string directory = "save_out") {
        if (batch() == 0) {
            return;
        }
        if (ctype() == BTHWC || ctype() == BCTHW) {
            save5Data<Dtype>(ex);
            return;
        }
        // std::filesystem::create_directory("save_out");
        struct stat info;
#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        if (stat(directory.c_str(), &info) != 0) {
            if (stat(directory.c_str(), &info) != 0) {
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            } else if (!(info.st_mode & S_IFDIR)) {
                // if the path exists but it is not a directory, also create it
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            }
        }
#endif
        std::ofstream outFile(directory + "/" + name() + ex + ".log");
        outFile << "----------------------------------------" << std::endl;
        if (impl_->ctype_ == BSHD) {
            outFile << name() << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        } else {
            outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        }

        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();

        if (impl_->ctype_ == BHSD) {
            for (int n = 0; n < batch(); ++n) {
                for (int c = 0; c < head(); ++c) {
                    for (int h = 0; h < sequence(); ++h) {
                        for (int w = 0; w < dimension(); ++w) {
                            outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
            outFile.close();
            return;
        }
        if (impl_->ctype_ == BSHD) {
            for (int n = 0; n < batch(); ++n) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int c = 0; c < head(); ++c) {
                        for (int w = 0; w < dimension(); ++w) {
                            outFile << std::fixed << std::setprecision(6) << static_cast<float>(dataAt<Dtype>(n, c, h, w)) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
            outFile.close();
            return;
        }
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, 0, h, c) << " ";
                }
                outFile << std::endl;
                outFile << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, c, h, 0) << " ";
                }
                outFile << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    for (int c = 0; c < C; ++c) {
                        for (int w = 0; w < W; ++w) {
                            outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
        }

        outFile.close();
    }

    void saveQ4Data_d(string ex = "", string directory = "save_out") {
        if (batch() == 0) {
            return;
        }
        struct stat info;
#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        if (stat(directory.c_str(), &info) != 0) {
            if (stat(directory.c_str(), &info) != 0) {
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            } else if (!(info.st_mode & S_IFDIR)) {
                // if the path exists but it is not a directory, also create it
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            }
        }
#endif
        std::ofstream outFile(directory + "/" + name() + ex + ".log");
        outFile << "----------------------------------------" << std::endl;
        if (impl_->ctype_ == BSHD) {
            outFile << name() << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << DataTypeName(dtype()) << " " << ctype() << std::endl;
        } else {
            outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << DataTypeName(dtype()) << " " << ctype() << std::endl;
        }

        if (impl_->dtype_ != MLLM_TYPE_Q4_0) {
            outFile << "Error: Tensor is not of type MLLM_TYPE_Q4_0." << std::endl;
            outFile.close();
            return;
        }

        block_q4_0 *data_ptr = hostPtr<block_q4_0>();
        if (data_ptr == nullptr) {
            outFile << "Error: Host pointer is null." << std::endl;
            outFile.close();
            return;
        }

        const int W_blocks = dimension() / QK4_0;

        // 保持原始的循环结构。注意变量名: h 代表 sequence, c 代表 head。
        for (int n = 0; n < batch(); ++n) {
            for (int h = 0; h < sequence(); ++h) {
                for (int c = 0; c < head(); ++c) {
                    for (int w = 0; w < W_blocks; ++w) {
                        uint64_t block_offset = offset(n, c, h, w) / QK4_0;
                        block_q4_0 &data_block = data_ptr[block_offset];
                        float da = MLLM_FP16_TO_FP32(data_block.d);
                        outFile << std::fixed << std::setprecision(6) << da << " ";
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
            outFile << std::endl;
        }
        outFile.close();
    }
    template <typename Dtype>
    void saveIntData(string ex = "") {
        if (Tensor::tensor_status != TENSOR_STATIC_READY) return;
        if (ctype() == BTHWC || ctype() == BCTHW) {
            save5Data<Dtype>(ex);
            return;
        }
        // std::filesystem::create_directory("save_out");
        string directory = "save_out";
        struct stat info;
#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        if (stat(directory.c_str(), &info) != 0) {
            if (stat(directory.c_str(), &info) != 0) {
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            } else if (!(info.st_mode & S_IFDIR)) {
                // if the path exists but it is not a directory, also create it
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            }
        }
#endif
        std::ofstream outFile(directory + "/" + name() + ex + ".log");
        outFile << "----------------------------------------" << std::endl;
        if (impl_->ctype_ == BSHD) {
            outFile << name() << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        } else {
            outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        }

        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (impl_->ctype_ == BSHD) {
            for (int n = 0; n < batch(); ++n) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int c = 0; c < head(); ++c) {
                        for (int w = 0; w < dimension(); ++w) {
                            outFile << (int)dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
            outFile.close();
            return;
        }
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, 0, h, c) << " ";
                }
                outFile << std::endl;
                outFile << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, c, h, 0) << " ";
                }
                outFile << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    for (int c = 0; c < C; ++c) {
                        for (int w = 0; w < W; ++w) {
                            outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
        }

        outFile.close();
    }

    template <typename Dtype>
    void saveNData(string new_name = "", string ex = "") {
        if (Tensor::tensor_status == TENSOR_STATIC_READY && !shape().empty()) {
            if (ctype() == BTHWC || ctype() == BCTHW) {
                save5Data<Dtype>(ex);
                return;
            }
            // std::filesystem::create_directory("save_out");
            string directory = "save_out";
            struct stat info;

#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            if (stat(directory.c_str(), &info) != 0) {
                if (stat(directory.c_str(), &info) != 0) {
                    mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
                } else if (!(info.st_mode & S_IFDIR)) {
                    // if the path exists but it is not a directory, also create it
                    mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
                }
            }
#endif
            auto tmp_name = name();
            if (new_name.empty()) {
            } else {
                tmp_name = new_name;
            }
            std::ofstream outFile(directory + "/" + tmp_name + ex + ".log");

            outFile << "----------------------------------------" << std::endl;
            if (new_name.empty()) {
                outFile << name();
            } else {
                outFile << new_name;
            }
            if (impl_->ctype_ == BSHD) {
                outFile << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
                for (int n = 0; n < batch(); ++n) {
                    for (int h = 0; h < sequence(); ++h) {
                        for (int c = 0; c < head(); ++c) {
                            for (int w = 0; w < dimension(); ++w) {
                                outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                            }
                            outFile << std::endl;
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile.close();
                return;
            }
            outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
            int N = batch();
            int C = head();
            int H = sequence();
            int W = dimension();
            if (N == 1 && C == 1) {
                for (int h = 0; h < H; ++h) {
                    for (int c = 0; c < W; ++c) {
                        outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, 0, h, c) << " ";
                    }
                    outFile << std::endl;
                    outFile << "---------" << std::endl;
                }
            } else if (N == 1 && W == 1) {
                for (int h = 0; h < H; ++h) {
                    for (int c = 0; c < C; ++c) {
                        outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, c, h, 0) << " ";
                    }
                    outFile << std::endl;
                }
            } else {
                for (int n = 0; n < N; ++n) {
                    for (int h = 0; h < H; ++h) {
                        for (int c = 0; c < C; ++c) {
                            for (int w = 0; w < W; ++w) {
                                outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                            }
                            outFile << std::endl;
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
            }

            outFile.close();
        }
    }

    template <typename Dtype>
    void print5Data() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << channel() << " " << time() << " " << height() << " " << width() << "]" << std::endl;
        int N = batch();
        int C = channel();
        int T = time();
        int H = height();
        int W = height();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < T; ++t) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, t, h, w) << " ";
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
    void save5Data(string ex = "") {
        // std::filesystem::create_directory("save_out");
        string directory = "save_out";
        struct stat info;

#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        if (stat(directory.c_str(), &info) != 0) {
            if (stat(directory.c_str(), &info) != 0) {
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            } else if (!(info.st_mode & S_IFDIR)) {
                // if the path exists but it is not a directory, also create it
                mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
            }
        }
#endif
        std::ofstream outFile(directory + "/" + name() + ex + ".log");
        outFile << "----------------------------------------" << std::endl;
        outFile << name() << ": shape:[" << batch() << " " << channel() << " " << time() << " " << height() << " " << width() << "]" << std::endl;
        int N = batch();
        int C = channel();
        int T = time();
        int H = height();
        int W = height();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < T; ++t) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            outFile << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, t, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
        }
    }

    template <typename Dtype>
    void printMem() {
        for (int i = 0; i < impl_->count_; ++i) {
            auto *typed_ptr = static_cast<Dtype *>(impl_->host_ptr_);
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
    }

    void printCtype() {
        std::string ctype;
        switch (impl_->ctype_) {
        case BHSD:
            ctype = "BHSD";
            break;
        case BSHD:
            ctype = "BSHD";
            break;
        case BHDS:
            ctype = "BHDS";
            break;
        case BCTHW:
            ctype = "BCTHW";
            break;
        case BTHWC:
            ctype = "BTHWC";
            break;
        case BWCTH:
            ctype = "BWCTH";
            break;
        case SBHD:
            ctype = "SBHD";
            break;
        case BDHS:
            ctype = "BDHS";
            break;
        case BDSH:
            ctype = "BDSH";
            break;
        case DBHS:
            ctype = "DBHS";
            break;
        }
        std::cout << name() << ": ctype:[" << ctype << "]" << std::endl;
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

    void fullDataVector(vector<int> values) {
        reshape(1, 1, values.size(), 1);
        alloc();
        for (int n = 0; n < batch(); ++n) {
            for (int c = 0; c < head(); ++c) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int w = 0; w < dimension(); ++w) {
                        setDataAt<float>(n, c, h, w, values[h]);
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
};
} // namespace mllm
#endif // MLLM_TENSOR_H
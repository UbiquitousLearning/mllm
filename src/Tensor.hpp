#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H
#include <climits>
#include "Backend.hpp"
#include "OpDefined.hpp"
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
class Tensor {
public:
    Tensor() :
        host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    Tensor(Backend *bn) :
        backend_(bn), host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    /*
    ~Tensor() {
        free();
    }
    */

    static TensorStatus tensor_status;

private:
    std::map<Chl, int> chls_ = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
    string name_;
    DataType dtype_;
    ChlType ctype_ = BSHD;
    TensorType ttype_ = NORMAL_TENSOR;

    Backend *backend_{};
    void *host_ptr_{};

    vector<uint64_t> shape_;
    uint64_t capacity_ = 0;
    uint64_t count_ = 0;
    uint64_t allocated_ = 0;

    bool transed_ = false;
    bool should_in_graphs_ = true;

    // used for ChildTensor
    vector<uint64_t> shape_offset_;
    vector<uint64_t> shape_master_;
    Tensor *master_tensor_ = nullptr;
    vector<Tensor *> child_tensors_;
    bool undiffusion_ = false;
    vector<std::pair<Chl, Chl>> trans_from_;

    //  used for AggregatedTensor
    bool aggregated_ = false;
    vector<shared_ptr<Tensor>> aggregated_tensors_;
    Tensor *deaggregated_tensor_ = nullptr;
    Chl aggregated_dim_;
    vector<int> aggregated_dims_;
    Module *module_{};

private:
    // XNN_INVALID_VALUE_ID = 4294967295U
    uint32_t uuid_ = 4294967295U;
    TensorType xnn_tensor_type_ = TensorType::NORMAL_TENSOR;

public:
    /**
     * \brief build 4-D Tensor with four dimensions: [batch, head, sequence, dimension].
     *        The four dimension designed for Transformer-based LLMs：
     * \param batch  batch size
     * \param head   multi-head number
     * \param sequence  tokens numbers in a sequence
     * \param dimension the hidden size
     */
    explicit Tensor(int batch, int head, int sequence, int dimension);
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
        dtype_ = dtype;
        alloc();
    }
    void alloc();
    void alloc(vector<unsigned int> alloc_size);

    /**
     * \brief free the memory of Tensor.
     */
    void free() {
        if (aggregated_) { return; }
        if (host_ptr_ != nullptr && masterTensor() == nullptr) {
            // std::cout << "dealloc " << name_ << std::endl;
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
            allocated_ = 0;
        }
    }

    /**
     * \brief  get the number of bytes occupied by Tensor's data in memory.
     *         depends on the total dimension sizes and data type.
     * \return the number of bytes occupied by Tensor's data in memory
     */
    size_t size() const {
        return capacity_ * dtypeSize();
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
        return chls_;
    }

    int batch() {
        return legacyShape(chls()[BATCH]);
    }
    int head() {
        return legacyShape(chls()[HEAD]);
    }
    int sequence() {
        return legacyShape(chls()[SEQUENCE]);
    }
    int dimension() {
        return legacyShape(chls()[DIMENSION]);
    }

    /**
     * \brief get the totol size of all dimensions.
     *        mostly, count() == batch() * head() * sequence() * dimension()
     * \return the totol size of all dimensions
     */
    int count() const {
        return count_;
    }
    int numAxes() const {
        return shape_.size();
    }
    string shapeString() const {
        std::ostringstream stream;
        for (int i : shape_) {
            stream << i << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
    }
    int canonicalAxisIndex(int axis_index) const {
        if (axis_index < 0) {
            return axis_index + numAxes();
        }
        return axis_index;
    }
    int legacyShape(int index) const {
        if (index >= numAxes() || index < -numAxes()) {
            return 0;
        }
        return shape(index);
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
            switch (ctype_) {
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
            switch (ctype_) {
            case BSHD:
                return ((b * shape_[1] + s) * shape_[2] + h) * shape_[3] + d;
            case BHDS:
                return ((b * shape_[1] + h) * shape_[2] + d) * shape_[3] + s;
            case BDHS:
                return ((b * shape_[1] + d) * shape_[2] + h) * shape_[3] + s;
            case SBHD:
                return ((s * shape_[1] + b) * shape_[2] + h) * shape_[3] + d;
            case DBHS:
                return ((d * shape_[1] + b) * shape_[2] + h) * shape_[3] + s;
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
            for (int i = 0; i < numAxes(); ++i) {
                offset *= shape(i);
                if (indices.size() > i) {
                    offset += indices[i];
                }
            }
            return offset;
        }
    }

    int memshape(int index) const {
        if (master_tensor_ != NULL) {
            if (master_tensor_->master_tensor_ != NULL) {
                return master_tensor_->master_tensor_->shape_[index];
            } else {
                return master_tensor_->shape_[index];
            }
        } else {
            return shape(index);
        }
    }
    int sequenceSkipDim() const {
        if (master_tensor_ != NULL) {
            if (master_tensor_->master_tensor_ != NULL) {
                auto shape = master_tensor_->master_tensor_->shape_;
                if (master_tensor_->master_tensor_->ctype_ == BSHD) {
                    return shape[3] * shape[2];
                } else if (master_tensor_->master_tensor_->ctype_ == BHDS) {
                    return shape[3];
                } else if (master_tensor_->master_tensor_->ctype_ == BDHS) {
                    return shape[3] * shape_[2];
                } else {
                    std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                    return -1;
                }
            } else {
                auto shape = master_tensor_->shape_;
                if (master_tensor_->ctype_ == BSHD) {
                    return shape[3] * shape[2];
                } else if (master_tensor_->ctype_ == BHDS) {
                    return shape[3];
                } else if (master_tensor_->ctype_ == BDHS) {
                    return shape[3] * shape_[2];
                } else {
                    std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                    return -1;
                }
            }
        } else {
            if (ctype_ == BSHD) {
                return shape_[3] * shape_[2];
            } else if (ctype_ == BHDS) {
                return shape_[3];
            } else if (ctype_ == BDHS) {
                return shape_[3] * shape_[2];
            } else if (ctype_ == DBHS) {
                return shape_[3] * shape_[2];
            } else if (ctype_ == SBHD) {
                return shape_[3] * shape_[2];
            } else {
                std::cout << "sequenceSkipDim() only support for BSHD and BHDS" << std::endl;
                return -1;
            }
            // return shape_[3]*shape_[2];
        }
    }

    /**
     * \brief obtain the raw pointer to the first address where tensor stores data.
     * \return the pointer(void *) to the first address where tensor stores data.
     */
    void *rawHostPtr() const {
        return host_ptr_;
    }

    /**
     * \brief obtain the pointer to the first address where tensor stores data.
     * \tparam Dtype float, mllm_fp16_t, etc.
     * \return the pointer to the first address where tensor stores data.
     */
    template <typename Dtype>
    Dtype *hostPtr() const {
        return (Dtype *)host_ptr_;
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
            return ((Dtype *)host_ptr_)[offset(batch, head, sequence, dimension)];
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
            return ((Dtype *)host_ptr_)[offset(batch, head, sequence, dimension)];
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
            return ((Dtype *)host_ptr_ + offset(batch, head, sequence, dimension));
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
            return ((Dtype *)host_ptr_ + offset(batch, head, sequence, dimension));
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
            Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
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
            return dtype_;
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->dtype_;
        }
    }

    Backend *backend() const {
        return backend_;
    }
    void setBackend(Backend *bn) {
        backend_ = bn;
    };

    DataType dtype() const {
        return dtype_;
    }
    void setDtype(DataType dtype) {
        dtype_ = dtype;
    }

    TensorType ttype() const {
        return ttype_;
    }
    void setTtype(TensorType ttype) {
        ttype_ = ttype;
    }

    std::vector<int> shape() const {
        std::vector<int> shape_int(shape_.size());
        std::transform(shape_.begin(), shape_.end(), shape_int.begin(), [](uint64_t val) {
            return static_cast<int>(val);
        });
        return shape_int;
    }

    ChlType ctype() const {
        return ctype_;
    }
    void setCtype(ChlType type) {
        ctype_ = type;
        switch (ctype_) {
        case BSHD:
            chls()[BATCH] = 0;
            chls()[SEQUENCE] = 1;
            chls()[HEAD] = 2;
            chls()[DIMENSION] = 3;
            break;
        case BHDS:
            chls()[BATCH] = 0;
            chls()[HEAD] = 1;
            chls()[DIMENSION] = 2;
            chls()[SEQUENCE] = 3;
            break;
        case SBHD:
            chls()[SEQUENCE] = 0;
            chls()[BATCH] = 1;
            chls()[HEAD] = 2;
            chls()[DIMENSION] = 3;
            break;
        case BTHWC:
            chls()[BATCH] = 0;
            chls()[TIME] = 1;
            chls()[HEIGHT] = 2;
            chls()[WIDTH] = 3;
            chls()[CHANNLE] = 3;
            break;
        case BCTHW:
            chls()[BATCH] = 0;
            chls()[CHANNLE] = 1;
            chls()[TIME] = 2;
            chls()[HEIGHT] = 3;
            chls()[WIDTH] = 3;
            break;
        default:
            break;
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
            ctype_ = BHDS;
            auto ori_seq_idx = chls()[SEQUENCE];
            auto ori_head_idx = chls()[HEAD];
            auto ori_dim_idx = chls()[DIMENSION];
            chls()[HEAD] = ori_seq_idx;
            chls()[DIMENSION] = ori_head_idx;
            chls()[SEQUENCE] = ori_dim_idx;
            reshape(b, h, s, d);
            transed_ = true;
            undiffusion_ = undiffusion;
        } else if (dim_a == SEQUENCE && dim_b == DIMENSION && ctype() == BHDS) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            ctype_ = BSHD;
            auto ori_seq_idx = chls()[SEQUENCE];
            auto ori_head_idx = chls()[HEAD];
            auto ori_dim_idx = chls()[DIMENSION];
            chls()[SEQUENCE] = ori_head_idx;
            chls()[HEAD] = ori_dim_idx;
            chls()[DIMENSION] = ori_seq_idx;
            reshape(b, h, s, d);
            transed_ = false;
            undiffusion_ = undiffusion;
        } else if (THW == dim_a && dim_b == CHANNLE && ctype() == BCTHW) {
            auto b = batch();
            auto c = channel();
            auto t = time();
            auto h = height();
            auto w = width();
            ctype_ = BTHWC;
            auto ori_chl_idx = chls()[CHANNLE];
            auto ori_time_idx = chls()[TIME];
            auto ori_height_idx = chls()[HEIGHT];
            auto ori_width_idx = chls()[WIDTH];
            chls()[TIME] = ori_chl_idx;
            chls()[HEIGHT] = ori_time_idx;
            chls()[WIDTH] = ori_height_idx;
            chls()[CHANNLE] = ori_width_idx;
            reshape(b, c, t, h, w);
            transed_ = true;
            undiffusion_ = undiffusion;
        } else if (dim_a == BATCH && dim_b == SEQUENCE && ctype() == BSHD) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            ctype_ = SBHD;
            auto ori_batch_idx = chls()[BATCH];
            auto ori_seq_idx = chls()[SEQUENCE];
            chls()[SEQUENCE] = ori_batch_idx;
            chls()[BATCH] = ori_seq_idx;
            reshape(b, h, s, d);
            transed_ = true;
            undiffusion_ = undiffusion;
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
        memcpy(host_ptr_, source.host_ptr_, cntSize());
    }
    void initFrom(const Tensor &source) {
        dtype_ = source.dtype();
        chls_ = source.chls_;
        ctype_ = source.ctype_;
        shape_ = source.shape_;
        count_ = source.count_;
        if (source.host_ptr_ != nullptr) {
            alloc();
        }
    }
    void copyFrom(const shared_ptr<Tensor> &source) {
        assert(masterTensor() == nullptr);
        assert(source->dtype() == dtype());
        assert(source->count() == count());
        memcpy(host_ptr_, source->host_ptr_, cntSize());
    }

    void changeCtype(int size = 0) {
        if (!shape().empty()) {
            size = shape().size();
        }
        if (size == 4) {
            vector<int> a = {chls()[BATCH], chls()[HEAD], chls()[SEQUENCE], chls()[DIMENSION]};
            ctype_ = Chls2Type[a];
        } else {
            vector<int> a = {chls()[BATCH], chls()[TIME], chls()[HEIGHT], chls()[WIDTH], chls()[CHANNLE]};
            ctype_ = Chls2Type[a];
        }
    }

    bool &transed() {
        return transed_;
    }
    bool &undiffusion() {
        return undiffusion_;
    }
    void setUndiffusion(bool undiffusion) {
        undiffusion_ = undiffusion;
        for (auto &child_tensor : child_tensors_) {
            child_tensor->undiffusion_ = undiffusion;
        }
    }

    vector<std::pair<Chl, Chl>> &transFrom() {
        return trans_from_;
    }

    bool &shouldInGraphs() {
        return should_in_graphs_;
    }

    static Tensor zeros(int batch, int head, int sequence, int dimension, BackendType bn_type = MLLM_CPU) {
        Tensor tensor1(batch, head, sequence, dimension, bn_type, true);
        memset(tensor1.hostPtr<float>(), 0, tensor1.count() * sizeof(float));
        tensor1.shouldInGraphs() = false;
        return tensor1;
    }
    static Tensor ones(int batch, int head, int sequence, int dimension, BackendType bn_type = MLLM_CPU) {
        Tensor tensor1(batch, head, sequence, dimension, bn_type, true);
        memset(tensor1.hostPtr<float>(), 1, tensor1.count() * sizeof(float));
        tensor1.shouldInGraphs() = false;
        return tensor1;
    }

    /**
     * \brief Overload the operators.
     * \param data binary data
     * \return Tensor
     */
    Tensor &operator+(float data);
    Tensor &operator-(float data);
    Tensor &operator*(float data);
    Tensor &operator/(float data);
    Tensor &operator/(double data);

    Tensor &operator/(int data);

    /**
     * \brief Overload the operators.
     * \param other The Other Tensor
     * \return Tensor
     */
    Tensor &operator+(Tensor &other);
    Tensor &operator-(Tensor &other);
    Tensor &operator*(Tensor &other);
    Tensor &operator/(Tensor &other);

    Tensor &mean(Chl axis);

    Tensor &view(int b, int h, int s, int d);
    Tensor &flatten(Chl axis_start, Chl axis_end);
    Tensor &transpose(Chl axis0, Chl axis1) {
        return transpose({{axis0, axis1}});
    }
    Tensor &transpose(vector<std::pair<Chl, Chl>> axiss);
    Tensor &clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d);
    Tensor &clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d);
    Tensor &clip(Tensor &index, Chl dim);
    Tensor &expand(int b, int h, int s, int d);
    static Tensor &cat(vector<Tensor> input_tensors, Chl dims);
    static Tensor &mm(Tensor &input0, Tensor &input1);
    Tensor &norm(int L_n);
    Tensor &where(float value, Chl axis);
    static Tensor &range(int start, int end);
    static vector<std::reference_wrapper<Tensor>> split(Tensor &input, std::vector<int> each_dims, Chl split_dim, int same_dim_size = -1);
    vector<std::reference_wrapper<Tensor>> split(std::vector<int> each_dims, Chl split_dim, int same_dim_size = -1) {
        return split(*this, each_dims, split_dim, same_dim_size);
    }
    Tensor &index_put(Tensor &value, Tensor &indices, bool accumulate);
    void scatter_reduce(Tensor &value, Tensor &indices);
    static vector<std::reference_wrapper<Tensor>> topk(Tensor &input, int k, Chl dim);
    Tensor &sum(Chl dim);
    Tensor &argsort();
    Tensor &bincount();
    Tensor &repeat(Chl dim, int dim_size);
    static Tensor &zero_like(Tensor &input);
    static Tensor &apply_rotary_pos_emb_vision(Tensor &input, Tensor&rotary_pos_emb);

    // models use only
    static Tensor &fuyu_gather_embd(Tensor &word, Tensor &image_patches, Tensor &image_patches_indices);
    static Tensor &phi3v_hd_merge(Tensor &input, int h_crop, int w_crop);

    /* Functions used for ChildTensor:
     * - shallowCopyFrom
     * - shape_offset
     * - shape_master
     * - masterTensor
     * - setMasterTensor
     * - childTensors
     * - addChildTensor
     */

    /**
     * \brief this Tensor is a DEEPCOPY of source, only used for ChildTensor.
     * \param source MasterTensor.
     * \param shape_offset the offset of each dimension of ChildTensor compared to MasterTensor.
     * \param head_rep the repeat number of heads of ChildTensor compared to MasterTensor.
     *                 used for repeat the head of K/V in Transformer-based LLMs. Default is 1.
     */
    void shallowCopyFrom(Tensor *source, bool copyshape = true, const vector<int> &shape_offset = {}, int head_rep = 1) {
        if (!shape_offset.empty()) {
            copyshape = false;
        }
        setMasterTensor(source);
        if (ctype_ != BCTHW && ctype_ != BTHWC && ctype_ != master_tensor_->ctype() && undiffusion_ == false) {
            if (transed_) { // child tensor have been transed(BSHD->BHDS);
                auto b = master_tensor_->batch();
                auto h = master_tensor_->head();
                auto d = master_tensor_->dimension();
                auto s = master_tensor_->sequence();
                master_tensor_->ctype_ = ctype_;
                master_tensor_->chls_ = chls_;
                master_tensor_->reshape(b, h, s, d);
            } else {
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                ctype_ = master_tensor_->ctype_;
                chls_ = master_tensor_->chls_;
                reshape(b, h, s, d);
            }
        } else if (child_tensors_.size() == 1 && child_tensors_[0]->ctype() == master_tensor_->ctype_ && ctype() != master_tensor_->ctype_) {
            auto b = child_tensors_[0]->batch();
            auto h = child_tensors_[0]->head();
            auto s = child_tensors_[0]->sequence();
            auto d = child_tensors_[0]->dimension();
            auto origin_c_0 = child_tensors_[0]->chls_;
            auto origin_c_1 = chls_;
            chls_ = master_tensor_->chls_;
            child_tensors_[0]->chls_ = master_tensor_->chls_;
            for (int i = trans_from_.size() - 1; i >= 0; --i) {
                auto tf = trans_from_[i];
                auto axis0 = tf.first;
                auto axis1 = tf.second;
                auto ori_0_idx = child_tensors_[0]->chls()[axis0];
                auto ori_1_idx = child_tensors_[0]->chls()[axis1];
                child_tensors_[0]->chls()[axis0] = ori_1_idx;
                child_tensors_[0]->chls()[axis1] = ori_0_idx;
            }
            changeCtype();
            child_tensors_[0]->changeCtype();
            child_tensors_[0]->reshape(b, h, s, d);
            transCopyShape(child_tensors_[0]->shape());
        } else if (child_tensors_.size() == 1 && child_tensors_[0]->ctype() == BCTHW && master_tensor_->ctype_ == BSHD && ctype() != BCTHW) {
            auto b = child_tensors_[0]->batch();
            auto c = child_tensors_[0]->channel();
            auto t = child_tensors_[0]->time();
            auto h = child_tensors_[0]->height();
            auto w = child_tensors_[0]->width();
            auto origin_c_0 = child_tensors_[0]->chls_;
            auto origin_c_1 = chls_;

            chls_ = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            child_tensors_[0]->chls_ = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            for (int i = trans_from_.size() - 1; i >= 0; --i) {
                auto tf = trans_from_[i];
                auto axis0 = tf.first;
                auto axis1 = tf.second;
                auto ori_0_idx = child_tensors_[0]->chls()[axis0];
                auto ori_1_idx = child_tensors_[0]->chls()[axis1];
                child_tensors_[0]->chls()[axis0] = ori_1_idx;
                child_tensors_[0]->chls()[axis1] = ori_0_idx;
            }
            changeCtype();
            child_tensors_[0]->changeCtype();
            child_tensors_[0]->reshape(b, c, t, h, w);
            transCopyShape(child_tensors_[0]->shape());
        }
        host_ptr_ = source->hostPtr<void>();
        capacity_ = source->capacity_;
        count_ = source->count_;
        if (copyshape) {
            shape_ = source->shape_;
        }
        allocated_ = source->allocated_;
        dtype_ = source->dtype_;
        if (!shape_offset.empty()) {
            shape_master_ = {(uint64_t)source->batch(),
                             (uint64_t)source->head(),
                             (uint64_t)source->sequence(),
                             (uint64_t)source->dimension()};
            shape_offset_ = {(uint64_t)shape_offset[0],
                             (uint64_t)shape_offset[1],
                             (uint64_t)shape_offset[2],
                             (uint64_t)shape_offset[3]};
            if (!std::equal(source->chls_.begin(), source->chls_.end(), chls_.begin()) && chls()[SEQUENCE] == source->chls()[DIMENSION] && source->chls()[SEQUENCE] == chls()[DIMENSION]) {
                shape_master_ = {(uint64_t)source->batch(),
                                 (uint64_t)source->head(),
                                 (uint64_t)source->dimension(),
                                 (uint64_t)source->sequence()};
                shape_offset_ = {(uint64_t)shape_offset[0],
                                 (uint64_t)shape_offset[1],
                                 (uint64_t)shape_offset[3],
                                 (uint64_t)shape_offset[2]};
            }
            if (source->head() != head()) { // TODO: need to check
                if (head() == 1 && head_rep == 1) {
                    shape_master_ = {(uint64_t)source->batch(),
                                     (uint64_t)head(),
                                     (uint64_t)source->sequence(),
                                     (uint64_t)source->dimension() * source->head() / head()};
                } else if (head() == 1 && head_rep > 1) {
                    shape_master_ = {(uint64_t)source->batch(),
                                     (uint64_t)head(),
                                     (uint64_t)source->sequence(),
                                     (uint64_t)source->dimension() * source->head() / head_rep};
                }
            }
        }
        auto it = child_tensors_.begin();
        while (it != child_tensors_.end()) {
            auto &child_tensor = *it;
            auto origin_shape_offset = child_tensor->shapeOffset();
            if (!origin_shape_offset.empty()) {
                if (!shape_offset.empty()) {
                    origin_shape_offset[2] = shape_offset[2];
                }
                child_tensor->shallowCopyFrom(source, false, origin_shape_offset, head_rep);
            } else if (!shape_offset.empty()) {
                child_tensor->shallowCopyFrom(source, false, shape_offset, head_rep);
            } else {
                child_tensor->shallowCopyFrom(source, false, {}, head_rep);
            }
            it = child_tensors_.erase(it);
        }
        source->addChildTensor(this);
    }
    void shallowCopyFrom(Tensor &source, bool copyshape = true, const vector<int> &shape_offset = {}, int head_rep = 1) {
        shallowCopyFrom(&source, copyshape, shape_offset, head_rep);
    }

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

    Tensor *masterTensor() const {
        return master_tensor_;
    }
    void setMasterTensor(Tensor *master_tensor) {
        master_tensor_ = master_tensor;
    }

    vector<Tensor *> &childTensors() {
        return child_tensors_;
    }
    void addChildTensor(Tensor *child) {
        auto it = std::find(child_tensors_.begin(), child_tensors_.end(), child);
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
            assert(sum == head());
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
            assert(sum == sequence());
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
            assert(sum == dimension());
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
    Tensor *deaggregatedTensor() const {
        return deaggregated_tensor_;
    }
    Chl aggregatedDim() const {
        return aggregated_dim_;
    }

    BackendType device() const {
        return backend_->type();
    }

    Tensor &to(BackendType backend_type);
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
        assert(shape().size() == 5);
        return legacyShape(chls()[CHANNLE]);
        // switch (ctype_) {
        // case BCTHW:
        //     return legacyShape(1);
        // case BTHWC:
        //     return legacyShape(4);
        // default: return -1;
        // }
    }
    int time() {
        assert(shape().size() == 5);
        return legacyShape(chls()[TIME]);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(2);
        case BTHWC:
            return legacyShape(1);
        default: return -1;
        }
    }
    int height() {
        assert(shape().size() == 5);
        return legacyShape(chls()[HEIGHT]);
        // switch (ctype_) {
        // case BCTHW:
        //     return legacyShape(3);
        // case BTHWC:
        //     return legacyShape(2);
        // default: return -1;
        // }
    }
    int width() {
        assert(shape().size() == 5);
        return legacyShape(chls()[WIDTH]);
        // switch (ctype_) {
        // case BCTHW:
        //     return legacyShape(4);
        // case BTHWC:
        //     return legacyShape(3);
        // default: return -1;
        // }
    }
    int offset(const int b, const int c, const int t, const int h, const int w) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return (((b * channel() + c) * time() + t) * height() + h) * width() + w;
        case BTHWC:
            return (((b * time() + t) * height() + h) * width() + w) * channel() + c;
        default: return -1;
        }
    }
    template <typename Dtype>
    Dtype dataAt(const int batch, const int channel, const int time, const int height, const int width) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        return ((Dtype *)host_ptr_)[offset(batch, channel, time, height, width)];
    }
    template <typename Dtype>
    Dtype *ptrAt(const int batch, const int channel, const int time, const int height, const int width) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        return ((Dtype *)host_ptr_ + offset(batch, channel, time, height, width));
    }
    template <typename Dtype>
    void setDataAt(const int batch, const int channel, const int time, const int height, const int width, Dtype value) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
        typed_ptr[offset(batch, channel, time, height, width)] = value;
    }
    Module *module() const {
        return module_;
    }
    void setModule(Module *module) {
        module_ = module;
    }
    void transCopyShape(const vector<int> &shape) {
        reshape(shape);
    }

private:
    bool reshape(const vector<int> &shape) {
        assert(shape.size() <= 32);
        count_ = 1;
        shape_.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            assert(shape[i] >= 0);
            if (count_ != 0) {
                assert(shape[i] <= std::numeric_limits<uint64_t>::max() / count_);
            }
            count_ *= shape[i];
            shape_[i] = shape[i];
        }
        if (count_ > capacity_) {
            capacity_ = count_;
            return true;
        }
        return false;
    }
    int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
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
    Tensor &getFunc(const std::string &suffix, TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors = {});
    void getFunc(TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors = {});

    static std::vector<std::reference_wrapper<Tensor>> getStaticFunc(vector<std::string> out_names, TensorFuncType type, vector<float> float_args, vector<Tensor *> input_tensors);

public:
    uint32_t &uuid();

    TensorType &xnnTensorType();

    void forceResetHostPointer(void *ptr);

    float i8_scale = 1.f;

















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
        auto printRecursive = [&](auto&& self, int n, int c, int h, int w, int depth) -> void {
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
                        std::cout << std::endl << " ";
                    }
                    std::cout << "..." << std::endl << " ";
                    for (int h_idx = H - half; h_idx < H; ++h_idx) {
                        self(self, n, c, h_idx, 0, depth + 1);
                        if (h_idx < H - 1) {
                            std::cout << std::endl << " ";
                        }
                    }
                } else {
                    for (int h_idx = 0; h_idx < H; ++h_idx) {
                        self(self, n, c, h_idx, 0, depth + 1);
                        if (h_idx < H - 1) {
                            std::cout << std::endl << " ";
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
                        std::cout << std::endl << std::endl;
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
                std::cout << std::endl << std::endl;
            }
        }
    }

    template <typename Dtype>
    void printData() {
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
    void saveData(string ex = "") {
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
        if (ctype_ == BSHD) {
            outFile << name() << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        } else {
            outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
        }

        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (ctype_ == BSHD) {
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
        if (H == 3) {
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
        } else if (N == 1 && C == 1) {
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
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
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
            if (ctype_ == BSHD) {
                outFile << name() << ": [BSHD]shape:[" << batch() << " " << sequence() << " " << head() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;
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
    }

    void printCtype() {
        std::string ctype;
        switch (ctype_) {
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
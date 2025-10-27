#include "OpenCLRoPEOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp" // For MLLM_FP32_TO_FP16
#include <cmath>

namespace mllm {

cl_mem OpenCLRoPEOp::sin_buffer_fp32_ = nullptr;
cl_mem OpenCLRoPEOp::cos_buffer_fp32_ = nullptr;
cl_mem OpenCLRoPEOp::sin_buffer_fp16_ = nullptr;
cl_mem OpenCLRoPEOp::cos_buffer_fp16_ = nullptr;
size_t OpenCLRoPEOp::buffer_size_fp32_ = 0;
size_t OpenCLRoPEOp::buffer_size_fp16_ = 0;
vector<vector<float>> OpenCLRoPEOp::sin_table_cpu_fp32_;
vector<vector<float>> OpenCLRoPEOp::cos_table_cpu_fp32_;
int OpenCLRoPEOp::partial_dim_cached_ = -1;

// === 本地辅助函数 (sinusoidal_position_embedding_*) 保持不变 ===
namespace {
void sinusoidal_position_embedding_llama(int seq_len, int output_dim, float rope_theta,
                                         vector<vector<float>> &sin_table, vector<vector<float>> &cos_table) {
    sin_table.resize(seq_len, vector<float>(output_dim));
    cos_table.resize(seq_len, vector<float>(output_dim));
    vector<float> theta(output_dim / 2);
    for (int i = 0; i < output_dim / 2; ++i) {
        theta[i] = 1.0f / powf(rope_theta, (float)(2 * i) / output_dim);
    }
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; d += 2) {
            float t = (float)s * theta[d / 2];
            float sin_val = sinf(t);
            float cos_val = cosf(t);
            sin_table[s][d] = sin_val;
            cos_table[s][d] = cos_val;
            sin_table[s][d + 1] = sin_val;
            cos_table[s][d + 1] = cos_val;
        }
    }
}

void sinusoidal_position_embedding_huggingface(int seq_len, int output_dim, float rope_theta,
                                               vector<vector<float>> &sin_table, vector<vector<float>> &cos_table) {
    sin_table.resize(seq_len, vector<float>(output_dim / 2));
    cos_table.resize(seq_len, vector<float>(output_dim / 2));
    vector<float> theta(output_dim / 2);
    for (int i = 0; i < output_dim / 2; ++i) {
        theta[i] = 1.0f / powf(rope_theta, (float)(2 * i) / output_dim);
    }
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim / 2; ++d) {
            float t = (float)s * theta[d];
            sin_table[s][d] = sinf(t);
            cos_table[s][d] = cosf(t);
        }
    }
}
} // namespace

// === 构造函数实现 (保持不变) ===
OpenCLRoPEOp::OpenCLRoPEOp(Backend *bn, string opName, OpParam &config, int threadCount) :
    Op(bn, opName), config_(config) {
    _init(threadCount);
}
OpenCLRoPEOp::OpenCLRoPEOp(Backend *bn, string opName, int pose_type, int threadCount) :
    Op(bn, opName) {
    config_["pose_type"] = pose_type;
    _init(threadCount);
}
OpenCLRoPEOp::OpenCLRoPEOp(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount) :
    Op(bn, opName) {
    config_["pose_type"] = pose_type;
    config_["rope_theta"] = rope_theta;
    config_["max_position_embeddings"] = max_position_embeddings;
    _init(threadCount);
}
OpenCLRoPEOp::OpenCLRoPEOp(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount) :
    Op(bn, opName) {
    config_["pose_type"] = pose_type;
    config_["rope_theta"] = rope_theta;
    config_["partial_rotary_factor"] = partial_rotary_factor;
    config_["max_position_embeddings"] = max_position_embeddings;
    _init(threadCount);
}

// ✨ 修改: _init 函数，创建所有内核
void OpenCLRoPEOp::_init(int threadCount) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    pose_type_ = (RoPEType)config_.at("pose_type");
    if (config_.find("rope_theta") != config_.end()) rope_theta_ = config_.at("rope_theta");
    if (config_.find("partial_rotary_factor") != config_.end()) partial_rotary_factor_ = config_.at("partial_rotary_factor");
    if (config_.find("max_position_embeddings") != config_.end()) pos_max_ = config_.at("max_position_embeddings");

    const std::string kernel_path = "kernel/rope.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);
    cl_int err;

    kernel_llama_fp32_ = clCreateKernel(program, "rope_llama_fp32", &err);
    check_cl_error(err, "clCreateKernel rope_llama_fp32");
    kernel_hf_fp32_ = clCreateKernel(program, "rope_hf_fp32", &err);
    check_cl_error(err, "clCreateKernel rope_hf_fp32");

    kernel_llama_fp16_ = clCreateKernel(program, "rope_llama_fp16", &err);
    check_cl_error(err, "clCreateKernel rope_llama_fp16");
    kernel_hf_fp16_ = clCreateKernel(program, "rope_hf_fp16", &err);
    check_cl_error(err, "clCreateKernel rope_hf_fp16");
}

// ✨ 修改: 析构函数，释放所有资源
OpenCLRoPEOp::~OpenCLRoPEOp() {
    if (kernel_llama_fp32_) clReleaseKernel(kernel_llama_fp32_);
    if (kernel_hf_fp32_) clReleaseKernel(kernel_hf_fp32_);
    // if (sin_buffer_fp32_) clReleaseMemObject(sin_buffer_fp32_);
    // if (cos_buffer_fp32_) clReleaseMemObject(cos_buffer_fp32_);

    if (kernel_llama_fp16_) clReleaseKernel(kernel_llama_fp16_);
    if (kernel_hf_fp16_) clReleaseKernel(kernel_hf_fp16_);
    // if (sin_buffer_fp16_) clReleaseMemObject(sin_buffer_fp16_);
    // if (cos_buffer_fp16_) clReleaseMemObject(cos_buffer_fp16_);
}

/*
void OpenCLRoPEOp::_computeSinCosTable(int partial_dim) {
    if (!sin_table_cpu_fp32_.empty() && partial_dim_cached_ == partial_dim) {
        return;
    }
    partial_dim_cached_ = partial_dim;

    if (pose_type_ == LLAMAROPE) {
        sinusoidal_position_embedding_llama(pos_max_, partial_dim, rope_theta_, sin_table_cpu_fp32_, cos_table_cpu_fp32_);
    } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
        sinusoidal_position_embedding_huggingface(pos_max_, partial_dim, rope_theta_, sin_table_cpu_fp32_, cos_table_cpu_fp32_);
    } else {
        throw std::runtime_error("Unsupported RoPE type for OpenCL");
    }

    int seq_len = sin_table_cpu_fp32_.size();
    int table_dim = sin_table_cpu_fp32_[0].size();

    // --- 1. 处理 FP32 Buffer ---
    size_t new_buffer_size_fp32 = (size_t)seq_len * table_dim * sizeof(float);
    if (buffer_size_fp32_ != new_buffer_size_fp32) {
        if (sin_buffer_fp32_) clReleaseMemObject(sin_buffer_fp32_);
        if (cos_buffer_fp32_) clReleaseMemObject(cos_buffer_fp32_);
        cl_int err;
        sin_buffer_fp32_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp32, nullptr, &err);
        check_cl_error(err, "clCreateBuffer for sin_buffer_fp32_");
        cos_buffer_fp32_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp32, nullptr, &err);
        check_cl_error(err, "clCreateBuffer for cos_buffer_fp32_");
        buffer_size_fp32_ = new_buffer_size_fp32;
    }

    vector<float> sin_flat_fp32, cos_flat_fp32;
    sin_flat_fp32.reserve(seq_len * table_dim);
    cos_flat_fp32.reserve(seq_len * table_dim);
    for (int i = 0; i < seq_len; ++i) {
        sin_flat_fp32.insert(sin_flat_fp32.end(), sin_table_cpu_fp32_[i].begin(), sin_table_cpu_fp32_[i].end());
        cos_flat_fp32.insert(cos_flat_fp32.end(), cos_table_cpu_fp32_[i].begin(), cos_table_cpu_fp32_[i].end());
    }
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), sin_buffer_fp32_, CL_TRUE, 0, buffer_size_fp32_, sin_flat_fp32.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), cos_buffer_fp32_, CL_TRUE, 0, buffer_size_fp32_, cos_flat_fp32.data(), 0, nullptr, nullptr);

    // --- 2. 处理 FP16 Buffer ---
    size_t new_buffer_size_fp16 = (size_t)seq_len * table_dim * sizeof(mllm_fp16_t);
    if (buffer_size_fp16_ != new_buffer_size_fp16) {
        if (sin_buffer_fp16_) clReleaseMemObject(sin_buffer_fp16_);
        if (cos_buffer_fp16_) clReleaseMemObject(cos_buffer_fp16_);
        cl_int err;
        sin_buffer_fp16_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp16, nullptr, &err);
        check_cl_error(err, "clCreateBuffer for sin_buffer_fp16_");
        cos_buffer_fp16_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp16, nullptr, &err);
        check_cl_error(err, "clCreateBuffer for cos_buffer_fp16_");
        buffer_size_fp16_ = new_buffer_size_fp16;
    }

    vector<mllm_fp16_t> sin_flat_fp16, cos_flat_fp16;
    sin_flat_fp16.reserve(seq_len * table_dim);
    cos_flat_fp16.reserve(seq_len * table_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < table_dim; ++j) {
            sin_flat_fp16.push_back(MLLM_FP32_TO_FP16(sin_table_cpu_fp32_[i][j]));
            cos_flat_fp16.push_back(MLLM_FP32_TO_FP16(cos_table_cpu_fp32_[i][j]));
        }
    }
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), sin_buffer_fp16_, CL_TRUE, 0, buffer_size_fp16_, sin_flat_fp16.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), cos_buffer_fp16_, CL_TRUE, 0, buffer_size_fp16_, cos_flat_fp16.data(), 0, nullptr, nullptr);
}
*/
void OpenCLRoPEOp::_computeSinCosTable(int partial_dim) {
    if (!sin_table_cpu_fp32_.empty() && partial_dim_cached_ == partial_dim) {
        return;
    }
    partial_dim_cached_ = partial_dim;

    if (pose_type_ == LLAMAROPE) {
        sinusoidal_position_embedding_llama(pos_max_, partial_dim, rope_theta_, sin_table_cpu_fp32_, cos_table_cpu_fp32_);
    } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
        sinusoidal_position_embedding_huggingface(pos_max_, partial_dim, rope_theta_, sin_table_cpu_fp32_, cos_table_cpu_fp32_);
    } else {
        throw std::runtime_error("Unsupported RoPE type for OpenCL");
    }

    int seq_len = sin_table_cpu_fp32_.size();
    int table_dim = sin_table_cpu_fp32_[0].size();

    // FP32
    size_t new_buffer_size_fp32 = (size_t)seq_len * table_dim * sizeof(float);
    if (buffer_size_fp32_ != new_buffer_size_fp32) {
        if (sin_buffer_fp32_) clReleaseMemObject(sin_buffer_fp32_);
        if (cos_buffer_fp32_) clReleaseMemObject(cos_buffer_fp32_);
        cl_int err;
        sin_buffer_fp32_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp32, nullptr, &err);
        check_cl_error(err, "clCreateBuffer sin_buffer_fp32_");
        cos_buffer_fp32_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp32, nullptr, &err);
        check_cl_error(err, "clCreateBuffer cos_buffer_fp32_");
        buffer_size_fp32_ = new_buffer_size_fp32;
    }

    vector<float> sin_flat_fp32, cos_flat_fp32;
    sin_flat_fp32.reserve(seq_len * table_dim);
    cos_flat_fp32.reserve(seq_len * table_dim);
    for (int i = 0; i < seq_len; ++i) {
        sin_flat_fp32.insert(sin_flat_fp32.end(), sin_table_cpu_fp32_[i].begin(), sin_table_cpu_fp32_[i].end());
        cos_flat_fp32.insert(cos_flat_fp32.end(), cos_table_cpu_fp32_[i].begin(), cos_table_cpu_fp32_[i].end());
    }
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), sin_buffer_fp32_, CL_TRUE, 0, buffer_size_fp32_, sin_flat_fp32.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), cos_buffer_fp32_, CL_TRUE, 0, buffer_size_fp32_, cos_flat_fp32.data(), 0, nullptr, nullptr);

    // FP16
    size_t new_buffer_size_fp16 = (size_t)seq_len * table_dim * sizeof(mllm_fp16_t);
    if (buffer_size_fp16_ != new_buffer_size_fp16) {
        if (sin_buffer_fp16_) clReleaseMemObject(sin_buffer_fp16_);
        if (cos_buffer_fp16_) clReleaseMemObject(cos_buffer_fp16_);
        cl_int err;
        sin_buffer_fp16_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp16, nullptr, &err);
        check_cl_error(err, "clCreateBuffer sin_buffer_fp16_");
        cos_buffer_fp16_ = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_ONLY, new_buffer_size_fp16, nullptr, &err);
        check_cl_error(err, "clCreateBuffer cos_buffer_fp16_");
        buffer_size_fp16_ = new_buffer_size_fp16;
    }

    vector<mllm_fp16_t> sin_flat_fp16, cos_flat_fp16;
    sin_flat_fp16.reserve(seq_len * table_dim);
    cos_flat_fp16.reserve(seq_len * table_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < table_dim; ++j) {
            sin_flat_fp16.push_back(MLLM_FP32_TO_FP16(sin_table_cpu_fp32_[i][j]));
            cos_flat_fp16.push_back(MLLM_FP32_TO_FP16(cos_table_cpu_fp32_[i][j]));
        }
    }
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), sin_buffer_fp16_, CL_TRUE, 0, buffer_size_fp16_, sin_flat_fp16.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocl_backend_->getQueue(), cos_buffer_fp16_, CL_TRUE, 0, buffer_size_fp16_, cos_flat_fp16.data(), 0, nullptr, nullptr);
}

ErrorCode OpenCLRoPEOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int partial_dim = inputs[0]->dimension() * partial_rotary_factor_;
    _computeSinCosTable(partial_dim);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype()); // 确保输出类型与输入一致
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLRoPEOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

// ✨ 修改: execute 函数，根据数据类型选择内核
ErrorCode OpenCLRoPEOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    // RoPE是in-place操作，先将输入数据拷贝到输出
    clEnqueueCopyBuffer(ocl_backend_->getQueue(), ocl_backend_->get_cl_mem(*input), ocl_backend_->get_cl_mem(*output), 0, 0, input->size(), 0, nullptr, nullptr);

    cl_kernel kernel_to_use = nullptr;
    cl_mem sin_buf_to_use = nullptr;
    cl_mem cos_buf_to_use = nullptr;
    int partial_dim = output->dimension() * partial_rotary_factor_;
    size_t d_work_size = 0;

    if (output->dtype() == MLLM_TYPE_F32) {
        sin_buf_to_use = sin_buffer_fp32_;
        cos_buf_to_use = cos_buffer_fp32_;
        if (pose_type_ == LLAMAROPE) {
            kernel_to_use = kernel_llama_fp32_;
            d_work_size = partial_dim / 2;
        } else { // HFHUBROPE, MLAROPE
            kernel_to_use = kernel_hf_fp32_;
            d_work_size = partial_dim / 2;
        }
    } else if (output->dtype() == MLLM_TYPE_F16) {
        sin_buf_to_use = sin_buffer_fp16_;
        cos_buf_to_use = cos_buffer_fp16_;
        if (pose_type_ == LLAMAROPE) {
            kernel_to_use = kernel_llama_fp16_;
            d_work_size = partial_dim / 2;
        } else { // HFHUBROPE, MLAROPE
            kernel_to_use = kernel_hf_fp16_;
            d_work_size = partial_dim / 2;
        }
    } else {
        std::runtime_error("Unsupported RoPE data type for OpenCL: " + std::to_string(output->dtype()));
        return NOT_SUPPORT;
    }

    if (kernel_to_use == nullptr || sin_buf_to_use == nullptr || cos_buf_to_use == nullptr) {
        std::runtime_error("RoPE kernel or buffers not initialized properly.");
        return NOT_SUPPORT; // 安全检查
    }

    cl_mem data_buf = ocl_backend_->get_cl_mem(*output);
    int head_dim = output->head() * output->batch(); // 将 batch 和 head 合并
    int seq_len = output->sequence();

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &data_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &sin_buf_to_use);
    clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &cos_buf_to_use);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &partial_dim);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &head_dim);
    clSetKernelArg(kernel_to_use, 5, sizeof(int), &seq_len);
    clSetKernelArg(kernel_to_use, 6, sizeof(int), &pos_offset_);

    size_t global_work_size[3] = {d_work_size, (size_t)seq_len, (size_t)head_dim};

    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name(), event);
    check_cl_error(err, "clEnqueueNDRangeKernel for RoPE");

    pos_offset_ += output->sequence();
    return MLLM_NO_ERROR;
}

// Creator 的实现保持不变
Op *OpenCLRoPEOpCreator::create(OpParam op_param, Backend *bn, string name, int threadCount) const {
    auto it = op_param.find("rope_type");
    if (it != op_param.end()) {
        return new OpenCLRoPEOp(bn, name, op_param, threadCount);
    }
    int pose_type = op_param["pose_type"];
    if (op_param.find("rope_theta") == op_param.end()) {
        return new OpenCLRoPEOp(bn, name, pose_type, threadCount);
    }
    float rope_theta = op_param["rope_theta"];
    int max_position_embeddings = op_param["max_position_embeddings"];
    if (op_param.find("partial_rotary_factor") == op_param.end()) {
        return new OpenCLRoPEOp(bn, name, pose_type, rope_theta, max_position_embeddings, threadCount);
    }
    float partial_rotary_factor = op_param["partial_rotary_factor"];
    return new OpenCLRoPEOp(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings, threadCount);
}

} // namespace mllm
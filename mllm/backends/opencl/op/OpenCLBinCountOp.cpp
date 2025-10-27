// opencl/op/OpenCLBinCountOp.cpp

#include "OpenCLBinCountOp.hpp"
#include "Types.hpp"
#include <algorithm>
#include <vector>
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"
// opencl/op/OpenCLBinCountOp.cpp
namespace mllm {

// 构造函数、reshape、setUp函数与上一版相同，这里省略...
OpenCLBinCountOp::OpenCLBinCountOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/bincount.cl";
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }

    auto program = ocl_backend_->getProgram(kernel_path, build_options);
    cl_int err;

    kernel_map_["bincount_count_fp32"] = clCreateKernel(program, "bincount_count_fp32", &err);
    check_cl_error(err, "clCreateKernel bincount_count_fp32");
    kernel_map_["bincount_count_fp16"] = clCreateKernel(program, "bincount_count", &err);
    check_cl_error(err, "clCreateKernel bincount_count_fp16");
    kernel_map_["cast_int_to_float"] = clCreateKernel(program, "cast_int_to_float", &err);
    check_cl_error(err, "clCreateKernel cast_int_to_float");
    kernel_map_["cast_int_to_half"] = clCreateKernel(program, "cast_int_to_half", &err);
    check_cl_error(err, "clCreateKernel cast_int_to_half");
}

OpenCLBinCountOp::~OpenCLBinCountOp() {
    for (auto &pair : kernel_map_) {
        if (pair.second) {
            clReleaseKernel(pair.second);
        }
    }
}

ErrorCode OpenCLBinCountOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[0]->batch() == 1 && inputs[0]->sequence() == 1 && inputs[0]->head() == 1);
    outputs[0]->reshape(1, 1, 1, 0);
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLBinCountOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->dtype() != MLLM_TYPE_F32 && inputs[0]->dtype() != MLLM_TYPE_F16) {
        return NOT_SUPPORT;
    }
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLBinCountOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    // ocl_backend_->finishQueue(); // todo 同步问题

    input->cpu();
    int size = input->dimension();
    int max_val = 0;
    if (size > 0) {
        if (input->dtype() == MLLM_TYPE_F32) {
            float *data_ptr = input->hostPtr<float>();
            max_val = static_cast<int>(*std::max_element(data_ptr, data_ptr + size));
        } else { // MLLM_TYPE_F16
            std::vector<float> float_vec(size);
            for (int i = 0; i < size; ++i) {
                float_vec[i] = MLLM_FP16_TO_FP32(input->dataAt<mllm_fp16_t>(0, 0, 0, i));
            }
            max_val = static_cast<int>(*std::max_element(float_vec.begin(), float_vec.end()));
        }
    }
    int output_size = max_val + 1;
    output->reshape(1, 1, 1, output_size);
    output->alloc();
    cl_int err;
    cl_mem tmp_count_buffer = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_WRITE, output_size * sizeof(int), nullptr, &err);
    check_cl_error(err, "clCreateBuffer for tmp_count_buffer");

    int zero = 0;
    cl_event fill_event;
    err = clEnqueueFillBuffer(ocl_backend_->getQueue(), tmp_count_buffer, &zero, sizeof(int), 0, output_size * sizeof(int), 0, nullptr, &fill_event);
    check_cl_error(err, "clEnqueueFillBuffer for tmp_count_buffer");
    ocl_backend_->addProfilingEvent("bincount_fill_zero", fill_event);

    input->cl();

    cl_kernel count_kernel = (input->dtype() == MLLM_TYPE_F32) ? kernel_map_["bincount_count_fp32"] : kernel_map_["bincount_count_fp16"];
    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    clSetKernelArg(count_kernel, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(count_kernel, 1, sizeof(cl_mem), &tmp_count_buffer);
    clSetKernelArg(count_kernel, 2, sizeof(int), &size);
    clSetKernelArg(count_kernel, 3, sizeof(int), &max_val);

    const size_t local_work_size = 256;
    const size_t global_work_size = ((size > 0 ? size : 1) + local_work_size - 1) / local_work_size * local_work_size;

    cl_event count_event;
    err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), count_kernel, 1, nullptr, &global_work_size, &local_work_size, 1, &fill_event, &count_event);
    check_cl_error(err, "clEnqueueNDRangeKernel for bincount_count");
    ocl_backend_->addProfilingEvent("bincount_count", count_event);

    cl_kernel cast_kernel = (output->dtype() == MLLM_TYPE_F32) ? kernel_map_["cast_int_to_float"] : kernel_map_["cast_int_to_half"];
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);
    clSetKernelArg(cast_kernel, 0, sizeof(cl_mem), &tmp_count_buffer);
    clSetKernelArg(cast_kernel, 1, sizeof(cl_mem), &out_buf);
    clSetKernelArg(cast_kernel, 2, sizeof(int), &output_size);

    const size_t cast_global_work_size = ((output_size > 0 ? output_size : 1) + local_work_size - 1) / local_work_size * local_work_size;
    cl_event cast_event;
    err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), cast_kernel, 1, nullptr, &cast_global_work_size, &local_work_size, 1, &count_event, &cast_event);
    check_cl_error(err, "clEnqueueNDRangeKernel for cast");
    ocl_backend_->addProfilingEvent("bincount_cast", cast_event);
    clReleaseMemObject(tmp_count_buffer);
    clWaitForEvents(1, &cast_event);
    clReleaseEvent(fill_event);
    clReleaseEvent(count_event);
    clReleaseEvent(cast_event);

    return MLLM_NO_ERROR;
}
} // namespace mllm
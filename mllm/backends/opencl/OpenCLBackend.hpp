#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include "Backend.hpp"
#include "OpenCLMemoryManager.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <dlfcn.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace mllm {
struct DeviceMemory;
class Layer;

// 【关键修正 1】: 在类定义之前，提供 OpenCLSymbols 的前向声明
#if defined(MLLM_TARGET_ANDROID)
struct OpenCLSymbols;
#endif

class OpenCLBackend : public Backend {
public:
    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const = 0;
        virtual ~Creator() = default;
    };

    OpenCLBackend(const BackendConfig &config);
    ~OpenCLBackend() override;

    cl_context getContext() const {
        return context_;
    }
    cl_device_id getDevice() const {
        return device_;
    }
    cl_command_queue getQueue() const {
        return queue_;
    }
    void finishQueue();
    cl_program getProgram(const std::string &program_name, const std::string &build_options = "");

    Op *opCreate(const OpParam &op_param, std::string name = "", int threadCount = 4) override;
    TensorFunction *funcCreate(TensorFuncType type) override;
    void alloc_device(DeviceMemory &mem, DataType dtype) override;
    void free_device(DeviceMemory &mem) override;
    void copy_from_host(const DeviceMemory &dest, const void *src) override;
    void copy_to_host(void *dest, const DeviceMemory &src) override;
    std::vector<Tensor> runLayer(Layer *layer, std::vector<Tensor> inputs, int N) override;
    std::vector<Tensor> runOp(Op *op, std::vector<Tensor> input, std::vector<std::string> out_names, bool in_place) override;
    std::vector<Tensor> runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) override;
    void registerOps() override;
    void registerFuncs() override;
    void convert_fp_data(Tensor *src, Tensor *dest) override;
    bool load_from_file(Tensor *tensor, ParamLoader *loader) override;

    cl_mem get_cl_mem(const Tensor &tensor) const;
    bool is_image_from_buffer_supported() const {
        return image_from_buffer_supported_;
    }
    cl_uint get_image_pitch_alignment_in_bytes() const {
        return image_pitch_alignment_bytes_;
    }
    // bool &has_fp16_support() {
    //     return has_fp16_support_;
    // }
    bool has_fp16_support() const { // 1. 移除引用'&'  2. 增加const关键字
        return has_fp16_support_;
    }
    void addProfilingEvent(const std::string &op_name, cl_event event);
    void reportProfilingResult(); // 新增分析结果的函数

    size_t getMaxImage2dWidth() const {
        return max_image2d_width_;
    }

#if defined(MLLM_TARGET_ANDROID)
public:
    // 【关键修正 2】: getSymbols 的声明保持 public，返回一个指向前向声明类型的指针
    static OpenCLSymbols *getSymbols();

private:
    static void loadOpenCLSymbols();
    static OpenCLSymbols symbols_;
#endif

private:
    static std::shared_ptr<OpenCLMemoryManager> createMemoryManager(cl_context &context, cl_device_id &device);

    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;

    std::map<std::string, cl_program> program_cache_;
    std::map<OpType, std::shared_ptr<Creator>> op_creator_map_;

    bool image_from_buffer_supported_ = false;
    cl_uint image_pitch_alignment_bytes_ = 0;
    bool has_fp16_support_ = false;

    size_t max_image2d_width_ = 0;
    std::string kernel_root_path_;

    cl_kernel kernel_fp32_to_fp16_buffer_ = nullptr;
    cl_kernel kernel_fp16_to_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_to_fp16_image_ = nullptr;
    cl_kernel kernel_fp16_to_fp32_image_ = nullptr;
    cl_sampler sampler_ = nullptr;

    std::vector<cl_event> profiling_events_;
    std::map<cl_event, std::string> event_op_names_;
};

class OpenCLBackendCreator : public BackendCreator {
public:
    Backend *create(BackendConfig config) override {
        return new OpenCLBackend(config);
    }
};

} // namespace mllm

#endif // OPENCL_BACKEND_H
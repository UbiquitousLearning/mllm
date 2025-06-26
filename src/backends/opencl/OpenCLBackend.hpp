#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include "Backend.hpp"
#include "OpenCLMemoryManager.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// 外部函数声明，用于错误检查
// void check_cl_error(cl_int err, const std::string& operation);

namespace mllm {
struct DeviceMemory;
class Tensor;

class OpenCLBackend : public Backend {
public:

    // 为工厂模式定义的 Creator 基类
    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const = 0;
        virtual ~Creator() = default;
    };

    
    OpenCLBackend(const BackendConfig &config);
    ~OpenCLBackend() override;

    // --- OpenCL 特有方法 ---
    cl_context getContext() const { return context_; }
    cl_device_id getDevice() const { return device_; }
    cl_command_queue getQueue() const { return queue_; }
    void finishQueue();
    // 获取或编译并缓存 Kernel Program
    cl_program getProgram(const std::string& program_name, const std::string& build_options = "");

    // --- 实现基类的纯虚函数 ---
    Op *opCreate(const OpParam &op_param, std::string name = "", int threadCount = 4) override;
    TensorFunction *funcCreate(TensorFuncType type) override;

    void alloc_device(DeviceMemory &mem, DataType dtype) override;
    void free_device(DeviceMemory &mem) override;
    void copy_from_host(const DeviceMemory &dest, const void *src) override;
    void copy_to_host(void *dest, const DeviceMemory &src) override;

    // 注意：下面几个函数的具体实现依赖于你的框架逻辑，这里暂时提供桩实现 (stub)
    std::vector<Tensor> runLayer(Layer *layer, std::vector<Tensor> inputs, int N) override;
    std::vector<Tensor> runOp(Op *op, std::vector<Tensor> input, std::vector<std::string> out_names, bool in_place) override;
    std::vector<Tensor> runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) override;
    
    void registerOps() override;
    void registerFuncs() override;
    void convert_fp_data(Tensor *src, Tensor *dest) override;

    cl_mem get_cl_mem(const Tensor &tensor) const;
  /**
     * @brief 检查当前设备是否支持从Buffer创建Image的零拷贝扩展。
     */
    bool is_image_from_buffer_supported() const { return image_from_buffer_supported_; }

    /**
     * @brief 获取设备要求的Image行间距对齐字节数。
     */
    cl_uint get_image_pitch_alignment_in_bytes() const { return image_pitch_alignment_bytes_; }



private:
    // 构造函数辅助函数，用于解决基类和子类成员的初始化顺序问题
    static std::shared_ptr<OpenCLMemoryManager> createMemoryManager(cl_context& context, cl_device_id& device);
    
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;

    // 内核程序缓存
    std::map<std::string, cl_program> program_cache_;
    std::map<OpType, std::shared_ptr<Creator>> op_creator_map_;

    /**
     * @brief 标记设备是否支持 cl_khr_image2d_from_buffer 扩展。
     */
    bool image_from_buffer_supported_ = false;

    /**
     * @brief 存储硬件要求的Image行间距对齐字节数。
     */
    cl_uint image_pitch_alignment_bytes_ = 0;
     /**
     * @brief 标记设备是否支持 cl_khr_fp16 扩展。
     */
    bool has_fp16_support_ = false;
private:
    cl_kernel kernel_fp32_to_fp16_buffer_ = nullptr;
    cl_kernel kernel_fp16_to_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_to_fp16_image_ = nullptr; // 新增
    cl_kernel kernel_fp16_to_fp32_image_ = nullptr; // 新增
    cl_sampler sampler_ = nullptr;
};


class OpenCLBackendCreator : public BackendCreator {
public:
    Backend *create(BackendConfig config) override {
        return new OpenCLBackend(config);
    }
};


} // namespace mllm

#endif // OPENCL_BACKEND_H
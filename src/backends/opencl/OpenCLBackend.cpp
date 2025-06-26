#include "OpenCLBackend.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "Tensor.hpp"

#include "./op/OpenCLAddFuncOp.hpp"
#include "utils/OpenCLTools.hpp"

// 错误检查函数
void check_cl_error(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error during " << operation << " (" << err << ")" << std::endl;
        //应该调度到cpu
        throw std::runtime_error("OpenCL Error: " + operation);
    }
}

// 从文件加载内核源码的辅助函数
std::string load_file_contents(const char* filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in) {
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    throw std::runtime_error(std::string("Could not open file: ") + filename);
}


namespace mllm {

// 静态辅助函数，用于在构造函数初始化列表中创建 MemoryManager
// 这样可以确保在调用基类构造函数之前，context 和 device 已经被正确初始化
std::shared_ptr<OpenCLMemoryManager> OpenCLBackend::createMemoryManager(cl_context& context, cl_device_id& device) {
    cl_int err;
    cl_platform_id platform;
    
    // 1. 获取平台
    err = clGetPlatformIDs(1, &platform, nullptr);
    check_cl_error(err, "clGetPlatformIDs");

    // 2. 获取设备 (优先GPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }
    check_cl_error(err, "clGetDeviceIDs");

    // 3. 创建上下文
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check_cl_error(err, "clCreateContext");

    // 4. 创建并返回 OpenCLMemoryManager
    return std::make_shared<OpenCLMemoryManager>(context);
} // namespace mllm

OpenCLBackend::OpenCLBackend(const BackendConfig &config) : Backend() {
    mem_manager_ = createMemoryManager(context_, device_);
    cl_int err;
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    check_cl_error(err, "clCreateCommandQueue");


    // =======================================================================
    // ========== 检查扩展支持和获取对齐值 ==========
    // =======================================================================
    size_t extensions_size;
    // 获取扩展字符串的长度
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensions_size);
    std::string extensions(extensions_size, ' ');
    // 获取完整的扩展字符串
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, extensions_size, &extensions[0], nullptr);

    if (extensions.find("cl_khr_fp16") != std::string::npos) {
        this->has_fp16_support_ = true;
    } else {
        this->has_fp16_support_ = false;
    }
    // 检查是否包含我们需要的扩展
    if (extensions.find("cl_khr_image2d_from_buffer") != std::string::npos) {
        this->image_from_buffer_supported_ = true;
        
        // 如果支持，则进一步查询行间距对齐要求（单位：字节）
        // CL_DEVICE_IMAGE_PITCH_ALIGNMENT 返回的是像素单位，需要转换，不如此处直接查询字节对齐可靠。
        // CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT 是一个更通用的字节对齐保证。
        clGetDeviceInfo(device_, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(cl_uint), &this->image_pitch_alignment_bytes_, nullptr);
        
        // 根据规范，如果返回0，则表示没有特殊的对齐要求，我们可以认为是1字节对齐。
        if (this->image_pitch_alignment_bytes_ == 0) {
            this->image_pitch_alignment_bytes_ = 1;
        }
    } else {
        // 如果不支持，明确设置为false
        this->image_from_buffer_supported_ = false;
        this->image_pitch_alignment_bytes_ = 0;
    }
    // =======================================================================
    const std::string convert_kernel_path = get_kernel_path(__FILE__, "./kernel/convert.cl");
    std::string build_options = "";
    
    // 如果硬件支持FP16，我们通过宏定义告诉编译器去编译高性能版本的内核
    if (this->has_fp16_support_) {
        build_options += " -DSUPPORTS_FP16";
    }
    
    cl_program convert_program = getProgram(convert_kernel_path, build_options);

    // --- 根据硬件能力，创建并加载相应版本的内核 ---
    if (this->has_fp16_support_) {
        // **高性能路径**: 加载使用原生 `half` 的内核
        kernel_fp32_to_fp16_buffer_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_buffer_ext", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_buffer_ext");
        kernel_fp16_to_fp32_buffer_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_buffer_ext", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_buffer_ext");

        // Image内核只有在支持FP16时才有意义
        kernel_fp32_to_fp16_image_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_image2d", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_image2d");
        kernel_fp16_to_fp32_image_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_image2d", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_image2d");

    } else {
        // **兼容性路径**: 加载使用 `ushort` 模拟的内核
        kernel_fp32_to_fp16_buffer_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_buffer_compat", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_buffer_compat");
        kernel_fp16_to_fp32_buffer_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_buffer_compat", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_buffer_compat");
        // 在不原生支持FP16的设备上，Image转换通常也不可行或无意义，所以不加载Image内核
    }
    
    sampler_ = clCreateSampler(context_, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler in Backend");
    
    this->type_ = MLLM_OPENCL;
    registerOps();
}

OpenCLBackend::~OpenCLBackend() {
    // 释放所有缓存的 cl_program
    if (kernel_fp32_to_fp16_buffer_) clReleaseKernel(kernel_fp32_to_fp16_buffer_);
    if (kernel_fp16_to_fp32_buffer_) clReleaseKernel(kernel_fp16_to_fp32_buffer_);
    if (kernel_fp32_to_fp16_image_) clReleaseKernel(kernel_fp32_to_fp16_image_);
    if (kernel_fp16_to_fp32_image_) clReleaseKernel(kernel_fp16_to_fp32_image_);
    if (sampler_) clReleaseSampler(sampler_);

    // (保留原有释放 program 和 queue 的代码)
    for (auto const& [key, program] : program_cache_) {
        if (program) {
            clReleaseProgram(program);
        }
    }
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    // std::cout << "OpenCLBackend cleaned up." << std::endl;
}

void OpenCLBackend::finishQueue() {
    if(queue_) {
        clFinish(queue_);
    }
}

cl_program OpenCLBackend::getProgram(const std::string& program_name, const std::string& build_options) {
    // 检查缓存
    auto it = program_cache_.find(program_name);
    if (it != program_cache_.end()) {
        return it->second;
    }

    // 加载和编译
    std::string kernel_source = load_file_contents(program_name.c_str());
    const char* source_ptr = kernel_source.c_str();
    size_t source_len = kernel_source.length();
    cl_int err;

    cl_program program = clCreateProgramWithSource(context_, 1, &source_ptr, &source_len, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_, build_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::string error_msg = "Kernel build error for " + program_name + ":\n" + log.data();
        throw std::runtime_error(error_msg);
    }
    
    // 存入缓存并返回
    program_cache_[program_name] = program;
    return program;
}
void OpenCLBackend::alloc_device(DeviceMemory &mem, DataType dtype) {
    if (context_ == nullptr) throw std::runtime_error("OpenCL context is not initialized.");
    cl_int err;
    switch (mem.type) {
        case MEM_TYPE_BUFFER: {
            // **新增逻辑**：检查是否是创建 Image 兼容 Buffer 的特殊请求
            if (mem.image_width > 0 && mem.image_height > 0 && image_from_buffer_supported_) {
                // === 创建 Image 兼容的、带行间距对齐的 Buffer ===
                
                // 1. 计算满足硬件对齐要求的行间距（字节单位）
                const size_t pixel_width = mem.image_width; // 此时 image_width 是像素数
                const cl_uint pitch_alignment = image_pitch_alignment_bytes_;
                
                size_t row_pitch = pixel_width * 4 * sizeof(float); // 理论行间距
                if (pitch_alignment > 0 && row_pitch % pitch_alignment != 0) {
                    row_pitch = (row_pitch + pitch_alignment - 1) / pitch_alignment * pitch_alignment;
                }

                // 2. 计算带填充的总 buffer 大小
                const size_t padded_buffer_size = mem.image_height * row_pitch;
                
                // 3. 更新 DeviceMemory 结构体，保存关键信息
                mem.size_in_bytes = padded_buffer_size; // 使用计算出的、更大的尺寸
                mem.image_row_pitch_in_bytes = row_pitch; // **保存行间距，为零拷贝做准备**
                
                // 4. 使用新的总大小创建 Buffer
                mem.handle = clCreateBuffer(context_, CL_MEM_READ_WRITE, padded_buffer_size, nullptr, &err);
                check_cl_error(err, "clCreateBuffer for Image-Compatible Buffer");

            } else {
                // === 创建一个标准的、紧密排列的 Buffer ===
                mem.handle = clCreateBuffer(context_, CL_MEM_READ_WRITE, mem.size_in_bytes, nullptr, &err);
                check_cl_error(err, "clCreateBuffer for Standard Buffer");
            }
            break;
        }
        case MEM_TYPE_IMAGE_2D: {
            // cl_image_format format = {CL_RGBA, CL_FLOAT};
             cl_image_format format = {CL_RGBA}; // 通道顺序不变
            
            // **核心修改：根据dtype设置通道数据类型**
            switch (dtype) {
                case MLLM_TYPE_F32:
                    format.image_channel_data_type = CL_FLOAT;
                    break;
                case MLLM_TYPE_F16:
                    format.image_channel_data_type = CL_HALF_FLOAT;
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for Image2D creation.");
            }
            cl_image_desc desc = {};
            desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width = mem.image_width;
            desc.image_height = mem.image_height;
            mem.handle = clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
            check_cl_error(err, "clCreateImage");
            break;
        }
        default: throw std::runtime_error("Unsupported device memory type for OpenCL.");
    }
}

void OpenCLBackend::free_device(DeviceMemory &mem) {
    if (mem.handle != nullptr) {
        clReleaseMemObject(static_cast<cl_mem>(mem.handle));
        mem.handle = nullptr;
    }
}

void OpenCLBackend::copy_from_host(const DeviceMemory &dest, const void *src) {
    if (dest.handle == nullptr || src == nullptr) return;
    cl_mem dest_handle = static_cast<cl_mem>(dest.handle);
    switch (dest.type) {
        case MEM_TYPE_BUFFER:{
            // 检查这个Buffer是否是Image兼容的
            if (dest.image_row_pitch_in_bytes > 0 && dest.image_height > 0) {
                // **使用 clEnqueueWriteBufferRect 来处理行间距**
                const size_t buffer_origin[3] = {0, 0, 0};
                const size_t host_origin[3] = {0, 0, 0};
                
                // region_in_bytes: {width, height, depth}
                const size_t region_in_bytes[3] = {
                    dest.image_width * 4 * sizeof(float), // 拷贝区域的宽度（字节）
                    dest.image_height,                    // 拷贝区域的高度（行数）
                    1                                     // 深度为1
                };

                clEnqueueWriteBufferRect(
                    queue_,
                    dest_handle,
                    CL_TRUE, // 阻塞调用
                    buffer_origin,
                    host_origin,
                    region_in_bytes,
                    dest.image_row_pitch_in_bytes, // Buffer中的行间距
                    0,                             // Buffer中的切片间距
                    dest.image_width * 4 * sizeof(float), // Host内存中的行间距（假设紧密排列）
                    0,                             // Host内存中的切片间距
                    src,
                    0, nullptr, nullptr);
            } else {
                // 标准的Buffer拷贝
                clEnqueueWriteBuffer(queue_, dest_handle, CL_TRUE, 0, dest.size_in_bytes, src, 0, nullptr, nullptr);
            }
            break;
        }
        case MEM_TYPE_IMAGE_2D: {
            const size_t origin[3] = {0, 0, 0};
            const size_t region[3] = {dest.image_width, dest.image_height, 1};
            clEnqueueWriteImage(queue_, dest_handle, CL_TRUE, origin, region, 0, 0, src, 0, nullptr, nullptr);
            break;
        }
        default: throw std::runtime_error("Unsupported copy for this memory type.");
    }
}

void OpenCLBackend::copy_to_host(void *dest, const DeviceMemory &src) {
    if (dest == nullptr || src.handle == nullptr) return;
    cl_mem src_handle = static_cast<cl_mem>(src.handle);
    switch (src.type) {
        case MEM_TYPE_BUFFER:
            clEnqueueReadBuffer(queue_, src_handle, CL_TRUE, 0, src.size_in_bytes, dest, 0, nullptr, nullptr);
            break;
        case MEM_TYPE_IMAGE_2D: {
            const size_t origin[3] = {0, 0, 0};
            const size_t region[3] = {src.image_width, src.image_height, 1};
            clEnqueueReadImage(queue_, src_handle, CL_TRUE, origin, region, 0, 0, dest, 0, nullptr, nullptr);
            break;
        }
        default: throw std::runtime_error("Unsupported copy for this memory type.");
    }
}

cl_mem OpenCLBackend::get_cl_mem(const Tensor &tensor) const {
    if (tensor.backend() != this) throw std::runtime_error("Tensor is not on this backend.");
    const auto& mem = tensor.device_memory();
    if (mem.handle == nullptr) throw std::runtime_error("Tensor CL handle is null.");
    return static_cast<cl_mem>(mem.handle);
}


Op *OpenCLBackend::opCreate(const OpParam &op_param, std::string name, int threadCount) {
    OpType type = (OpType)op_param.find("type")->second;
    auto it = op_creator_map_.find(type);
    if (it == op_creator_map_.end()) {
        throw std::runtime_error("Op type " + std::to_string(type) + " not supported by OpenCLBackend");
    }
    return it->second->create(op_param, this, name, threadCount);
}

TensorFunction *OpenCLBackend::funcCreate(TensorFuncType type) {
    throw std::runtime_error("funcCreate not implemented for OpenCLBackend");
    return nullptr;
}

void OpenCLBackend::registerOps() {
    // 在这里注册该后端支持的所有 Op
    op_creator_map_[F_TTADD] = std::make_shared<OpenCLAddFuncOpCreator>();
    // std::cout << "OpenCLBackend ops registered." << std::endl;
}

void OpenCLBackend::registerFuncs() {
    std::cout << "OpenCLBackend funcs is abanded." << std::endl;
}


// 用以下最终完整版替换您的 convert_fp_data 函数
void OpenCLBackend::convert_fp_data(Tensor *src, Tensor *dest) {
    if (src->device() != MLLM_OPENCL || dest->device() != MLLM_OPENCL) {
        throw std::runtime_error("Type conversion on GPU requires both tensors to be on OpenCL backend.");
    }
    
    // 获取源和目标的内存描述符
    auto& src_mem = src->device_memory();
    auto& dest_mem = dest->device_memory();

    // ===================================================
    // ============   第一层决策：根据内存类型   ============
    // ===================================================

    if (src_mem.type == MEM_TYPE_BUFFER) {
        // ========== Buffer-to-Buffer 转换路径 ==========
        if (dest_mem.type != MEM_TYPE_BUFFER) throw std::runtime_error("Destination must be a Buffer for Buffer conversion.");

        cl_kernel kernel_to_use = nullptr;
        if (src->dtype() == MLLM_TYPE_F32 && dest->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp32_to_fp16_buffer_;
        } else if (src->dtype() == MLLM_TYPE_F16 && dest->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp16_to_fp32_buffer_;
        } else {
            if(src->dtype() == dest->dtype()) return;
            throw std::runtime_error("Unsupported Buffer conversion types.");
        }

        cl_mem src_buf = get_cl_mem(*src);
        cl_mem dest_buf = get_cl_mem(*dest);
        int count = src->count();
        
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dest_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(int), &count);

        const size_t global_work_size[1] = { (size_t)count };
        clEnqueueNDRangeKernel(queue_, kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    } else if (src_mem.type == MEM_TYPE_IMAGE_2D) {

         if (dest_mem.type != MEM_TYPE_IMAGE_2D) throw std::runtime_error("Destination must be an Image for Image conversion.");

        cl_kernel kernel_to_use = nullptr;
        
        if (src->dtype() == MLLM_TYPE_F32 && dest->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp32_to_fp16_image_;
        } else if (src->dtype() == MLLM_TYPE_F16 && dest->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp16_to_fp32_image_;
        } else {
            if(src->dtype() == dest->dtype()) return;
            throw std::runtime_error("Unsupported Image conversion types.");
        }
        
        // **新增**: 健壮性检查，如果内核未创建（例如硬件不支持FP16），则报错
        if (!kernel_to_use) {
            throw std::runtime_error("Image conversion kernel is not available. This may be due to lack of FP16 hardware support.");
        }
        
        cl_mem src_img = get_cl_mem(*src);
        cl_mem dest_img = get_cl_mem(*dest);
        const int width = src_mem.image_width;
        const int height = src_mem.image_height;

        // **修改**: 取消注释并使用成员变量 sampler_
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &src_img);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &dest_img);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &height);

        const size_t global_work_size[2] = { (size_t)width, (size_t)height };
        clEnqueueNDRangeKernel(queue_, kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }
}

void registerOpenCLBackendCreator() {
    InsertBackendCreatorMap(MLLM_OPENCL, std::make_shared<OpenCLBackendCreator>());
}


std::vector<Tensor> OpenCLBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    throw std::runtime_error("runLayer not implemented for OpenCLBackend");
    return {};
}

std::vector<Tensor> OpenCLBackend::runOp(Op *op, std::vector<Tensor> inputs, std::vector<std::string> out_names, bool in_place) {
    Module *module = inputs[0].module();
    // map<string, shared_ptr<Tensor>> &actime] = std::make_shared<Tensor>(op->backend());
    //             activation_tensors[out_name]->setName(out_name);
    //             activation_tensors[out_name]->setModule(module);
    //         }vation_tensors = module->activation_tensors;
    // if (module->doTrace) { // trace
    //     for (const auto &out_name : out_names) {
    //         if (activation_tensors.find(out_name) == activation_tensors.end()) {
    //             activation_tensors[out_na
    //     }
    //     vector<shared_ptr<Tensor>> inPtrs;
    //     for (auto &input : inputs) {
    //         inPtrs.push_back(input.shouldInGraphs() ? activation_tensors[input.name()] :
    //                                                   std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
    //     }
    //     vector<shared_ptr<Tensor>> outPtrs = {};
    //     for (auto &name : out_names) outPtrs.push_back(activation_tensors[name]);
    //     op->setUp(inPtrs, outPtrs);
    //     vector<Tensor> results = {};
    //     for (auto &name : out_names) results.push_back(*activation_tensors[name]);
    //     return results;
    // }

#ifdef DEBUGOPTIME
    uint64_t time_start = mllm_time_us();
#endif
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
    }
    vector<shared_ptr<Tensor>> out_tensors;
    // Part 1: Create tensor shells
    for (const auto &out_name : out_names) {
        auto out_tensor = std::make_shared<Tensor>(op->backend());
        out_tensor->setName(out_name);
        // out_tensor->setModule(module);
        out_tensors.push_back(out_tensor);
    }
    // if (!in_place) {
    //     _create_output_tensors(out_tensors, input_tensors, out_names, module, activation_tensors, op->backend());
    // } else {
    //     // If in-place, we already have out_tensors filled with input tensors.
    //     for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
    //         input_tensors[i]->setName(out_names[i]);
    //         out_tensors.push_back(input_tensors[i]);
    //     }
    // }
    // Part 2: Reshape the tensors
    op->reshape(input_tensors, out_tensors);
    // Part 3: Allocate memory
    op->setUp(input_tensors, out_tensors);
    // if (!in_place) {
    //     for (auto &out_tensor : out_tensors) {
    //         auto act_it = activation_tensors.find(out_tensor->name());
    //         auto template_it = act_it != activation_tensors.end() ? act_it->second : nullptr;
    //         out_tensor->allocFromTemplate(template_it);
    //     }
    // }
    // Part 4: Execute the operation
    op->execute(input_tensors, out_tensors);

#ifdef DEBUGOPTIME
    uint64_t time_end = mllm_time_us();
    double inference_time_ = (time_end - time_start) / 1000.0F; // ms
    std::cout << layer->op_->name() << " | time: " << inference_time_ << "ms" << std::endl;
#endif

    vector<Tensor> results;
    for (const auto &out_tensor : out_tensors) { results.push_back(*out_tensor); }
    return results;
}

std::vector<Tensor> OpenCLBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    throw std::runtime_error("runForward not implemented for OpenCLBackend");
    return {};
}


} // namespace mllm
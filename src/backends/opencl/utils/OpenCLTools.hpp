#ifndef OPENCL_TOOLS_HPP
#define OPENCL_TOOLS_HPP

#include "Tensor.hpp"
#include "../OpenCLBackend.hpp"
#include <vector>

namespace mllm {

std::string inline get_kernel_path(const std::string &current_file, const std::string &relative_kernel_path) {
    // 将源文件路径转换为 filesystem::path 对象
    std::filesystem::path source_path(current_file);
    // 获取源文件所在的目录
    std::filesystem::path source_dir = source_path.parent_path();
    // 组合目录和相对内核路径，生成绝对路径
    std::filesystem::path kernel_path = source_dir / relative_kernel_path;
    // 返回字符串格式的路径
    return kernel_path.string();
}

/**
 * @brief 从一个已在设备上的Tensor获取一个可用于内核计算的Image2D句柄。
 * 该函数是算子内部使用的核心工具。
 *
 * 工作流程:
 * 1. 检查输入Tensor是否已经是Image2D类型，如果是，直接返回其句柄。
 * 2. 如果输入Tensor是Buffer类型，则在设备上创建一个临时的Image2D对象。
 * 3. 执行一次设备内的Buffer-to-Image内存拷贝。
 * 4. 返回新创建的临时Image2D的句柄。
 *
 * @param input_tensor 一个指向已在OpenCL设备上的Tensor的共享指针。
 * @param ocl_backend OpenCL后端实例，用于执行OpenCL命令。
 * @param temp_storage 一个Tensor的vector引用，用于存储函数内部创建的临时Image Tensor，
 * 以确保其生命周期至少持续到内核执行完毕。调用者必须管理此vector的生命周期。
 * @return 一个可用于 clSetKernelArg 的 cl_mem 句柄（指向一个Image2D对象）。
 */
static inline cl_mem get_image_from_tensor(
    const std::shared_ptr<Tensor> &input_tensor,
    OpenCLBackend *ocl_backend,
    std::vector<Tensor> &temp_storage) {
    auto &dev_mem = input_tensor->device_memory();

    if (dev_mem.type == MEM_TYPE_IMAGE_2D) {
        return ocl_backend->get_cl_mem(*input_tensor);
    }
    if (dev_mem.type != MEM_TYPE_BUFFER) {
        throw std::runtime_error("Input must be a Buffer or Image type on device.");
    }

    // ================== 零拷贝路径 ==================
    // 条件：硬件支持扩展，且输入Buffer是带有正确行间距信息创建的。
    if (ocl_backend->is_image_from_buffer_supported() && dev_mem.image_row_pitch_in_bytes > 0) {
        cl_image_format format = {CL_RGBA, CL_FLOAT};
        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = dev_mem.image_width; // 使用创建时保存的元数据
        desc.image_height = dev_mem.image_height;
        desc.image_row_pitch = dev_mem.image_row_pitch_in_bytes;
        desc.buffer = ocl_backend->get_cl_mem(*input_tensor);

        cl_int err;
        cl_mem image_view = clCreateImage(ocl_backend->getContext(), CL_MEM_READ_ONLY, &format, &desc, nullptr, &err);
        check_cl_error(err, "clCreateImage from buffer (Zero-Copy)");

        Tensor wrapper_tensor(ocl_backend);
        wrapper_tensor.device_memory().handle = image_view;
        wrapper_tensor.device_memory().type = MEM_TYPE_IMAGE_2D;
        temp_storage.push_back(std::move(wrapper_tensor));

        return image_view;
    }

    // ====================================================================================
    // 最终推荐的实现：直接、高效的内存拷贝路径
    //
    // 这是在无法从源头控制Buffer创建方式时，最简单、最高效的解决方案。
    // ====================================================================================
    {
        Tensor temp_image(input_tensor->batch(), input_tensor->head(), input_tensor->sequence(), input_tensor->dimension(), ocl_backend, false);
        auto &img_mem = temp_image.device_memory();
        img_mem.type = MEM_TYPE_IMAGE_2D;
        img_mem.image_width = input_tensor->dimension() / 4;
        img_mem.image_height = input_tensor->batch() * input_tensor->head() * input_tensor->sequence();
        temp_image.alloc();

        cl_mem src_buffer = ocl_backend->get_cl_mem(*input_tensor);
        cl_mem dst_image = ocl_backend->get_cl_mem(temp_image);

        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {img_mem.image_width, img_mem.image_height, 1};
        cl_int err = clEnqueueCopyBufferToImage(
            ocl_backend->getQueue(),
            src_buffer, dst_image,
            0, origin, region,
            0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueCopyBufferToImage (Fallback Copy)");

        temp_storage.push_back(std::move(temp_image));
        return dst_image;
    }
}

/**
 * @brief  将一个Tensor的设备内存从Buffer类型原地转换为Image2D类型。
 *
 * 该函数直接修改传入Tensor的内部状态。它会分配一个新的Image2D内存，
 * 将原始Buffer的数据拷贝过去，然后释放原始的Buffer，最后更新Tensor的内存类型信息。
 * 如果已经是Image2D，则不执行任何操作。
 *
 * @param tensor 要转换的Tensor的引用。Tensor必须在OpenCL设备上。
 */
static inline void tensorGlobal2Image(Tensor &tensor) {
    auto ocl_backend = dynamic_cast<OpenCLBackend *>(tensor.backend());
    if (!ocl_backend) {
        throw std::runtime_error("Tensor backend is not OpenCLBackend for tensorGlobal2Image.");
    }

    auto &dev_mem = tensor.device_memory();

    // 如果已经是Image2D，则无需转换
    if (dev_mem.type == MEM_TYPE_IMAGE_2D) {
        return;
    }

    if (dev_mem.type != MEM_TYPE_BUFFER || dev_mem.handle == nullptr) {
        throw std::runtime_error("tensorGlobal2Image requires a valid Buffer on the device.");
    }

    if (tensor.dimension() % 4 != 0) {
        throw std::runtime_error("Image2D conversion requires the dimension to be a multiple of 4.");
    }

    // 1. 创建一个新的Image2D内存对象
    cl_image_format format = {CL_RGBA};
    format.image_channel_data_type = (tensor.dtype() == MLLM_TYPE_F32) ? CL_FLOAT : CL_HALF_FLOAT;

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = tensor.dimension() / 4;
    desc.image_height = tensor.batch() * tensor.head() * tensor.sequence();

    cl_int err;
    cl_mem new_image_handle = clCreateImage(ocl_backend->getContext(), CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
    check_cl_error(err, "clCreateImage in tensorGlobal2Image");

    // 2. 将数据从旧的Buffer拷贝到新的Image
    cl_mem src_buffer_handle = static_cast<cl_mem>(dev_mem.handle);
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {desc.image_width, desc.image_height, 1};
    err = clEnqueueCopyBufferToImage(
        ocl_backend->getQueue(),
        src_buffer_handle, new_image_handle,
        0, origin, region,
        0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueCopyBufferToImage in tensorGlobal2Image");

    // 3. 释放旧的Buffer内存
    clReleaseMemObject(src_buffer_handle);

    // 4. 更新Tensor的内部状态
    dev_mem.handle = new_image_handle;
    dev_mem.type = MEM_TYPE_IMAGE_2D;
    dev_mem.image_width = desc.image_width;
    dev_mem.image_height = desc.image_height;
}

/**
 * @brief  将一个Tensor的设备内存从Image2D类型原地转换为Buffer类型。
 *
 * 该函数直接修改传入Tensor的内部状态。它会分配一个新的Buffer内存，
 * 将原始Image2D的数据拷贝过去，然后释放原始的Image，最后更新Tensor的内存类型信息。
 * 如果已经是Buffer，则不执行任何操作。
 *
 * @param tensor 要转换的Tensor的引用。Tensor必须在OpenCL设备上。
 */
static inline void tensorImage2Global(Tensor &tensor) {
    auto ocl_backend = dynamic_cast<OpenCLBackend *>(tensor.backend());
    if (!ocl_backend) {
        throw std::runtime_error("Tensor backend is not OpenCLBackend for tensorImage2Global.");
    }

    auto &dev_mem = tensor.device_memory();

    // 如果已经是Buffer，则无需转换
    if (dev_mem.type == MEM_TYPE_BUFFER) {
        return;
    }

    if (dev_mem.type != MEM_TYPE_IMAGE_2D || dev_mem.handle == nullptr) {
        throw std::runtime_error("tensorImage2Global requires a valid Image2D on the device.");
    }

    // 1. 创建一个新的Buffer内存对象
    size_t buffer_size = tensor.count() * tensor.dtypeSize();
    cl_int err;
    cl_mem new_buffer_handle = clCreateBuffer(ocl_backend->getContext(), CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    check_cl_error(err, "clCreateBuffer in tensorImage2Global");

    // 2. 将数据从旧的Image拷贝到新的Buffer
    cl_mem src_image_handle = static_cast<cl_mem>(dev_mem.handle);
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {dev_mem.image_width, dev_mem.image_height, 1};
    err = clEnqueueCopyImageToBuffer(
        ocl_backend->getQueue(),
        src_image_handle, new_buffer_handle,
        origin, region, 0,
        0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueCopyImageToBuffer in tensorImage2Global");

    // 3. 释放旧的Image内存
    clReleaseMemObject(src_image_handle);

    // 4. 更新Tensor的内部状态
    dev_mem.handle = new_buffer_handle;
    dev_mem.type = MEM_TYPE_BUFFER;
    // 清理Image相关的元数据
    dev_mem.image_width = 0;
    dev_mem.image_height = 0;
    dev_mem.image_row_pitch_in_bytes = 0;
}

} // namespace mllm

#endif // OPENCL_TOOLS_HPP
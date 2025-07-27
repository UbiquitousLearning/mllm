#ifndef MLLM_TENSORIMPL_H
#define MLLM_TENSORIMPL_H
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
// #include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <assert.h>

#include "OpDefined.hpp"
#include "Backend.hpp"
#include <Types.hpp>

namespace mllm {
class Backend;
class Module;

enum DeviceMemType {
    MEM_TYPE_GENERIC,  // 通用设备指针 (可用于 CUDA 的 `cudaMalloc` 结果)
    MEM_TYPE_BUFFER,   // OpenCL 缓冲区 (cl_buffer)
    MEM_TYPE_IMAGE_2D, // OpenCL 2D图像 (cl_image)
    MEM_TYPE_IMAGE_3D, // OpenCL 3D图像 (cl_image)
    MEM_TYPE_TEXTURE,  // 没用
};

// 通用设备内存描述符结构体
struct DeviceMemory {
    void *handle = nullptr;               // 通用句柄 (存放 cl_mem, cuda pointer, etc.)
    DeviceMemType type = MEM_TYPE_BUFFER; // 内存类型，默认为 Buffer

    // 后端无关的元数据
    size_t size_in_bytes = 0;

    // 专门为 Image 类型准备的元数据
    size_t image_width = 0;
    size_t image_height = 0;
    size_t image_depth = 0; // 用于 3D 图像

    size_t image_row_pitch_in_bytes = 0;
};

class TensorImpl {
public:
    void *host_ptr_ = nullptr;
    bool owns_host_ptr_ = true;
    bool owns_device_memory_ = true;
    std::shared_ptr<void> memory_handle_ = nullptr;

    //=====GPU======
    enum Location {
        UNSPECIFIED, // 未指定位置
        ON_HOST,     // 在主机 (CPU) 内存中
        ON_DEVICE    // 在设备 (如 OpenCL GPU) 内存中
    };
    Location location_ = ON_HOST; // 默认位置设为 ON_HOST，
    DeviceMemory device_memory_;
    //=====GPU======

    std::map<Chl, int> chls_ = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
    string name_;
    DataType dtype_ = MLLM_TYPE_F32;
    ChlType ctype_ = BSHD;

    Backend *backend_ = nullptr;

    vector<uint64_t> shape_;
    uint64_t capacity_ = 0;
    uint64_t count_ = 0;
    uint64_t allocated_ = 0;

    bool transed_ = false;
    bool should_in_graphs_ = true;

    bool undiffusion_ = false;
    vector<std::pair<Chl, Chl>> trans_from_;

    Module *module_ = nullptr;

    // 构造函数
    TensorImpl() = default;
    TensorImpl(Backend *bn) :
        backend_(bn) {
        if (backend_->type() == MLLM_OPENCL) {
            location_ = ON_DEVICE;
        }
    }

    // 析构函数负责资源释放
    ~TensorImpl() {
        free_memory();
    }

    // 禁止拷贝（使用shared_ptr管理）
    TensorImpl(const TensorImpl &) = delete;
    TensorImpl &operator=(const TensorImpl &) = delete;

    size_t cntSize() {
        return DataTypeSize(dtype_, count_);
    }
    void alloc() {
        if (backend_->type() == MLLM_OPENCL) {
            location_ = ON_DEVICE;
        }
        if (location_ == ON_DEVICE) {
            if (device_memory_.handle != nullptr) return;
            if (host_ptr_ != nullptr && owns_host_ptr_) {
                backend_->free(host_ptr_);
                host_ptr_ = nullptr;
                allocated_ = 0;
            }
            device_memory_.size_in_bytes = cntSize();
            backend_->alloc_device(device_memory_, dtype_);
            // owns_device_memory_ = true;
            allocated_ = count_;
            return;
        }
        if (allocated_ != count_) {
            if (host_ptr_ != nullptr && owns_host_ptr_) {
                backend_->free(host_ptr_);
                host_ptr_ = nullptr;
            }
            if (count_ > 0) {
                backend_->alloc(&host_ptr_, cntSize() + 16, 128);
            }
            allocated_ = count_;
        }
    }

    void free_memory() {
        if (location_ == ON_HOST && host_ptr_ != nullptr && owns_host_ptr_) {
            if (backend_) backend_->free(host_ptr_);
        } else if (location_ == ON_DEVICE && device_memory_.handle != nullptr && owns_device_memory_) { //
            if (backend_) backend_->free_device(device_memory_);
            device_memory_.handle = nullptr; // 清理句柄
            // owns_device_memory_ = false;
        }
        host_ptr_ = nullptr;
        allocated_ = 0;
    }

    void unload() {
        memory_handle_.reset();
        if (owns_host_ptr_ && host_ptr_ != nullptr) {
            free_memory();
        }
        host_ptr_ = nullptr;
        allocated_ = 0;
        owns_host_ptr_ = true;
    }

    // 保留旧的 free() 接口，但让它调用新的 free_memory()
    void free() {
        free_memory();
    }

    // void free() {
    //     if (host_ptr_ != nullptr && owns_host_ptr_) { // 直接访问成员变量
    //         if (backend_) {
    //             backend_->free(host_ptr_);
    //         }
    //         host_ptr_ = nullptr;
    //         allocated_ = 0;
    //     }
    // }

    void to(Backend *target_backend) {
        if (backend_ == target_backend) {
            return;
        }
        // 路径1: 从任何后端迁移到主机 (CPU)
        if (target_backend->type() == MLLM_CPU) {
            if (location_ == ON_DEVICE) { // 从设备迁移到Host
                void *new_host_ptr = nullptr;
                target_backend->alloc(&new_host_ptr, cntSize() + 16, 128);
                backend_->copy_to_host(new_host_ptr, device_memory_);
                backend_->free_device(device_memory_);
                host_ptr_ = new_host_ptr;
                // cl_device_buffer_ = nullptr;
                device_memory_.handle = nullptr;
                location_ = ON_HOST;
                allocated_ = count_;
            }
        }
        // 路径2: 从主机 (CPU) 迁移到某个设备
        else if (backend_->type() == MLLM_CPU) {
            if (location_ == ON_HOST) {
                device_memory_.size_in_bytes = cntSize();
                target_backend->alloc_device(device_memory_, dtype_);
                target_backend->copy_from_host(device_memory_, host_ptr_);
                if (owns_host_ptr_) {
                    backend_->free(host_ptr_);
                }
                host_ptr_ = nullptr;
                location_ = ON_DEVICE;
                // allocated_ = 0;// todo1418
            }
        } else {
            std::cout << "Device -> Device migration via Host" << std::endl;
            this->to(Backend::global_backends[MLLM_CPU].get());
            this->to(target_backend);
        }
        backend_ = target_backend;
    }

    //
    int canonicalAxisIndex(int axis_index) const {
        if (axis_index < 0) {
            return axis_index + shape_.size();
        }
        return axis_index;
    }
    int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
    }

    int numAxes() const {
        return shape_.size();
    }

    int legacyShape(int index) const {
        if (index >= numAxes() || index < -numAxes()) {
            return 0;
        }
        return shape(index);
    }

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
    int channel() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[CHANNLE]);
    }
    int time() {
        assert(shape_.size() == 5);
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
        assert(shape_.size() == 5);
        return legacyShape(chls()[HEIGHT]);
    }
    int width() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[WIDTH]);
    }

    bool private_reshape(const vector<int> &shape) {
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

    bool reshape(const int batch, const int head, const int sequence, const int dimension) {
        vector<int> shape(4);
        shape[chls()[BATCH]] = batch;
        shape[chls()[HEAD]] = head;
        shape[chls()[SEQUENCE]] = sequence;
        shape[chls()[DIMENSION]] = dimension;
        return private_reshape(shape);
    }

    void changeCtype(int size = 0) {
        if (!shape_.empty()) {
            size = shape_.size();
        }
        // BCTHW = 3,
        // BTHWC = 4,
        // BWCTH = 5,
        if (size == 5 || (ctype_ >= 3 && ctype_ <= 5)) {
            vector<int> a = {chls()[BATCH], chls()[TIME], chls()[HEIGHT], chls()[WIDTH], chls()[CHANNLE]};
            ctype_ = Chls2Type[a];
        } else {
            vector<int> a = {chls()[BATCH], chls()[HEAD], chls()[SEQUENCE], chls()[DIMENSION]};
            ctype_ = Chls2Type[a];
        }
    }
};

} // namespace mllm

#endif // MLLM_TENSORIMPL_H
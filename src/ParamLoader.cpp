#include "ParamLoader.hpp"
#include "Types.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <tuple>
#include <utility>
// TODO:
// #define USE_MMAP
/*
 * ┌───────┬──────┬───────┬────────┬───────────┬─────────┬─────────┬──────┬──────────────────────┬─────────────────────────┐
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │Index │       │        │           │         │         │      │                      │                         │
 * │       │ Len  │       │        │           │         │         │      │                      │                         │
 * │ Magic │ INT  │ Name  │Name    │ Weights   │ Offset  │ DataType│....  │   Weights Contents   │   Weights Contents      │
 * │       │      │ Length│String  │ Length    │  INT    │  INT    │      │                      │                         │
 * │       │      │ INT   │        │  INT      │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * └───────┴──────┴───────┴────────┴───────────┴─────────┴─────────┴──────┴──────────────────────┴─────────────────────────┘
 * Weights File Structure
 */
namespace mllm {

/*
bool ParamLoader::load(mllm::Tensor *tensor) {
    string name = tensor->name();
    if (!use_mmap_) {
        std::lock_guard<std::mutex> lock(mtx);
        if (offsets_.find(name) == offsets_.end()) { return false; }
        std::pair<uint64_t, uint64_t> offset = offsets_[name];
        auto *p = tensor->hostPtr<char>();
        fseek(fp_, offset.first, SEEK_SET);
        size_t read_size = std::min(tensor->cntSize(), static_cast<size_t>(offset.second));
        auto _ = fread(p, sizeof(uint8_t), read_size, fp_);

        return true;
    } else { // USE_MMAP is defined
         if (mmap_buffer_ == nullptr || offsets_.find(name) == offsets_.end()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx);
        auto offset_info = offsets_[name];

        // ---  在这里加入对齐诊断代码 ---
        int required_alignment = DataTypeSize(tensor->dtype()); // 获取数据类型的大小，如 float 是 4
        if (required_alignment == 0) required_alignment = 1; // 避免除零

        bool is_aligned = (offset_info.first % required_alignment == 0);

        if (!is_aligned) {
            fprintf(stderr, "[ALIGNMENT ERROR] Tensor: '%s', DataType: %d, Offset: %llu, Required Alignment: %d. DATA IS MISALIGNED!\n",
                    name.c_str(),
                    tensor->dtype(),
                    (unsigned long long)offset_info.first,
                    required_alignment);
        } else {
            // (可选) 打印出对齐正确的信息，用于确认
            // fprintf(stdout, "[ALIGNMENT OK]    Tensor: '%s', Offset: %llu, Alignment: %d.\n",
            //         name.c_str(), (unsigned long long)offset_info.first, required_alignment);
        }
        // --- 诊断代码结束 ---

        // 如果不对齐，直接返回失败，因为我们不允许拷贝
        if (!is_aligned) {
            return false;
        }

        // 只有在对齐检查通过后，才执行零拷贝的指针赋值
        if (tensor->cntSize() != offset_info.second) { return false; }
        uint8_t* source_ptr = mmap_buffer_.get() + offset_info.first;
        tensor->setHostPtr(source_ptr, mmap_buffer_);

        return true;
    }
}
    */
// 在 ParamLoader.cpp 中
bool ParamLoader::load(mllm::Tensor *tensor) {
    string name = tensor->name();
    if (!use_mmap_) {
        // --- 非 mmap 模式，逻辑保持不变 ---
        // 确保 Tensor 已经分配内存
        // if (tensor->rawHostPtr() == nullptr) {
            // tensor->alloc();
        // }
        std::lock_guard<std::mutex> lock(mtx);
        if (offsets_.find(name) == offsets_.end()) { return false; }
        std::pair<uint64_t, uint64_t> offset = offsets_[name];
        auto *p = tensor->hostPtr<char>();
        fseek(fp_, offset.first, SEEK_SET);
        size_t read_size = std::min(tensor->cntSize(), static_cast<size_t>(offset.second));
        auto _ = fread(p, sizeof(uint8_t), read_size, fp_);
        return true;
    } else {
        // --- mmap 模式，实现智能选择 ---
        if (mmap_buffer_ == nullptr || offsets_.find(name) == offsets_.end()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx);
        auto offset_info = offsets_[name];

        // 1. 检查对齐
        int required_alignment = DataTypeSize(tensor->dtype());
        if (required_alignment == 0) required_alignment = 1;
        bool is_aligned = (offset_info.first % required_alignment == 0);

        // 2. 尺寸检查
        if (tensor->cntSize() != offset_info.second) {
            fprintf(stderr, "Error: Tensor '%s' size mismatch. Code wants %zu, file has %llu.\n",
                    name.c_str(), tensor->cntSize(), (unsigned long long)offset_info.second);
            return false;
        }
        is_aligned = false;
        if (is_aligned) {
            // -- 对齐：执行零拷贝 --
            uint8_t* source_ptr = mmap_buffer_.get() + offset_info.first;
            // setHostPtr 会处理好一切（包括释放之前 alloc 的内存）
            tensor->setHostPtr(source_ptr, mmap_buffer_);
        } else {
            // -- 未对齐：回退到 memcpy --
            // 从 mmap 区域拷贝数据到 Tensor 自己拥有的内存中
            uint8_t* source_ptr = mmap_buffer_.get() + offset_info.first;
            memcpy(tensor->rawHostPtr(), source_ptr, tensor->cntSize());
        }

        return true;
    }
}
ParamLoader::~ParamLoader() {
    if (use_mmap_) {
        if (use_mmap_ && buffer_ != nullptr && buffer_ != MAP_FAILED) {
            munmap(buffer_, size_);
            buffer_ = nullptr;
        }
    } else {
        if (fp_ != nullptr) {
            fclose(fp_);
            fp_ = nullptr;
        }
    }
}
// #ifdef ANDROID_API
// ParamLoader::ParamLoader(std::string filename, AAssetManager *asset_manager,
// bool use_mmap ):asset_manager_(asset_manager), #else
ParamLoader::ParamLoader(std::string filename, bool use_mmap_param) :                                 // Renamed parameter
    path_(std::move(filename)), use_mmap_(use_mmap_param), fp_(nullptr), buffer_(nullptr), size_(0) { // Initialize new members

    if (use_mmap_) {
                // --- 1. 打开文件并获取文件描述符 ---
        FILE *temp_fp = fopen(this->path_.c_str(), "rb");
        if (temp_fp == nullptr) {
            // perror(("Error opening file for mmap: " + this->path_).c_str());
            // 无法打开文件，直接返回，保持 ParamLoader 为不可用状态
            return;
        }

        // --- 2. 获取文件大小 ---
        fseek(temp_fp, 0, SEEK_END);
        size_ = ftell(temp_fp);
        // fseek(temp_fp, 0, SEEK_SET); // mmap不需要重置文件指针

        // --- 3. 执行内存映射 ---
        int fd = fileno(temp_fp); // 获取文件描述符
        uint8_t* mapped_ptr = (uint8_t *)mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd, 0);

        // mmap完成后，文件描述符即可关闭，映射关系依然存在
        fclose(temp_fp);

        if (mapped_ptr == MAP_FAILED) {
            perror("mmap failed");
            use_mmap_ = false; // mmap失败，可以考虑回退到非mmap模式或标记为失败
            return;
        }

        // --- 4. 将 mmap 指针包装到带自定义删除器的 shared_ptr 中 ---
        auto mmap_size = this->size_;
        this->mmap_buffer_ = std::shared_ptr<uint8_t>(mapped_ptr, [mmap_size](uint8_t *p) {
            // 自定义删除器：当 shared_ptr 引用计数归零时，调用 munmap
            munmap(p, mmap_size);
        });

        // ====================================================================
        // --- 5. 从 mmap 内存区域中解析元数据 ---
        // ====================================================================
        
        // 定义一系列在内存指针上操作的辅助 lambda 函数，用于替代原先在 FILE* 上的操作
        auto mmap_readInt = [&](uint8_t *&ptr) {
            int32_t val;
            memcpy(&val, ptr, sizeof(int32_t));
            ptr += sizeof(int32_t); // 移动指针
            return val;
        };
        auto mmap_readu64 = [&](uint8_t *&ptr) {
            uint64_t val;
            memcpy(&val, ptr, sizeof(uint64_t));
            ptr += sizeof(uint64_t); // 移动指针
            return val;
        };
        auto mmap_readString = [&](uint8_t *&ptr) {
            int len = mmap_readInt(ptr);
            if (len == 0) return std::string("");
            std::string str(reinterpret_cast<const char *>(ptr), len);
            ptr += len; // 移动指针
            return str;
        };

        // 获取指向 mmap 区域开头的当前指针
        uint8_t *current_ptr = mmap_buffer_.get();
        
        // a. 读取并验证幻数
        int magic = mmap_readInt(current_ptr);
        if (magic != _MAGIC_NUMBER) {
            fprintf(stderr, "Mmap: magic number error\n");
            this->mmap_buffer_.reset(); // 释放 shared_ptr，触发 munmap
            use_mmap_ = false;
            return;
        }

        // b. 读取索引区域的总长度
        uint64_t index_size = mmap_readu64(current_ptr);
        
        // c. 计算索引区域的结束地址
        uint8_t *index_end_ptr = current_ptr + index_size;

        // d. 循环读取所有张量的元信息，直到遍历完整个索引区域
        while (current_ptr < index_end_ptr) {
            std::string name = mmap_readString(current_ptr);
            uint64_t length = mmap_readu64(current_ptr);
            uint64_t offset_in_file = mmap_readu64(current_ptr);
            
            // 将解析出的信息存入 map
            offsets_[name] = std::make_pair(offset_in_file, length);
            data_type_[name] = static_cast<DataType>(mmap_readInt(current_ptr));
        }

    } else { // USE_MMAP is NOT defined
        // Original logic when USE_MMAP is not defined (ensures use_mmap_param is ignored)
        use_mmap_ = false; // Force false if USE_MMAP macro is not defined
        this->fp_ = fopen(this->path_.c_str(), "rb");
        if (this->fp_ == nullptr) {
            // perror(("Error opening file: " + this->path_).c_str());
            // exit(1);
            return;
        }
        fseek(fp_, 0, SEEK_SET);
        int magic = readInt(fp_);
        if (magic != _MAGIC_NUMBER) {
            fprintf(stderr, "File: magic number error\n");
            fclose(fp_);
            fp_ = nullptr;
            // exit(1);
            return;
        }
        uint64_t index_size = readu64(fp_);
        uint64_t index_offset = index_size + ftell(fp_);
        while (ftell(fp_) < index_offset) {
            std::string name = readString(fp_);
            uint64_t length = readu64(fp_);
            uint64_t offset = readu64(fp_);
            offsets_[name] = std::make_pair(offset, length);
            data_type_[name] = readInt(fp_);
        }
    }
}
bool ParamLoader::load(std::shared_ptr<mllm::Tensor> tensor) {
    return load(tensor.get());
}
vector<std::string> ParamLoader::getParamNames() {
    // get keys of data_type_
    vector<std::string> keys;
    keys.reserve(data_type_.size());
    for (auto &[fst, snd] : data_type_) {
        keys.push_back(fst);
    }
    return keys;
}
std::tuple<uint8_t *, uint64_t> ParamLoader::load(string name) {
    auto [offset, length] = offsets_[name];
    auto *data = new uint8_t[length];
    fseek(fp_, offset, SEEK_SET);
    auto _ = fread(data, sizeof(uint8_t), length, fp_);
    return std::make_tuple(data, length);
}
DataType ParamLoader::getDataType(string name) {
    if (data_type_.count(name) != 1) {
        if (!this->path_.empty() && this->fp_ == nullptr) {
            MLLM_LOG_ERROR_STREAM << this->path_ << " not found" << std::endl;
            exit(0);
        } else if (this->fp_ != nullptr && !this->path_.empty()) {
            MLLM_LOG_ERROR_STREAM << name << " not found" << std::endl;
        }
        return DataType::MLLM_TYPE_COUNT;
    }
    int type = data_type_[name];
    // check if exists
    return static_cast<DataType>(type);
}

MultiFileParamLoader::MultiFileParamLoader(const std::initializer_list<std::string> &filenames) {
    for (const auto &filename : filenames) {
        load_file(filename);
    }
}

bool MultiFileParamLoader::load(mllm::Tensor *tensor) {
    string name = tensor->name();
    auto it = files_.find(name);
    if (it == files_.end())
        return false;
    auto fp = it->second;
    auto [offset, size] = offsets_[name];
    void *p = tensor->rawHostPtr();
    fseek(fp, (long)offset, SEEK_SET);
    auto read_size = fread(p, sizeof(uint8_t), size, fp);
    assert(read_size == size);
    auto tensor_size = tensor->cntSize();
    //    tensor->printShape();
    assert(tensor_size == size);
    return true;
}

bool MultiFileParamLoader::load(std::shared_ptr<mllm::Tensor> tensor) {
    return load(tensor.get());
}

size_t MultiFileParamLoader::getTensorSize(string name) {
    auto it = files_.find(name);
    if (it == files_.end())
        throw std::runtime_error("name: '" + name + "'not found");
    auto t = offsets_[name];
    return t.second;
}

DataType MultiFileParamLoader::getDataType(string name) {
    auto it = data_type_.find(name);
    if (it == data_type_.end())
        throw std::runtime_error("name: '" + name + "' not found, can not get data type");
    return data_type_[name];
}

void MultiFileParamLoader::load_file(const string &filename) {
    auto fp = fopen(filename.c_str(), "rb");

    if (fp == nullptr) {
        throw std::ios_base::failure("Failed to open file: " + filename);
    }

    int magic = readInt(fp);
    if (magic != _MAGIC_NUMBER) {
        throw std::runtime_error("Open file " + filename + "error: Magic number error");
    }

    uint64_t index_size = readu64(fp);
    uint64_t index_end = index_size + ftell(fp);
    while (ftell(fp) < index_end) {
        std::string name = readString(fp);
        uint64_t length = readu64(fp);
        uint64_t offset = readu64(fp);
        auto type = static_cast<DataType>(readInt(fp));
        offsets_[name] = std::make_pair(offset, length);
        data_type_[name] = type;
        files_[name] = fp;
    }
}
MultiFileParamLoader::~MultiFileParamLoader() {
#include <set>
    std::set<FILE *> closed;
    for (const auto &p : files_) {
        if (closed.find(p.second) != closed.end()) {
            fclose(p.second);
            closed.insert(p.second);
        }
    }
}

bool ParamLoader::partialLoad(mllm::Tensor *tensor, std::set<int> validRow, int rowNum, int colNum) {
    string name = tensor->name();
    if (!use_mmap_) {
        std::lock_guard<std::mutex> lock(mtx); // Assuming mtx is still relevant for file ops
        if (offsets_.find(name) == offsets_.end()) { return false; }
        std::pair<uint64_t, uint64_t> offset_info = offsets_[name];
        int perValueLength = offset_info.second / rowNum / colNum; // Make sure this division is safe
        if (rowNum == 0 || colNum == 0) {                          /* handle error, division by zero */
            return false;
        }

        // Temporary buffer to hold all valid rows before copying to tensor
        // This might be inefficient if validRow.size() * colNum * perValueLength is very large
        // Consider copying row by row directly to the tensor if memory is a concern
        // and tensor layout allows for it.
        uint8_t *temp_data_buffer = new uint8_t[perValueLength * validRow.size() * colNum];
        size_t totalBytesCopiedToTempBuffer = 0;

        for (auto row : validRow) {
            fseek(fp_, offset_info.first + (static_cast<uint64_t>(row) * colNum) * perValueLength, SEEK_SET);
            auto bytes_read_this_row = fread(temp_data_buffer + totalBytesCopiedToTempBuffer, sizeof(uint8_t), perValueLength * colNum, fp_);
            if (bytes_read_this_row != static_cast<size_t>(perValueLength * colNum)) {
                // Handle read error or incomplete read
                delete[] temp_data_buffer;
                return false;
            }
            totalBytesCopiedToTempBuffer += bytes_read_this_row;
        }

        auto *p_tensor = tensor->hostPtr<char>();
        memcpy(static_cast<void *>(p_tensor), static_cast<void *>(temp_data_buffer), totalBytesCopiedToTempBuffer);
        delete[] temp_data_buffer;
        return true;
    } else {
        if (!use_mmap_ || buffer_ == nullptr || offsets_.find(name) == offsets_.end()) {
            fprintf(stderr, "Error: mmap not initialized or tensor name not found for mmap partialLoad.\n");
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx); // If concurrent access to tensor or offsets_
        std::pair<uint64_t, uint64_t> offset_info = offsets_[name];
        if (rowNum == 0 || colNum == 0) { /* handle error */
            return false;
        }
        int perValueLength = offset_info.second / rowNum / colNum;

        auto *p_tensor = tensor->hostPtr<char>(); // Get tensor's host pointer
        size_t bytesCopiedToTensor = 0;

        // Assuming tensor is pre-sized to hold exactly the data from validRow
        // and that rows in tensor are contiguous in the order of validRow iteration.
        // If tensor structure is different, this logic needs adjustment.
        for (auto row : validRow) {
            uint8_t *source_ptr_in_mmap = buffer_ + offset_info.first + (static_cast<uint64_t>(row) * colNum) * perValueLength;
            uint8_t *dest_ptr_in_tensor = reinterpret_cast<uint8_t *>(p_tensor) + bytesCopiedToTensor;
            size_t bytes_to_copy_this_row = static_cast<size_t>(perValueLength * colNum);

            memcpy(dest_ptr_in_tensor, source_ptr_in_mmap, bytes_to_copy_this_row);
            bytesCopiedToTensor += bytes_to_copy_this_row;
        }
        // Ensure tensor->cntSize() matches totalBytesCopied. This function assumes tensor is correctly sized.
        return true;
    }
}
} // namespace mllm
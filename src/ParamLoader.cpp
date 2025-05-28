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
bool ParamLoader::load(mllm::Tensor *tensor) {
    string name = tensor->name();
    if (!use_mmap_) {
        std::lock_guard<std::mutex> lock(mtx);
        if (offsets_.find(name) == offsets_.end()) { return false; }
        std::pair<uint64_t, uint64_t> offset = offsets_[name];
        auto *p = tensor->hostPtr<char>();
        fseek(fp_, offset.first, SEEK_SET);
        size_t read_size = std::min(tensor->cntSize(), offset.second);
        auto _ = fread(p, sizeof(uint8_t), read_size, fp_);

        /*
        if (offsets_.find(name) == offsets_.end()) { return false; }
        std::pair<uint64_t, uint64_t> offset = offsets_[name];
        uint8_t *data = new uint8_t[offset.second];
        fseek(fp_, offset.first, SEEK_SET);
        auto _ = fread(data, sizeof(uint8_t), offset.second, fp_);
        // TODO:Data?
        //  tenor. = data;
        auto *p = tensor->hostPtr<char>();

        if (tensor->cntSize() >= offset.second)
            memcpy(static_cast<void *>(p), static_cast<void *>(data),
                   offset.second); // Cast pointers to void*
        else
            memcpy(static_cast<void *>(p), static_cast<void *>(data),
                   tensor->cntSize()); // Cast pointers to void*
        delete[] data;                 // Free the memory allocated by new
        */
        return true;
    } else { // USE_MMAP is defined
        // 确保 buffer_ 和 offsets_ 已经为 mmap 正确初始化
        if (!use_mmap_ || buffer_ == nullptr || offsets_.find(name) == offsets_.end()) {
            // 可以选择打印错误信息或返回 false
            // TODO
            // if (offsets_.find(name) == offsets_.end()) {
            //     fprintf(stderr, "Tensor name '%s' not found in offsets.\n", name.c_str());
            // } else {
            //     fprintf(stderr, "Buffer is null or mmap not initialized.\n");
            // }
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx); // mmap 访问也可能需要同步，取决于使用场景
        std::pair<uint64_t, uint64_t> offset_info = offsets_[name];
        auto *p = tensor->hostPtr<char>(); // 获取 tensor 的主机指针

        // 计算源数据指针
        uint8_t *source_ptr = buffer_ + offset_info.first;

        // 要拷贝的数据大小，取 tensor 大小和参数大小的最小值
        size_t copy_size = std::min(tensor->cntSize(), offset_info.second);

        // 从内存映射的 buffer_ 拷贝数据到 tensor
        memcpy(static_cast<void *>(p), static_cast<const void *>(source_ptr), copy_size);

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
        // 1. 打开文件
        FILE *temp_fp = fopen(this->path_.c_str(), "rb");
        if (temp_fp == nullptr) {
            // perror(("Error opening file for mmap: " + this->path_).c_str());
            // exit(1); // Or handle error differently
            return;
        }

        // 2. 获取文件大小
        fseek(temp_fp, 0, SEEK_END);
        size_ = ftell(temp_fp);
        fseek(temp_fp, 0, SEEK_SET); // Reset to beginning

        // 3. 内存映射 (示例使用 POSIX mmap)
        // #include <sys/mman.h>
        // #include <fcntl.h>
        // #include <unistd.h>
        int fd = fileno(temp_fp); // Get file descriptor
        buffer_ = (uint8_t *)mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (buffer_ == MAP_FAILED) {
            perror("mmap failed");
            fclose(temp_fp);   // Close the temporary file pointer
            buffer_ = nullptr; // Mark buffer as invalid
            use_mmap_ = false; // Fallback or indicate error
            // exit(1); // Or handle error differently
            return;
        }
        // 文件描述符可以关闭了，mmap 会保持映射
        // fclose(temp_fp); // Or keep fp_ as the original file pointer if needed for non-mmap fallback

        // 4. 从 buffer_ 读取元数据 (类似于 readInt, readString 等，但操作指针)
        uint8_t *current_ptr = buffer_;
        auto mmap_readInt = [&](uint8_t *&ptr) {
            int32_t val;
            memcpy(&val, ptr, sizeof(int32_t));
            ptr += sizeof(int32_t);
            return val;
        };
        auto mmap_readu64 = [&](uint8_t *&ptr) {
            uint64_t val;
            memcpy(&val, ptr, sizeof(uint64_t));
            ptr += sizeof(uint64_t);
            return val;
        };
        auto mmap_readString = [&](uint8_t *&ptr) {
            int len = mmap_readInt(ptr);
            std::string str((char *)ptr, len);
            ptr += len;
            return str;
        };

        int magic = mmap_readInt(current_ptr);
        if (magic != _MAGIC_NUMBER) {
            fprintf(stderr, "Mmap: magic number error\n");
            munmap(buffer_, size_); // Unmap memory
            buffer_ = nullptr;
            use_mmap_ = false;
            // exit(1); // Or handle error
            return;
        }

        uint64_t index_size = mmap_readu64(current_ptr);
        uint8_t *index_end_ptr = current_ptr + index_size;

        while (current_ptr < index_end_ptr) {
            std::string name = mmap_readString(current_ptr);
            uint64_t length = mmap_readu64(current_ptr);
            // 对于 mmap，offset 通常是相对于 buffer_ 开始的偏移
            // 如果文件格式中的 offset 是相对于文件数据区的绝对偏移，需要调整
            uint64_t offset_in_file = mmap_readu64(current_ptr);     // This is the offset as stored in the file
            offsets_[name] = std::make_pair(offset_in_file, length); // Store the original offset from file
            data_type_[name] = mmap_readInt(current_ptr);
        }
        // Mmap is set up, fp_ might not be needed or could be temp_fp if kept open
        // If you want to keep fp_ for potential non-mmap operations or cleanup:
        this->fp_ = temp_fp; // Assign after successful mmap, or keep it as NULL
                             // If keeping temp_fp, ensure it's closed in destructor if mmap was used.
                             // Alternatively, just close temp_fp here if all mmap ops are done.
                             // fclose(temp_fp) was already called if mmap failed. If successful, and you don't need fp_ for mmap path:
        // fclose(temp_fp); // Or defer to destructor. For mmap, fd is what mattered.
        // Let's assume we close it here if mmap succeeded and fp_ isn't used by mmap logic itself
        if (buffer_ != MAP_FAILED) { // if mmap succeeded
                                     // fclose(temp_fp); // Decide on fp_ lifecycle. If mmap is primary, fp_ might be set to nullptr
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
#include "ParamLoader.hpp"
#include "Types.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
// TODO:
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
#ifndef USE_MMAP
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
#endif
}
ParamLoader::~ParamLoader() {
    if (fp_ != nullptr) { fclose(fp_); }
}
// #ifdef ANDROID_API
// ParamLoader::ParamLoader(std::string filename, AAssetManager *asset_manager,
// bool use_mmap ):asset_manager_(asset_manager), #else
ParamLoader::ParamLoader(std::string filename, bool use_mmap) :
    // #endif
    path_(std::move(filename)), use_mmap_(use_mmap) {
    // #ifdef ANDROID_API
    //     this->fp_ = AAssetManager_open(asset_manager_, this->path_.c_str(),
    //     AASSET_MODE_RANDOM);
    // #else
    this->fp_ = fopen(this->path_.c_str(), "rb");
    // #endif

    if (this->fp_ == nullptr) {
        // std::cout << "param open file failed" << std::endl;
        return;
        int errorCode = errno;
        char *errorMsg = strerror(errorCode);
        printf("Open file fail, errorCode:%d, errorMsg:%s\n", errorCode,
               errorMsg);
        exit(1);
    }
#ifndef USE_MMAP
    use_mmap_ = false;
#endif
    fseek(fp_, 0, SEEK_SET);
#ifndef USE_MMAP
    int magic = readInt(fp_);
    if (magic != _MAGIC_NUMBER) {
        std::cout << "magic number error" << std::endl;
        exit(1);
    }
    uint64_t index_size = readu64(fp_);
    uint64_t index_offset = index_size + ftell(fp_);
    while (ftell(fp_) < index_offset) {
        std::string name = readString(fp_);
        uint64_t length = readu64(fp_);
        uint64_t offset = readu64(fp_);
        offsets_[name] = std::make_pair(offset, length);
        // std::cout<<name<<"   length:"<<length<<std::endl;
        data_type_[name] = readInt(fp_);
    }
// int len = sizeof(int);
// while (len<size) {
//     int index = readInt(fp_);
//     len+=sizeof(int);
//     std::string name = readString(fp_);
//     int length = readInt(fp_);
//     len+=name.size()+sizeof(int)+sizeof(int);
//     offsets_[name] = std::make_pair(len,length);
//     len+=length; //Align?
// }
#endif
    // std::cout << "load param file success" << std::endl;
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
//        printf("loaded %s\n", name.c_str());
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
#ifndef USE_MMAP
    if (offsets_.find(name) == offsets_.end()) { return false; }
    std::pair<uint64_t, uint64_t> offset = offsets_[name];
    // for data longer then 1 byte
    int perValueLength = offset.second / rowNum / colNum;
    uint8_t *data = new uint8_t[perValueLength * validRow.size() * colNum];
    size_t totalBytesRead = 0;

    // load begin
    for (auto row : validRow) {
        fseek(fp_, offset.first + (row * colNum) * perValueLength, SEEK_SET);
        auto s = fread(data + totalBytesRead, sizeof(uint8_t), perValueLength * colNum, fp_);
        totalBytesRead += perValueLength * colNum;
    }

    auto *p = tensor->hostPtr<char>();
    // Cast pointers to void*
    memcpy(static_cast<void *>(p), static_cast<void *>(data),
           perValueLength * validRow.size() * colNum);

    // Free the memory allocated by new
    delete[] data;
    return true;
#endif
}
} // namespace mllm
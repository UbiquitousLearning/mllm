#include "ParamLoader.hpp"
#include "Types.hpp"
#include <cstdint>
#include <cstdio>
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
    std::cout << "load: " << tensor->name() << std::endl;
    string name = tensor->name();
#ifndef USE_MMAP
    if (offsets_.find(name) == offsets_.end()) { return false; }
    std::pair<uint64_t, uint64_t> offset = offsets_[name];
    uint8_t *data = new uint8_t[offset.second];
    fseek(fp_, offset.first, SEEK_SET);
    fread(data, sizeof(uint8_t), offset.second, fp_);
    // TODO:Data?
    //  tenor. = data;
    auto *p = tensor->hostPtr<char>();
    memcpy(static_cast<void *>(p), static_cast<void *>(data),
           offset.second); // Cast pointers to void*
    delete[] data;         // Free the memory allocated by new
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
        std::cout << "param open file failed" << std::endl;
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
    fread(data, sizeof(uint8_t), length, fp_);
    return std::make_tuple(data, length);
}
DataType ParamLoader::getDataType(string name) {
    if (data_type_.count(name) != 1) {
        std::cerr<<name<<" not found"<<std::endl;
        return DataType::MLLM_TYPE_COUNT;
    }
    int type = data_type_[name];
    // check if exists
    return static_cast<DataType>(type);
}
} // namespace mllm
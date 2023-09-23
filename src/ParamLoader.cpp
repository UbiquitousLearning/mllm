#include "ParamLoader.hpp"
#include "NetParameter.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
// TODO:
/*
 * ┌───────┬──────┬───────┬────────┬───────────┬─────────┬─────────┬──────┬──────────────────────┬─────────────────────────┐
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │ Index│       │        │           │         │         │      │                      │                         │
 * │       │ Len  │       │        │           │         │         │      │                      │                         │
 * │ Magic │ INT  │ Name  │Name    │ Weights   │ Offset  │ DataType│....  │   Weights Contents   │   Weights Content       │
 * │       │      │ Length│String  │ Length    │  INT    │  INT    │      │                      │                         │
 * │       │      │ INT   │        │  INT      │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * └───────┴──────┴───────┴────────┴───────────┴─────────┴─────────┴──────┴──────────────────────┴─────────────────────────┘
 * Weights File Structure
 */

static int readInt(FILE *fp_) {
    int tmp;
    fread(&tmp, sizeof(int), 1, fp_);
    return tmp;
}
static std::string readString(FILE *fp_) {
    int len = readInt(fp_);
    char *tmp = new char[len];
    fread(tmp, sizeof(char), len, fp_);
    std::string str(tmp);
    delete[] tmp;
    return str;
}
namespace mllm {
bool ParamLoader::load(mllm::Tensor *tensor) {
    string name = tensor->name();
#ifndef USE_MMAP
    if (offsets_.find(name) == offsets_.end()) {
        return false;
    }
    std::pair<uint8_t, uint8_t> offset = offsets_[name];
    uint8_t *data = new uint8_t[offset.second];
    fseek(fp_, offset.first, SEEK_SET);
    fread(data, sizeof(uint8_t), offset.second, fp_);
    // TODO:Data?
    //  tenor. = data;
    auto *p = tensor->hostPtr<char>();
    memcpy(static_cast<void *>(p), static_cast<void *>(data), offset.second); // Cast pointers to void*
    delete[] data;                                                            // Free the memory allocated by new
    return true;
#endif
}
ParamLoader::~ParamLoader() {
    if (fp_ != nullptr) {
        fclose(fp_);
    }
}
ParamLoader::ParamLoader(std::string filename, bool use_mmap) :
    path_(std::move(filename)), use_mmap_(use_mmap) {
    this->fp_ = fopen(filename.c_str(), "rb");
    if (fp_ == nullptr) {
        std::cout << "open file failed" << std::endl;
        exit(1);
    }
#ifndef USE_MMAP
    use_mmap_ = false;
#endif
    fseek(fp_, 0, SEEK_SET);
#ifndef USE_MMAP
    int magic = readInt(fp_);
    if (magic != MAGIC_NUMBER) {
        std::cout << "magic number error" << std::endl;
        exit(1);
    }
    int index_size = readInt(fp_);
    int index_offset = index_size + ftell(fp_);
    while (ftell(fp_) < index_offset) {
        std::string name = readString(fp_);
        int length = readInt(fp_);
        int offset = readInt(fp_);
        offsets_[name] = std::make_pair(offset, length);
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
}
} // namespace mllm
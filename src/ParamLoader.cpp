#include "ParamLoader.hpp"
#include "NetParameter.hpp"
// #include <bits/stdint-uintn.h>
#include <cstdio>
#include <string>
#include <utility>
/*
 * ┌───────┬──────────────────────────┬──────────────────────────┬────────┬──────┬─────────┬───────────┬─────────┬─────────┐
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │ Magic │     Weights Contents 1   │      Weights Contents 2  │ .....  │ Name │ Name    │   Weights │ Offsets │ Weights │
 * │       │                          │                          │        │ Length String  │   Length  │   INT   │  Index  │
 * │       │                          │                          │        │ INT  │         │    INT    │         │  Length │
 * │       │                          │                          │        │      │         │           │         │   INT   │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * └───────┴──────────────────────────┴──────────────────────────┴────────┴──────┴─────────┴───────────┴─────────┴─────────┘
 *  Weights File Structure
 */

static int read_int(FILE *fp_) {
    int tmp;
    fread(&tmp, sizeof(int), 1, fp_);
    return tmp;
}
static std::string read_string(FILE *fp_) {
    int len = read_int(fp_);
    char *tmp = new char[len];
    fread(tmp, sizeof(char), len, fp_);
    std::string str(tmp);
    delete[] tmp;
    return str;
}
namespace mllm {
bool ParamLoader::Load(mllm::Tensor *tenor) {
    string name = tenor->Name();
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
    void *tensorPtr = tenor->HostPtr<char>();
    memcpy(tensorPtr, data, offset.second);
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
    int magic = read_int(fp_);
    if (magic != MAGIC_NUMBER) {
        std::cout << "magic number error" << std::endl;
        exit(1);
    }
    fseek(fp_, 0, SEEK_END);
    this->size_ = ftell(fp_);
    fseek(fp_, this->size_ - sizeof(int), SEEK_CUR);
    int table_len = read_int(fp_);
    fseek(fp_, this->size_ - table_len - sizeof(int), SEEK_SET);
    int table_offset = ftell(fp_);
    while (table_offset < this->size_ - sizeof(int)) {
        std::string name = read_string(fp_);
        int length = read_int(fp_);
        int offset = read_int(fp_);
        offsets_[name] = std::make_pair(offset, length);
        // table_offset+=name.size()+sizeof(int)+sizeof(int);
        table_offset = ftell(fp_);
    }

// int len = sizeof(int);
// while (len<size) {
//     int index = read_int(fp_);
//     len+=sizeof(int);
//     std::string name = read_string(fp_);
//     int length = read_int(fp_);
//     len+=name.size()+sizeof(int)+sizeof(int);
//     offsets_[name] = std::make_pair(len,length);
//     len+=length; //Align?
// }
#endif
}
} // namespace mllm
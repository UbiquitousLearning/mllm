#ifndef MLLM_ParamLoader_H
#define MLLM_ParamLoader_H
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "Tensor.hpp"

namespace mllm {
class Tensor;
static int readInt(FILE *fp_) {
    int tmp;
    fread(&tmp, sizeof(int), 1, fp_);
    return tmp;
}
static uint64_t readu64(FILE *fp_) {
    uint64_t tmp;
    fread(&tmp, sizeof(uint64_t), 1, fp_);
    return tmp;
}
static float readf32(FILE *fp_) {
    float tmp;
    fread(&tmp, sizeof(float), 1, fp_);
    return tmp;
}
static double readf64(FILE *fp_) {
    double tmp;
    fread(&tmp, sizeof(double), 1, fp_);
    return tmp;
}
static std::string readString(FILE *fp_) {
    int len = readInt(fp_);
    char *tmp = new char[len+1];
    fread(tmp, sizeof(char), len, fp_);
    tmp[len] = '\0';
    std::string str(tmp);
    if (len == 0) {
        str = "";
    }
    delete[] tmp;
    return str;
}
#define _MAGIC_NUMBER 20012
class ParamLoader {
public:
    ParamLoader(std::string filename, bool use_mmap = false);
#ifdef USE_MMAP
    ParamLoader(void *buffer);
#endif
    ~ParamLoader();
    bool load(mllm::Tensor *tensor);

private:
    FILE *fp_;
    uint8_t *buffer_;
    std::string path_;
    std::uint64_t size_;
    std::map<std::string, std::pair<uint64_t, uint64_t>> offsets_; // offsets,length
    std::map<std::string, int> data_type_;
    bool use_mmap_;
};

} // namespace mllm
#endif

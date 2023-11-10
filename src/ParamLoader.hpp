#ifndef MLLM_ParamLoader_H
#define MLLM_ParamLoader_H
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;
static int readInt(FILE *fp_) {
    int tmp;
    fread(&tmp, sizeof(int32_t), 1, fp_);
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
    char *tmp = new char[len + 1];
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
class AbstructLoader {
public:
    virtual bool load(mllm::Tensor *tensor) = 0;
    virtual bool load(std::shared_ptr<mllm::Tensor> tensor) = 0;
    virtual DataType getDataType(string name) = 0;
};
class ParamLoader : public AbstructLoader {
    friend class QuantWriter;

public:
    ParamLoader(std::string filename, bool use_mmap = false);
#ifdef USE_MMAP
    ParamLoader(void *buffer);
#endif
    ~ParamLoader();
    bool load(mllm::Tensor *tensor) override;
    bool load(std::shared_ptr<mllm::Tensor> tensor) override;
    vector<std::string> getParamNames();
    std::tuple<uint8_t *, uint64_t> load(string name);
    DataType getDataType(string name) override;

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

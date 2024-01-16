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
#define mllm_file FILE
// #ifdef ANDROID_API
// #include <android/asset_manager.h>
// #define fseek AAsset_seek64
// #define fclose AAsset_close
// #define mllm_file AAsset
// #define fread(buffer, size, count, fp) AAsset_read(fp, buffer, size * count)
// #define ftell(fp) AAsset_getLength64(fp) - AAsset_getRemainingLength64(fp)
// #endif

namespace mllm {
class Tensor;
static int readInt(mllm_file *fp_) {
    int tmp;
    fread(&tmp, sizeof(int32_t), 1, fp_);
    return tmp;
}
static uint64_t readu64(mllm_file *fp_) {
    uint64_t tmp;
    fread(&tmp, sizeof(uint64_t), 1, fp_);
    return tmp;
}
static float readf32(mllm_file *fp_) {
    float tmp;
    fread(&tmp, sizeof(float), 1, fp_);
    return tmp;
}
static double readf64(mllm_file *fp_) {
    double tmp;
    fread(&tmp, sizeof(double), 1, fp_);
    return tmp;
}
static std::string readString(mllm_file *fp_) {
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
    virtual DataType getDataType(string name) {return MLLM_TYPE_COUNT;}
    // virtual int length (string name) =0;
};
class ParamLoader : public AbstructLoader {
    friend class QuantWriter;

public:
// #ifdef ANDROID_API
//     ParamLoader(std::string filename, AAssetManager *asset_manager, bool use_mmap = false);
// #else
    ParamLoader(std::string filename, bool use_mmap = false);
// #endif

#ifdef USE_MMAP
    ParamLoader(void *buffer);
#endif
    ~ParamLoader();
    bool load(mllm::Tensor *tensor) override;
    bool load(std::shared_ptr<mllm::Tensor> tensor) override;
    vector<std::string> getParamNames();
    std::tuple<uint8_t *, uint64_t> load(string name);
    DataType getDataType(string name) override;
    bool isAvailible() const {
        return fp_ != nullptr&& !offsets_.empty();
    }
    unsigned int getParamSize() const {
        return offsets_.size();
    }
// #ifdef ANDROID_API
// void setAssetManager(AAssetManager *asset_manager) {
//     asset_manager_ = asset_manager;
// };
    // #endif

    // int length (string name) override {
    //     auto [offset, length] = offsets_[name];
    //     auto type = getDataType(name);
    //     return length/(DataTypeSize(type, 1));
    // }


private:
// #ifdef ANDROID_API
//     AAssetManager *asset_manager_;
// #endif

    mllm_file *fp_;
    uint8_t *buffer_;
    std::string path_;
    std::uint64_t size_;
    std::map<std::string, std::pair<uint64_t, uint64_t>> offsets_; // offsets,length
    std::map<std::string, int> data_type_;
    bool use_mmap_;
};

} // namespace mllm
#endif

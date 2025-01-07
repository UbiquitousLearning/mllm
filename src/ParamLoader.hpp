#ifndef MLLM_ParamLoader_H
#define MLLM_ParamLoader_H
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include "Tensor.hpp"
#include "Types.hpp"
#include <initializer_list>
#include <mutex>
#define mllm_file FILE

namespace mllm {
class Tensor;
static int readInt(mllm_file *fp_) {
    int tmp;
    auto _ = fread(&tmp, sizeof(int32_t), 1, fp_);
    return tmp;
}
static uint64_t readu64(mllm_file *fp_) {
    uint64_t tmp;
    auto _ = fread(&tmp, sizeof(uint64_t), 1, fp_);
    return tmp;
}
static float readf32(mllm_file *fp_) {
    float tmp;
    auto _ = fread(&tmp, sizeof(float), 1, fp_);
    return tmp;
}
static double readf64(mllm_file *fp_) {
    double tmp;
    auto _ = fread(&tmp, sizeof(double), 1, fp_);
    return tmp;
}
static std::string readString(mllm_file *fp_) {
    int len = readInt(fp_);
    char *tmp = new char[len + 1];
    auto _ = fread(tmp, sizeof(char), len, fp_);
    tmp[len] = '\0';
    std::string str(tmp);
    if (len == 0) {
        str = "";
    }
    delete[] tmp;
    return str;
}

#define _MAGIC_NUMBER 20012
/**
 * \brief The AbstructLoader abstract class provides an interface for loading parameters.
 */
class AbstructLoader {
public:
    virtual bool load(mllm::Tensor *tensor) = 0;
    virtual bool load(std::shared_ptr<mllm::Tensor> tensor) = 0;

    virtual size_t getTensorSize(string name) {
        fprintf(stderr, "loader not support getTensorSize");
        return NOT_SUPPORT;
    }
    virtual DataType getDataType(string name) {
        return MLLM_TYPE_COUNT;
    }
    // virtual bool partialLoad(mllm::Tensor *tensor, std::set<int> validRow, int rowNum, int colNum) = 0;
};

/**
 * \brief The ParamLoader class is the default and only(currently) implementation of the AbstructLoader class.
 */
class ParamLoader : public AbstructLoader {
    friend class QuantWriter;

public:
    ParamLoader(std::string filename, bool use_mmap = false);

#ifdef USE_MMAP
    ParamLoader(void *buffer);
#endif
// no param loader for debug
#ifdef DEBUG
    ParamLoader() {
        std::cout << "ParamLoader" << std::endl;
    }
#endif
    ~ParamLoader();
    bool load(mllm::Tensor *tensor) override;
    bool load(std::shared_ptr<mllm::Tensor> tensor) override;
    bool partialLoad(mllm::Tensor *tensor, std::set<int> validRow, int rowNum, int colNum);
    vector<std::string> getParamNames();
    std::tuple<uint8_t *, uint64_t> load(string name);
    DataType getDataType(string name) override;
    bool isAvailible() const {
        return fp_ != nullptr && !offsets_.empty();
    }
    unsigned int getParamSize() const {
        return offsets_.size();
    }

protected:
    std::mutex mtx;
    mllm_file *fp_;
    uint8_t *buffer_;
    std::string path_;
    std::uint64_t size_;
    std::map<std::string, std::pair<uint64_t, uint64_t>> offsets_; // offsets,length
    std::map<std::string, int> data_type_;
    bool use_mmap_;
};

/**
 * \brief The MultiFileParamLoader class is similar to ParamLoader. The difference is that this class can load weights from multiple files.
 */
class MultiFileParamLoader : public AbstructLoader {
    friend class QuantWriter;

public:
    MultiFileParamLoader(const std::initializer_list<string> &filenames);
    ~MultiFileParamLoader();
    bool load(mllm::Tensor *tensor) override;
    bool load(std::shared_ptr<mllm::Tensor> tensor) override;
    size_t getTensorSize(string name) override;
    DataType getDataType(string name) override;

private:
    map<string, mllm_file *> files_; // tensor in which file <tensor_name, fp to file that tensor is in>
    map<string, DataType> data_type_;
    map<string, std::pair<uint64_t, uint64_t>> offsets_; // offsets, datasize

    void load_file(const string &filename);
};

} // namespace mllm
#endif

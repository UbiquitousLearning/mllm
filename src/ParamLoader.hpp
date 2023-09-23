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

#define MAGIC_NUMBER 20012
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
    std::uint8_t size_;
    std::map<std::string, std::pair<uint8_t, uint8_t>> offsets_; // offsets,length
    bool use_mmap_;
};

} // namespace mllm
#endif

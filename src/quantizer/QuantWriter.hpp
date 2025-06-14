#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/compute/QuantizeQ6.hpp"
#include "backends/cpu/compute/QuantizeQ2.hpp"
#include "backends/cpu/compute/QuantizeQ3.hpp"
#include "backends/cpu/compute/QuantizeQ4.hpp"
#include "backends/cpu/compute/QuantizeQ8.hpp"
#include "backends/cpu/compute/GemmAarch64.hpp"
#include "backends/cpu/compute/GemmKleidiai.hpp"
#include <cassert>
#include <string>
#include <unordered_map>
#ifndef MLLM_QUANTWRITER_HPP
#define MLLM_QUANTWRITER_HPP
#define NOT_IMPLEMENTED(x)                                                            \
    std::cout << "Quantize params to " << DataTypeName(x) << " is not implemented\n"; \
    __exit(-1);
#define UNREACHABLE()                  \
    std::cout << "Unreachable code\n"; \
    __exit(-1);
#define __exit(status)                        \
    {                                         \
        if (status != 0) {                    \
            std::cout << "Quantize failed\n"; \
            remove(output_path_.c_str());     \
        }                                     \
        exit(status);                         \
    }
static std::pair<void *, uint64_t> alloc_quant_block(uint64_t count, DataType type) {
    uint64_t size = DataTypeSize(type, count);
    if (size <= 0) {
        return std::make_pair(nullptr, 0);
    }
    void *data = new char[size];
    return std::make_pair(data, size);
}

#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
static std::pair<void *, uint64_t> alloc_kleidiai_quant_block(DataType type, int N, int K) {
    assert(type == MLLM_TYPE_KLEIDIAI_Q4_0);
    uint64_t size = mllm_kleidai_get_packed_b_qsi4_size(N, K);
    if (size <= 0) {
        return std::make_pair(nullptr, 0);
    }
    void *data = new uint8_t[size];
    return std::make_pair(data, size);
}
#endif
namespace mllm {
extern std::vector<std::string> vl_q4x4_2_q4_k_layers;

class QuantWriter : public ParamWriter {
public:
    ~QuantWriter();
    explicit QuantWriter(std::string output_path, std::string input_path);
    int readParams();
    void quantParams(DataType dataType);
    void quantParams_q4_(DataType dataType);
    void quantParams_q4_vl(DataType dataType);
    void quantParams_kai_vl(DataType dataType);

#ifdef TEST
    std::unordered_map<string, char *> data_;

#endif
private:
    string output_path_;
    mllm::ParamLoader *param_loader_;
    DataType quant_type_;
    std::vector<std::string> param_names_;
    float *getParam(std::string param_name);
    void writeParam(string name, DataType type, void *data, uint64_t size) override;
};
} // namespace mllm
#endif
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ6.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ2.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ3.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ4.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ8.hpp"
#include "backends/cpu/third_party/ggml/GemmPack.hpp"
#include "backends/cpu/compute/GemmKleidiai.hpp"
#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

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

#ifndef KAI_FP16_CAL
    uint64_t size = mllm_kleidai_get_packed_b_qsi4_size(N, K);
#else
    uint64_t size = mllm_kleidai_get_packed_b_qsi4_size_to_fp16(N, K);
#endif
    if (size <= 0) {
        return std::make_pair(nullptr, 0);
    }
    void *data = new uint8_t[size];
    return std::make_pair(data, size);
}
#endif

namespace mllm {
extern const std::vector<std::string> q4_0_kai_to_q4_0_4x4_layers;

class QuantWriter : public ParamWriter {
public:
    ~QuantWriter();
    explicit QuantWriter(std::string output_path, std::string input_path);
    int readParams();

    void quantize(DataType target_quant_type, const std::string &other_flag = "");

private:
    std::string output_path_;
    ParamLoader *param_loader_;
    std::vector<std::string> param_names_;
    std::vector<std::string> original_param_names_;

    DataType getQuantizationTypeFor(const std::string &name, DataType target_type, const std::string &other_flag);

    std::vector<float> load_full_fp32_param(const std::string &name);
};
} // namespace mllm
#endif
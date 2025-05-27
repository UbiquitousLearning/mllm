#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/quantize/QuantizeQ6.hpp"
#include "backends/cpu/quantize/QuantizeQ2.hpp"
#include "backends/cpu/quantize/QuantizeQ3.hpp"
#include "backends/cpu/quantize/QuantizeQ4.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"
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
namespace mllm {
class QuantWriter : public ParamWriter {
public:
    ~QuantWriter();
    explicit QuantWriter(std::string output_path, std::string input_path);
    int readParams();
    void quantParams(DataType dataType);
    void quantParams_q4_(DataType dataType);
    void quantParams_q4_vl(DataType dataType);

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
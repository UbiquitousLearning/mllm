#include <map>
#include <string>
namespace mllm::opencl {
extern const char* transpose;
extern const char* rmsnorm;
extern const char* causal_mask;
extern const char* embedding;
extern const char* softmax;
extern const char* sub;
extern const char* mul;
extern const char* div;
extern const char* fill;
extern const char* matmul_transb_bias;
extern const char* rope;
extern const char* silu;
extern const char* add;
extern const char* matmul;
const std::map<std::string, const char*> OpenCLProgramMap = {
    {"transpose", transpose},
    {"rmsnorm", rmsnorm},
    {"causal_mask", causal_mask},
    {"embedding", embedding},
    {"softmax", softmax},
    {"sub", sub},
    {"mul", mul},
    {"div", div},
    {"fill", fill},
    {"matmul_transb_bias", matmul_transb_bias},
    {"rope", rope},
    {"silu", silu},
    {"add", add},
    {"matmul", matmul},
};
}  // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = {
    {"transpose", "87caf150f16b16711c85f250ef3b4078"},   {"rmsnorm", "c80a3d46b5760cddc6cca0205a7b6e85"},
    {"causal_mask", "91d58f2fc38e1011d07dee99abefe12b"}, {"embedding", "2f53599ee23db57480589c629c0d2069"},
    {"softmax", "ed7b176870d74194ae683712b71bc2fc"},     {"sub", "e85bb10e1ddad3d0e38f09e6b9fec7c3"},
    {"mul", "1241c7141443ad9e18dadd14e150516a"},         {"div", "5936825bc33ffe218c11ade20fd68203"},
    {"fill", "0378c49b52d12aee7d7fb0d8495945c7"},        {"matmul_transb_bias", "9fd958568df8525ae1b6d620ff0bf5e5"},
    {"rope", "a0fb1d3c9e5f3cfbb6384471d8a40591"},        {"silu", "c9c2197b4b426d12cc652296738c24e1"},
    {"add", "a1ed4a6207f790f5dc436cf841744cf3"},         {"matmul", "1a07079712a494d8134540564ce4fad4"},
};

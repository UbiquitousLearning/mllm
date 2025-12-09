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
    {"add", add},
    {"matmul", matmul},
};
}  // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = {
    {"transpose", "87caf150f16b16711c85f250ef3b4078"},   {"rmsnorm", "c80a3d46b5760cddc6cca0205a7b6e85"},
    {"causal_mask", "91d58f2fc38e1011d07dee99abefe12b"}, {"embedding", "2f53599ee23db57480589c629c0d2069"},
    {"softmax", "ed7b176870d74194ae683712b71bc2fc"},     {"sub", "3593265b73a076f2e79aaff180b9d6f6"},
    {"mul", "162e0bc6ce3ed44fedae5ab099b1d238"},         {"div", "698d01e387e3da981675655bcdcce90e"},
    {"fill", "0378c49b52d12aee7d7fb0d8495945c7"},        {"matmul_transb_bias", "a492b265b35d5f92810d2c88057f2b85"},
    {"rope", "a0fb1d3c9e5f3cfbb6384471d8a40591"},        {"add", "1e498f3580b345629f5798cb9a281f05"},
    {"matmul", "1a07079712a494d8134540564ce4fad4"},
};

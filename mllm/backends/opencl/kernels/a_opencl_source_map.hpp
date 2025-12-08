#include <map>
#include <string>
namespace mllm::opencl {
extern const char* transpose;
extern const char* rmsnorm;
extern const char* embedding;
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
    {"embedding", embedding},
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
    {"transpose", "87caf150f16b16711c85f250ef3b4078"}, {"rmsnorm", "c80a3d46b5760cddc6cca0205a7b6e85"},
    {"embedding", "2f53599ee23db57480589c629c0d2069"}, {"sub", "3593265b73a076f2e79aaff180b9d6f6"},
    {"mul", "162e0bc6ce3ed44fedae5ab099b1d238"},       {"div", "698d01e387e3da981675655bcdcce90e"},
    {"fill", "0378c49b52d12aee7d7fb0d8495945c7"},      {"matmul_transb_bias", "a492b265b35d5f92810d2c88057f2b85"},
    {"rope", "a0fb1d3c9e5f3cfbb6384471d8a40591"},      {"add", "f99367757685844e54326166a9e44840"},
    {"matmul", "1a07079712a494d8134540564ce4fad4"},
};

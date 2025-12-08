#include <map>
#include <string>
namespace mllm::opencl {
extern const char* embedding;
extern const char* add;
const std::map<std::string, const char*> OpenCLProgramMap = {
    {"embedding", embedding},
    {"add", add},
};
}  // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = {
    {"embedding", "2f53599ee23db57480589c629c0d2069"},
    {"add", "f99367757685844e54326166a9e44840"},
};

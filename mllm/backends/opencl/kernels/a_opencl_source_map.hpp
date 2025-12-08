#include <map>
#include <string>
namespace mllm::opencl {
extern const char* transpose;
extern const char* embedding;
extern const char* add;
const std::map<std::string, const char*> OpenCLProgramMap = {
    {"transpose", transpose},
    {"embedding", embedding},
    {"add", add},
};
}  // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = {
    {"transpose", "87caf150f16b16711c85f250ef3b4078"},
    {"embedding", "2f53599ee23db57480589c629c0d2069"},
    {"add", "f99367757685844e54326166a9e44840"},
};

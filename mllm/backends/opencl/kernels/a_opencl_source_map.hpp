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
    {"embedding", "eddb43cd3a7ae942a64023718b5b4555"},
    {"add", "59750673e272af2cf01d2f9cd8422ede"},
};

#include <map>
#include <string>
namespace mllm::opencl {
extern const char* add;
const std::map<std::string, const char*> OpenCLProgramMap = {
    {"add", add},
};
}  // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = {
    {"add", "59750673e272af2cf01d2f9cd8422ede"},
};

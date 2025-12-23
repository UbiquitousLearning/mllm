#include "mllm/backends/qnn/aot/passes/AOTPipeline.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/backends/qnn/aot/passes/MarkQnnGraphPass.hpp"
#include "mllm/backends/qnn/aot/passes/MarkTensorIO.hpp"
#include "mllm/backends/qnn/aot/passes/OpNamingPass.hpp"

namespace mllm::qnn::aot {
std::vector<std::shared_ptr<ir::Pass>> createQnnAOTLoweringPipeline(QnnAOTEnv* env, const std::string& config_path) {
  std::vector<ir::Pass::ptr_t> ret;

  AOTCompileContext::getInstance().setEnv(env);
  AOTCompileContext::getInstance().setConfig(config_path);

  ret.emplace_back(createMarkQnnGraphPass());
  ret.emplace_back(createOpNamingPass());

  ret.emplace_back(createMarkTensorIOPass());

  return ret;
}
}  // namespace mllm::qnn::aot

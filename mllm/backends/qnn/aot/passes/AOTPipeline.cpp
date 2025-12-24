#include "mllm/backends/qnn/aot/passes/AOTPipeline.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/backends/qnn/aot/passes/LLMQuantRecipePass.hpp"
#include "mllm/backends/qnn/aot/passes/MarkQnnGraphPass.hpp"
#include "mllm/backends/qnn/aot/passes/MarkTensorIO.hpp"
#include "mllm/backends/qnn/aot/passes/MergeLLMHeadIntoMainGraphPass.hpp"
#include "mllm/backends/qnn/aot/passes/OpNamingPass.hpp"

namespace mllm::qnn::aot {
std::vector<std::shared_ptr<ir::Pass>> createQnnAOTLoweringPipeline(QnnAOTEnv* env, const std::string& config_path) {
  std::vector<ir::Pass::ptr_t> ret;

  AOTCompileContext::getInstance().setEnv(env);
  AOTCompileContext::getInstance().setConfig(config_path);
  auto config = AOTCompileContext::getInstance().getConfig();

  if (config.contains("quant_recipe") && config["quant_recipe"].contains("llm_recipe")
      && config["quant_recipe"]["llm_recipe"] == true) {
    ret.emplace_back(createMarkQnnGraphPass());
    ret.emplace_back(createOpNamingPass());
    ret.emplace_back(createMergeLLMHeadIntoMainGraphPass());
    ret.emplace_back(createLLMQuantRecipePass());
  } else {
    MLLM_WARN("This pass currently only supports LLM applications. Please ensure your config contains 'quant_recipe.llm_recipe "
              "= true'.");
  }

  return ret;
}
}  // namespace mllm::qnn::aot

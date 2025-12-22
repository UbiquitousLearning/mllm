#include "mllm/backends/qnn/aot/passes/AOTPipeline.hpp"

namespace mllm::qnn::aot {
std::vector<std::shared_ptr<ir::Pass>> createQnnAOTLoweringPipeline() {
  std::vector<ir::Pass::ptr_t> ret;

  return ret;
}
}  // namespace mllm::qnn::aot

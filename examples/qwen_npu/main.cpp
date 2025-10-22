#include <fmt/core.h>
#include <cstdint>
#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphBuildPipeline.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"

using mllm::Argparse;

MLLM_MAIN({
  mllm::initQnnBackend();

  const std::string config_path = "./config_1.8B_w8a16_qnn.json";
  const std::string model_path = "./qwen1.5-1.8b-chat-rot-qnn.mllm";

  auto qwen_tokenizer = mllm::models::qwen_npu::QwenTokenizer("./tokenizer.json", "./qwen_merges.txt");

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;

  auto cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);
  auto model = mllm::models::qwen_npu::QwenForCausalLM("", cfg);

  auto param = mllm::load(model_path, file_version);
  model.load(param);

  mllm::models::ARGenerationOutputPast inputs{{"sequence", mllm::Tensor::empty({1, 32}, mllm::kInt64, mllm::kCPU).alloc()}};

  auto irs = model.trace(inputs, {});

  // QNN Graph Rewrite Pass
  mllm::ir::PassManager rewritePM(irs["model"]);
  rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
  rewritePM.run();

  // have a look at the IR after QNN Graph Rewrite Pass
  mllm::redirect("qwen_npu.mir", [&]() { mllm::print(irs["model"]); });

  // QNN Graph Build Pass
  mllm::ir::PassManager graphBuildPM(irs["model"]);
  graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
  graphBuildPM.run();

  // cache has been updated due to trace, clear cache
  model.model.clearKVCache();

  auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "How are you?"})["sequence"];
  print(raw_input_tokens);

  // manually set input data as fill op is not supported in QNN
  auto ptr = inputs["sequence"].ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();
  for (int i = 0; i < raw_input_tokens.shape()[1]; ++i) { ptr[i] = input_data[i]; }
  for (int i = raw_input_tokens.shape()[1]; i < 32; ++i) { ptr[i] = -1; }

  auto out = model.forward(inputs, {{"seq_len", mllm::AnyValue((int)raw_input_tokens.shape()[1])}})["sequence"];

  auto sampled = model.sampleGreedy(out);
  std::wcout << "token: " << sampled << " " << qwen_tokenizer.detokenize(sampled) << "\n";

  return 0;
})
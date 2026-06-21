#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <vector>
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/models/llama/configuration_llama.hpp"
#include "mllm/models/llama/tokenization_tiny_llama.hpp"
#include "mllm/models/llama/modeling_llama_opencl.hpp"

MLLM_MAIN({
  mllm::initOpenCLBackend();

  auto llama_cfg = mllm::models::llama::LLaMAConfig("./config_tiny_llama.json");
  auto llama_tokenizer = mllm::models::llama::TinyLlamaTokenizer("./tiny-llama-tokenizer.json");
  auto llama = mllm::models::llama::LlamaForCausalLM("", llama_cfg);

  auto param = mllm::load("tinyllama-1.1b-chat-q40.mllm", mllm::ModelFileVersion::kV1);

  llama.load(param);
  llama.to(mllm::kOpenCL);

  auto inputs = llama_tokenizer.convertMessage({
      {
          .role = "system",
          .content = "You are a friendly chatbot who always responds in the style of a pirate",
      },
      {.role = "user", .content = "hello how are you?"},
  });

  for (int i = 0; i < 10; i++) {
    inputs["sequence"] = inputs["sequence"].to(mllm::kOpenCL);

    inputs = llama.forward(inputs, {});

    inputs["sequence"] = inputs["sequence"].to(mllm::kCPU);

    int64_t next_token_id = llama.sampleGreedy(inputs["sequence"]);
    std::wcout << llama_tokenizer.detokenize(next_token_id) << std::flush;

    inputs["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
    inputs["sequence"].at<mllm::mllm_int64_t>({0, 0}) = next_token_id;
  }
});
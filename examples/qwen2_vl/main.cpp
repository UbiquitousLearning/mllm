#include <mllm/mllm.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

using mllm::Argparse;

int main(int argc, char** argv) {
  mllm::initializeContext();

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");

  {
    auto qwen2vl_cfg = mllm::models::qwen2vl::Qwen2VLConfig();
    auto qwen2vl = mllm::models::qwen2vl::Qwen2VLForCausalLM(qwen2vl_cfg);

    mllm::print(qwen2vl.llm);
    mllm::print(qwen2vl.visual);
  }

  mllm::memoryReport();
  mllm::shutdownContext();
}

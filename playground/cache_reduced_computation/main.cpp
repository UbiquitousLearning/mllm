#include <chrono>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3.hpp>
#include <mllm/models/qwen3/configuration_qwen3.hpp>

using mllm::Argparse;

struct Qwen3BenchOptions {
  int prefix_cache_length = 0;
  int prefill_tokens = 0;
};

int qwen3Bench(const std::string& cfg_fp, const std::string& model_fp, const Qwen3BenchOptions& options) {
  mllm::models::qwen3::Qwen3Config cfg(cfg_fp);
  cfg.max_cache_length = 4096;
  mllm::models::qwen3::Qwen3ForCausalLM model(cfg);
  model.load(mllm::load(model_fp, mllm::ModelFileVersion::kV2));

  // Reset Elements in kv cache
  model.kvCache().setCurrentSeqCnt(options.prefix_cache_length);

  // Make Fake Data
  auto inputs = mllm::Tensor::random({1, options.prefill_tokens}, -1, 1, mllm::kInt64);
  auto t0 = std::chrono::steady_clock::now();
  model.generate(mllm::models::ARGenerationOutputPast{{"sequence", inputs}},
                 mllm::models::ARGenerationArgs{{"max_length", mllm::AnyValue(1)}});
  auto t1 = std::chrono::steady_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  mllm::print("cost (int ms) = ", ms_int.count(), " ms\n");
  return 0;
}

int main(int argc, char** argv) {
  mllm::initializeContext();
  auto& config_files = Argparse::add<std::string>("-c|--config").help("config file path.");
  auto& model_files = Argparse::add<std::string>("-m|--model").help("model file path.");
  auto& prefix_cache_length = Argparse::add<int>("-p|--pcl").help("prefix_cache_length");
  auto& prefill_tokens = Argparse::add<int>("-t|--pt").help("prefill_tokens");

  Argparse::parse(argc, argv);

  (void)qwen3Bench(config_files.get(), model_files.get(),
                   {
                       .prefix_cache_length = prefix_cache_length.get(),
                       .prefill_tokens = prefill_tokens.get(),
                   });

  mllm::shutdownContext();
}

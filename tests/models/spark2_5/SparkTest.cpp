// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include "mllm/models/spark2_5/modeling_spark2_5.hpp"
#include "mllm/models/spark2_5/tokenization_spark2_5.hpp"
using namespace mllm;
using namespace mllm::models::spark2_5;
class SparkTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { initializeContext(); }
};
static std::string exampleDir() {
  auto* p = std::getenv("MLLM_SPARK_EXAMPLE_DIR");
  return p ? p : SPARK_EXAMPLE_DIR;
}
TEST_F(SparkTest, OfficialConfigAndTemplate) {
  SparkConfig c(exampleDir() + "/config_1.7B_fp32.json");
  EXPECT_EQ(c.full_layers, 7);
  EXPECT_EQ(c.rotary_dim, 64);
  EXPECT_EQ(c.head_dim, 256);
  EXPECT_EQ(c.sliding_window, 512);
  EXPECT_EQ(SparkTokenizer::applyChatTemplate({"Hi", "", false}),
            "<｜start▁of▁sentence｜><|System|>\nyou are a helpful "
            "assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Hi<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|"
            "></think>");
  EXPECT_THROW(SparkTokenizer::applyChatTemplate({"", "", true}), std::invalid_argument);
}
TEST_F(SparkTest, PretokenizerUnicodeAndDigits) {
  EXPECT_EQ(sparkPieces(L"12345"), (std::vector<std::wstring>{L"1", L"2", L"3", L"4", L"5"}));
  EXPECT_EQ(sparkPieces(L"中文hello"), (std::vector<std::wstring>{L"中文", L"hello"}));
  EXPECT_EQ(sparkPieces(L"e\u0301"), (std::vector<std::wstring>{L"e\u0301"}));
  EXPECT_EQ(sparkPieces(L" hello"), (std::vector<std::wstring>{L" hello"}));
}
TEST_F(SparkTest, OfficialTokenizerParity) {
  auto* root = std::getenv("MLLM_SPARK_TOKENIZER_FIXTURES");
  if (!root) GTEST_SKIP() << "Official checkpoint fixture is external";
  const auto j = nlohmann::json::parse(std::ifstream(std::string(root) + "/tokenizer_cases.json"));
  SparkTokenizer t(std::string(root) + "/tokenizer.json");
  for (const auto& c : j) {
    const auto text = c.at("text").get<std::string>();
    EXPECT_EQ(t.encode(text), c.at("ids").get<std::vector<int64_t>>()) << text;
    std::string decoded;
    for (auto id : t.encode(text)) decoded += t.detokenizeBytes(id);
    EXPECT_EQ(decoded, text);
  }
}
TEST_F(SparkTest, SmallModelChunkAndReset) {
  SparkConfig c(exampleDir() + "/config_1.7B_fp32.json");
  c.vocab_size = 32;
  c.hidden_size = 16;
  c.intermediate_size = 24;
  c.head_dim = 8;
  c.rotary_dim = 2;
  c.num_attention_heads = 2;
  c.num_key_value_heads = 1;
  c.num_hidden_layers = 4;
  c.full_layers = 1;
  c.sliding_window = 4;
  c.max_cache_length = 32;
  c.layer_types = {"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"};
  auto params = ParameterFile::create();
  auto add = [&](std::string name, Tensor::shape_t shape, bool norm = false) {
    auto t = Tensor::empty(shape, kFloat32, kCPU).alloc();
    for (size_t i = 0; i < t.numel(); ++i) t.ptr<float>()[i] = norm ? 1.0F : 0.07F * std::sin(float(i + 1));
    params->push(name, t);
  };
  add("model.embedding.weight", {32, 16});
  add("lm_head.weight", {32, 16});
  add("model.norm.weight", {16}, true);
  for (int l = 0; l < 4; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    add(p + "input_layernorm.weight", {16}, true);
    add(p + "post_attention_layernorm.weight", {16}, true);
    add(p + "self_attn.q_k_v_proj.weight", {32, 16});
    add(p + "self_attn.g_proj.weight", {2, 16});
    add(p + "self_attn.out_proj.weight", {16, 16});
    add(p + "mlp.gate_proj.weight", {24, 16});
    add(p + "mlp.up_proj.weight", {24, 16});
    add(p + "mlp.down_proj.weight", {16, 24});
  }
  SparkForCausalLM m(c);
  m.load(params);
  m.all_logits = true;
  auto tokens = Tensor::empty({1, 13}, kInt64, kCPU).alloc();
  for (int i = 0; i < 13; ++i) tokens.ptr<int64_t>()[i] = i;
  auto full = m.forward({{"sequence", tokens}}, {}).at("sequence").clone();
  for (int chunk : {1, 3, 7}) {
    m.resetState();
    for (int i = 0; i < 13; i += chunk) {
      int end = std::min(i + chunk, 13);
      auto part = m.forward({{"sequence", tokens[{kAll, {i, end}}].contiguous()}}, {}).at("sequence");
      for (int t = i; t < end; ++t)
        for (int d = 0; d < 32; ++d) EXPECT_NEAR(part.ptr<float>()[(t - i) * 32 + d], full.ptr<float>()[t * 32 + d], 2e-5F);
    }
  }
  m.resetState();
  auto again = m.forward({{"sequence", tokens}}, {}).at("sequence");
  for (size_t i = 0; i < full.numel(); ++i) EXPECT_FLOAT_EQ(full.ptr<float>()[i], again.ptr<float>()[i]);
}

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <iomanip>
#include <random>

#include <mllm/mllm.hpp>
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/models/qwen3/modeling_qwen3_probing_service.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using namespace mllm;
using namespace mllm::models::qwen3_probing;
using namespace mllm::models::qwen3;
namespace fs = std::filesystem;

struct Sample {
  std::string question;
  std::string exact_answer;
};

// Robust CSV Parser
std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> result;
  bool in_quote = false;
  std::string field;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') {
      if (in_quote && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';  // escaped quote
        i++;
      } else {
        in_quote = !in_quote;
      }
    } else if (c == ',' && !in_quote) {
      result.push_back(field);
      field.clear();
    } else {
      field += c;
    }
  }
  result.push_back(field);
  return result;
}

std::vector<Sample> load_samples(const std::string& path) {
  std::vector<Sample> samples;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Cannot open file: " << path << std::endl;
    return samples;
  }
  std::string line;
  std::getline(file, line);  // header

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    auto fields = parse_csv_line(line);

    std::string q, exact;

    // 0: Unnamed, 1: raw_q, 2: q, 3: model_ans, 4: correct, 5: auto, 6: exact
    if (fields.size() > 6) {
      q = fields[2];
      exact = fields[6];
    } else {
      continue;
    }

    if (exact == "NO ANSWER" || exact == "Answer:" || exact == "[]") continue;
    if (exact.empty()) continue;

    samples.push_back({q, exact});
  }
  return samples;
}

std::string normalize(std::string s) {
  std::string out;
  for (char c : s) {
    if (!std::ispunct(c)) out += std::tolower(c);
  }
  return out;
}

MLLM_MAIN({
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <model_path> <probes_path> <csv_path> [limit]" << std::endl;
    return 1;
  }
  std::string model_path = argv[1];
  std::string probes_path = argv[2];
  std::string csv_path = argv[3];
  int limit = (argc > 4) ? std::stoi(argv[4]) : -1;

  // try {
  std::cout << "Loading samples..." << std::endl;
  auto samples = load_samples(csv_path);
  if (samples.empty()) {
    std::cerr << "No samples loaded from " << csv_path << std::endl;
    return 1;
  }
  std::cout << "Loaded " << samples.size() << " samples." << std::endl;

  std::cout << "Initializing session..." << std::endl;
  // Load Model
  auto session = std::make_unique<Qwen3ProbingSession>();
  std::cout << "Loading model from " << model_path << "..." << std::endl;
  session->fromPreTrain(model_path);

  // Config Probing
  std::cout << "Loading probes..." << std::endl;
  ProbingArgs p_args;
  p_args.enable_prefill_check = false;
  p_args.enable_decode_check = true;
  p_args.decode_stop_threshold = 1.1f;
  p_args.pos_threshold = 0.9f;  // Set a realistic high threshold for debounce strategy

  session->setProbingArgs(p_args);
  session->loadProbes(probes_path, p_args);

  // Tokenizer
  std::cout << "Loading tokenizer..." << std::endl;
  Qwen3Tokenizer tokenizer(model_path + "/tokenizer.json");

  int total_gen_tokens = 0;
  int global_tp = 0;
  int global_activations = 0;
  int global_fn = 0;  // Estimate
  int global_real_positives = 0;

  for (int i = 0; i < samples.size(); ++i) {
    if (limit > 0 && i >= limit) break;
    const auto& s = samples[i];

    std::cout << "\n=== Q [" << i << "]: " << s.question.substr(0, 100) << "..." << std::endl;
    std::cout << "Target Exact: " << s.exact_answer << std::endl;

    session->clearLastProbeResults();

    nlohmann::json request;
    request["messages"] = nlohmann::json::array();
    request["messages"].push_back({{"role", "user"}, {"content", s.question}});
    request["max_length"] = 512;
    request["do_sample"] = false;

    std::string full_response;
    std::vector<std::string> generated_tokens_list;

    try {
      session->streamGenerate(request, [&](const nlohmann::json& chunk, bool is_finish) {
        if (chunk.is_string()) {
          std::string t = chunk.get<std::string>();
          generated_tokens_list.push_back(t);
          full_response += t;
        }
      });
    } catch (const std::exception& e) {
      std::cerr << "Generation failed for Q[" << i << "]: " << e.what() << std::endl;
      continue;
    }

    std::cout << "Model Answer: " << full_response.substr(0, 100) << "..." << std::endl;

    auto results = session->getLastProbeResults();
    std::string target_norm = normalize(s.exact_answer);

    std::set<int> real_positive_indices;

    // 1. Identify ALL Real Positives in the generated sequence
    for (int t_idx = 0; t_idx < generated_tokens_list.size(); ++t_idx) {
      std::string t_norm = normalize(generated_tokens_list[t_idx]);
      bool matches_target = false;
      if (!t_norm.empty() && target_norm.find(t_norm) != std::string::npos) {
        // Heuristic: Length > 2 OR exact match short word
        if (t_norm.length() > 2 || target_norm == t_norm) { matches_target = true; }
      }
      if (matches_target) {
        real_positive_indices.insert(t_idx);
        global_real_positives++;
      }
      total_gen_tokens++;
    }

    // 2. Check Probe Activations (TP vs FP)
    int local_tp = 0;
    int local_fp = 0;

    for (const auto& res : results) {
      if (res.type != "pos_check") continue;

      // Note: res.token_idx is the index in the generated sequence
      if (real_positive_indices.count(res.token_idx)) {
        local_tp++;
        global_tp++;
      } else {
        local_fp++;
      }
      global_activations++;
    }

    // 3. Estimate FN (Real Positives not detected)
    // Since one activation could cover "a phrase", strict token-wise matching is harsh for Recall.
    // But let's stick to token-wise for now.

    std::cout << " -> Stats: TP=" << local_tp << " FP=" << local_fp << " RealPos=" << real_positive_indices.size() << std::endl;
  }

  std::cout << "\n=== Strategy Evaluation (Thr=0.9, Debounced) ===" << std::endl;
  std::cout << "Total Checked Tokens: " << total_gen_tokens << std::endl;
  std::cout << "Total Real Positives: " << global_real_positives << std::endl;
  std::cout << "Total Activations:    " << global_activations << std::endl;

  double precision = (global_activations > 0) ? (double)global_tp / global_activations : 0.0;
  double recall = (global_real_positives > 0) ? (double)global_tp / global_real_positives : 0.0;
  double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

  std::cout << "\nFinal Metrics:" << std::endl;
  std::cout << "Precision: " << std::fixed << std::setprecision(2) << precision * 100.0 << "% (" << global_tp << "/"
            << global_activations << ")" << std::endl;
  std::cout << "Recall:    " << std::fixed << std::setprecision(2) << recall * 100.0 << "% (" << global_tp << "/"
            << global_real_positives << ")" << std::endl;
  std::cout << "F1-Score:  " << std::fixed << std::setprecision(2) << f1 * 100.0 << "%" << std::endl;

  return 0;
})

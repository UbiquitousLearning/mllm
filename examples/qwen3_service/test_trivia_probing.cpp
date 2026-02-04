#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <map>
#include <set>
#include <iomanip>
#include <cmath>

#include <fmt/core.h>
#include <fmt/color.h>
#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include "mllm/models/qwen3/modeling_qwen3_probing_service.hpp"

#include <chrono>

using namespace mllm;
using namespace mllm::models::qwen3_probing;
namespace fs = std::filesystem;

struct TriviaSample {
  std::string question;
  std::vector<std::string> answers;
  std::string expected_label;
};

struct GlobalStat {
  float max_prefill_score;  // 该样本所有Prefill层中的最高分
  bool is_model_wrong;      // 1=Wrong/Hallucination, 0=Correct
};

std::vector<std::string> split(const std::string& s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) { tokens.push_back(token); }
  return tokens;
}

std::vector<TriviaSample> loadTrivia(const std::string& path, int max_lines = -1) {
  std::vector<TriviaSample> samples;
  std::ifstream file(path);
  if (!file.is_open()) return samples;
  std::string line;
  std::getline(file, line);
  std::vector<std::string> all_lines;
  while (std::getline(file, line)) {
    if (!line.empty()) all_lines.push_back(line);
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 g(seed);
  std::shuffle(all_lines.begin(), all_lines.end(), g);
  for (const auto& l : all_lines) {
    bool in_quote = false;
    std::vector<std::string> fields;
    std::string current_field;
    for (char c : l) {
      if (c == '"') {
        in_quote = !in_quote;
      } else if (c == ',' && !in_quote) {
        fields.push_back(current_field);
        current_field.clear();
      } else {
        current_field += c;
      }
    }
    fields.push_back(current_field);
    if (fields.size() < 2) continue;
    std::string q, a_str;
    std::string f0 = fields[0];
    if (f0.size() >= 2 && f0.front() == '"' && f0.back() == '"') f0 = f0.substr(1, f0.size() - 2);
    if ((fields.size() >= 2) && (f0.find('_') != std::string::npos || f0.find("tc-") != std::string::npos)) {
      q = fields[1];
      if (fields.size() > 3)
        a_str = fields[3];
      else if (fields.size() > 2)
        a_str = fields[2];
      else
        a_str = "";
    } else {
      q = fields[0];
      int ans_idx = (fields.size() > 6) ? 6 : 1;
      if (fields.size() <= ans_idx) ans_idx = fields.size() - 1;
      a_str = fields[ans_idx];
    }
    if (q.size() >= 2 && q.front() == '"' && q.back() == '"') q = q.substr(1, q.size() - 2);
    if (a_str.size() >= 2 && a_str.front() == '"' && a_str.back() == '"') a_str = a_str.substr(1, a_str.size() - 2);
    if (q.find("bt_") == 0 || q.find("tc_") == 0 || q.length() < 5) continue;
    if (q.size() >= 2 && q.front() == '"' && q.back() == '"') q = q.substr(1, q.size() - 2);
    if (a_str.size() >= 2 && a_str.front() == '"' && a_str.back() == '"') a_str = a_str.substr(1, a_str.size() - 2);
    size_t val_pos = a_str.find("'Value': '");
    if (val_pos != std::string::npos) {
      size_t end_pos = a_str.find("'", val_pos + 10);
      if (end_pos != std::string::npos) {
        a_str = a_str.substr(val_pos + 10, end_pos - (val_pos + 10));
        samples.push_back({q, {a_str}});
        continue;
      }
    }
    auto ans_list = split(a_str, '|');
    samples.push_back({q, ans_list});
  }
  return samples;
}

std::vector<TriviaSample> loadReplay(const std::string& path) {
  std::vector<TriviaSample> samples;
  std::ifstream file(path);
  if (!file.is_open()) return samples;
  nlohmann::json j;
  file >> j;
  for (const auto& item : j) {
    TriviaSample s;
    if (item.contains("question")) s.question = item["question"].get<std::string>();
    if (item.contains("refs"))
      for (const auto& r : item["refs"]) s.answers.push_back(r.get<std::string>());
    if (item.contains("label")) s.expected_label = item["label"].get<std::string>();
    if (!s.question.empty()) samples.push_back(s);
  }
  return samples;
}

std::string normalize(std::string s) {
  if (s.empty()) return "";
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    unsigned char c = s[i];
    if (c < 128) {
      if (!std::ispunct(c) && !std::iscntrl(c)) out += std::tolower(c);
    }
  }
  return out;
}

std::string removeThinking(const std::string& text) {
  std::string out = text;
  size_t start = out.find("<think>");
  size_t end = out.find("</think>");
  if (start != std::string::npos && end != std::string::npos && end > start) { out.erase(start, end - start + 8); }
  return out;
}

bool checkAnswer(const std::string& generated, const std::vector<std::string>& refs) {
  std::string clean_gen = removeThinking(generated);
  std::string gen_norm = normalize(clean_gen);
  for (const auto& ref : refs) {
    std::string ref_norm = normalize(ref);
    if (ref_norm.empty()) continue;
    if (gen_norm.find(ref_norm) != std::string::npos) return true;
  }
  return false;
}

// --- AUC ---
double calculate_auc(const std::vector<float>& positive_scores, const std::vector<float>& negative_scores) {
  if (positive_scores.empty() || negative_scores.empty()) return 0.0;

  struct Pair {
    float score;
    int label;
  };
  std::vector<Pair> all_samples;
  all_samples.reserve(positive_scores.size() + negative_scores.size());

  for (float s : positive_scores) all_samples.push_back({s, 1});
  for (float s : negative_scores) all_samples.push_back({s, 0});

  std::sort(all_samples.begin(), all_samples.end(), [](const Pair& a, const Pair& b) { return a.score > b.score; });

  double auc_sum = 0;
  double current_pos_count = 0;

  for (const auto& p : all_samples) {
    if (p.label == 1) {
      current_pos_count++;
    } else {
      auc_sum += current_pos_count;
    }
  }

  return auc_sum / (double)(positive_scores.size() * negative_scores.size());
}

void print_stats(const std::map<std::string, std::map<int, std::map<int, std::vector<float>>>>& stats) {
  for (auto& [phase, layer_map] : stats) {
    float threshold = (phase == "decode") ? 0.6f : 0.7f;

    std::cout << "\n--- Per-Layer Analysis (" << phase << ") [Thres=" << threshold << "] ---\n";
    std::cout << "Layer | Acc (Det) | AUC    | AvgScore (All) | Samples (C/W)\n";
    std::cout << "----------------------------------------------------------------\n";

    for (auto& [layer, correctness_map] : layer_map) {
      const auto& correct_vec = correctness_map.count(0) ? correctness_map.at(0) : std::vector<float>{};
      const auto& wrong_vec = correctness_map.count(1) ? correctness_map.at(1) : std::vector<float>{};

      int total = correct_vec.size() + wrong_vec.size();
      if (total == 0) continue;

      // Acc
      int tn = 0;
      for (float s : correct_vec)
        if (s < threshold) tn++;
      int tp = 0;
      for (float s : wrong_vec)
        if (s >= threshold) tp++;
      double acc = (double)(tn + tp) / total * 100.0;

      // Avg
      double sum_s = 0;
      for (float s : correct_vec) sum_s += s;
      for (float s : wrong_vec) sum_s += s;

      // AUC
      double auc = calculate_auc(wrong_vec, correct_vec);  // Wrong=Positive, Correct=Negative

      std::cout << "L" << std::setw(2) << layer << "   | " << std::fixed << std::setprecision(1) << std::setw(5) << acc
                << "%   | " << std::setprecision(3) << std::setw(6) << auc << " | " << std::setprecision(4) << (sum_s / total)
                << "         | " << correct_vec.size() << "/" << wrong_vec.size() << "\n";
    }
  }
}

void print_global_stats(const std::vector<GlobalStat>& global_stats) {
  float threshold = 0.7f;
  if (global_stats.empty()) return;

  std::vector<float> pos_scores, neg_scores;
  int g_tp = 0, g_tn = 0, g_fp = 0, g_fn = 0;

  for (const auto& s : global_stats) {
    if (s.is_model_wrong)
      pos_scores.push_back(s.max_prefill_score);
    else
      neg_scores.push_back(s.max_prefill_score);

    bool predicted_hallucination = (s.max_prefill_score >= threshold);
    if (s.is_model_wrong) {
      if (predicted_hallucination)
        g_tp++;
      else
        g_fn++;
    } else {
      if (!predicted_hallucination)
        g_tn++;
      else
        g_fp++;
    }
  }

  int total = global_stats.size();
  int total_wrong = g_tp + g_fn;
  int total_correct = g_tn + g_fp;

  double acc = (double)(g_tp + g_tn) / total * 100.0;
  double recall = (total_wrong > 0) ? (double)g_tp / total_wrong * 100.0 : 0.0;
  double precision = (g_tp + g_fp > 0) ? (double)g_tp / (g_tp + g_fp) * 100.0 : 0.0;
  double auc = calculate_auc(pos_scores, neg_scores);

  std::cout << "\n>>> WHOLE MODEL PREFILL STATS (Any Layer >= " << threshold << ") <<<\n";
  std::cout << "Total: " << total << " (Wrong: " << total_wrong << ", Correct: " << total_correct << ")\n";
  std::cout << "AUC:       " << std::fixed << std::setprecision(4) << auc << "  <--- Classification Capability\n";
  std::cout << "Accuracy:  " << std::fixed << std::setprecision(2) << acc << "%\n";
  std::cout << "Recall:    " << recall << "% (Caught " << g_tp << ")\n";
  std::cout << "Precision: " << precision << "%\n";
  std::cout << "Confusion: [TP:" << g_tp << " FN:" << g_fn << "] [FP:" << g_fp << " TN:" << g_tn << "]\n";
  std::cout << "---------------------------------------------------------\n";
}

MLLM_MAIN({
  mllm::setLogLevel(mllm::LogLevel::kError);
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& probe_path = mllm::Argparse::add<std::string>("-p|--probe_path").help("Probes dir").required(true);
  auto& data_path = mllm::Argparse::add<std::string>("-d|--data_path").help("Trivia CSV path");
  auto& replay_path = mllm::Argparse::add<std::string>("-r|--replay_file").help("Replay from JSON");
  auto& limit = mllm::Argparse::add<int>("--limit").help("Max samples").def(200);
  auto& balanced_target = mllm::Argparse::add<int>("-b|--balanced_target").def(0);
  mllm::Argparse::parse(argc, argv);

  if (data_path.get().empty() && replay_path.get().empty()) {
    std::cerr << "Error: Must provide either -d or -r" << std::endl;
    return 1;
  }

  auto session = std::make_shared<Qwen3ProbingSession>();
  try {
    session->fromPreTrain(model_path.get());
  } catch (const std::exception& e) {
    std::cerr << "Load Model Error: " << e.what() << std::endl;
    return 1;
  }

  ProbingArgs p_args;
  p_args.enable_prefill_check = true;
  p_args.enable_decode_check = true;
  p_args.prefill_stop_threshold = 1.1f;  // Don't stop early
  p_args.decode_stop_threshold = 1.1f;
  p_args.pos_threshold = 0.9f;

  for (int i = 0; i < 36; ++i) p_args.default_prefill_layers.push_back(i);
  session->setProbingArgs(p_args);
  session->loadProbes(probe_path.get(), p_args);

  std::cout << "Loading data..." << std::endl;
  std::vector<TriviaSample> samples;
  if (!replay_path.get().empty()) {
    samples = loadReplay(replay_path.get());
  } else {
    samples = loadTrivia(data_path.get());
  }
  std::cout << "Loaded " << samples.size() << " samples." << std::endl;

  std::map<std::string, std::map<int, std::map<int, std::vector<float>>>> stats;
  std::vector<GlobalStat> global_prefill_stats;

  nlohmann::json output_samples = nlohmann::json::array();
  int model_correct_total = 0;
  int model_wrong_total = 0;
  int processed_count = 0;
  int max_lim = limit.get();

  for (size_t i = 0; i < samples.size(); ++i) {
    if (processed_count >= max_lim && max_lim > 0) break;

    const auto& sample = samples[i];
    session->clearLastProbeResults();

    nlohmann::json req;
    nlohmann::json msg;
    msg["role"] = "user";
    msg["content"] = "Answer the question directly. Do not use <think> tags. Question: " + sample.question;
    req["messages"] = nlohmann::json::array({msg});
    req["max_length"] = 50;
    req["do_sample"] = false;
    req["enable_thinking"] = false;

    std::string generated_text = "";
    auto start_t = std::chrono::high_resolution_clock::now();
    int tok_cnt = 0;
    try {
      session->streamGenerate(req, [&](const nlohmann::json& j, bool finished) {
        if (j.is_string()) {
          generated_text += j.get<std::string>();
          tok_cnt++;
        }
      });
    } catch (...) { continue; }
    auto end_t = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(end_t - start_t).count();
    double tps = (dur > 0) ? (tok_cnt / dur) : 0.0;

    bool is_correct = checkAnswer(generated_text, sample.answers);
    if (is_correct)
      model_correct_total++;
    else {
      model_wrong_total++;
      std::cout << "\n[WRONG] Q: " << sample.question << "\n";
      std::cout << "        Gen: " << removeThinking(generated_text) << "\n";
      std::cout << "        Ref: " << sample.answers[0] << "\n";
    }

    std::cout << "\r[" << (i + 1) << "] C:" << model_correct_total << " W:" << model_wrong_total << " TPS:" << std::fixed
              << std::setprecision(1) << tps << " | " << sample.question.substr(0, 20) << "..." << std::flush;

    auto probe_data = session->getLastProbeResults();
    std::map<std::string, std::map<int, std::vector<float>>> sample_scores;
    float max_prefill_this_sample = 0.0f;

    for (const auto& p : probe_data) {
      sample_scores[p.phase][p.layer].push_back(p.score);
      if (p.phase == "prefill") {
        if (p.score > max_prefill_this_sample) max_prefill_this_sample = p.score;
      }
    }

    global_prefill_stats.push_back({max_prefill_this_sample, !is_correct});

    for (const auto& [ph, layer_map] : sample_scores) {
      for (const auto& [lay, scores] : layer_map) {
        if (scores.empty()) continue;
        double sum = 0;
        for (float s : scores) sum += s;
        double avg = sum / scores.size();
        stats[ph][lay][is_correct ? 0 : 1].push_back(avg);
      }
    }

    {
      nlohmann::json s_obj;
      s_obj["q"] = sample.question;
      s_obj["g"] = generated_text;
      s_obj["l"] = is_correct ? "correct" : "wrong";
      output_samples.push_back(s_obj);
    }
    processed_count++;

    if (processed_count % 10 == 0) {
      print_stats(stats);
      print_global_stats(global_prefill_stats);
    }
  }

  std::cout << "\n\nTotal: " << processed_count << " (Acc: " << (float)model_correct_total / processed_count * 100.0 << "%)\n";
  std::ofstream out_f("probing_results_replay.json");
  if (out_f.is_open()) {
    out_f << output_samples.dump(4);
    out_f.close();
  }

  print_stats(stats);
  print_global_stats(global_prefill_stats);

  return 0;
});
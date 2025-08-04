// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>

#include "mllm/utils/Common.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm {

ConfigFile::ConfigFile(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open config file: {}", file_path); }
  json_ = nlohmann::json::parse(file);
}

// Loads the configuration from a file.
void ConfigFile::load(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open config file: {}", file_path); }

  try {
    // The nlohmann::json stream operator overload handles parsing directly.
    file >> json_;
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("JSON parse error in file " + file_path + ": " + e.what());
  }
}

// Loads the configuration from a string.
void ConfigFile::loadString(const std::string& json_str) {
  try {
    json_ = nlohmann::json::parse(json_str);
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("JSON parse error in string: " + std::string(e.what()));
  }
}

// Dumps the JSON object to a formatted string.
std::string ConfigFile::dump() const {
  // The argument '4' pretty-prints the JSON with an indent of 4 spaces.
  // Use dump() with no arguments for a compact output.
  return json_.dump(4);
}

// Saves the current JSON object to a file.
void ConfigFile::save(const std::string& file_path) {
  std::ofstream file(file_path);
  if (!file.is_open()) { throw std::runtime_error("Failed to open file for saving: " + file_path); }
  // Write the pretty-printed version of the JSON to the file.
  file << dump();
}

}  // namespace mllm

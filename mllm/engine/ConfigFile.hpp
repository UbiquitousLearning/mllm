// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "mllm/core/DataTypes.hpp"

namespace mllm {

/**
 * @class ConfigFile
 * @brief A utility class for handling JSON-based configuration files.
 *
 * This class wraps the nlohmann::json library to provide a simple interface
 * for loading, saving, and manipulating configuration data from files or strings.
 */
class ConfigFile {
 public:
  /**
   * @brief Default constructor.
   */
  ConfigFile() = default;

  /**
   * @brief Constructor that loads a JSON configuration file.
   * @param file_path Path to the JSON file to load.
   * @throws std::runtime_error if the file cannot be opened or parsing fails.
   */
  explicit ConfigFile(const std::string& file_path);

  /**
   * @brief Loads configuration from a JSON string.
   * @param json_str The string containing the JSON data.
   * @throws std::runtime_error if parsing the string fails.
   */
  void loadString(const std::string& json_str);

  /**
   * @brief Loads configuration from a JSON file.
   * @param file_path Path to the JSON file to load.
   * @throws std::runtime_error if the file cannot be opened or parsing fails.
   */
  void load(const std::string& file_path);

  /**
   * @brief Dumps the current configuration data to a JSON string.
   * @return A formatted (pretty-printed) string representing the JSON.
   */
  [[nodiscard]] std::string dump() const;

  /**
   * @brief Saves the current configuration to a file.
   * @param file_path Path to the file where the JSON data will be saved.
   * @throws std::runtime_error if the file cannot be opened for writing.
   */
  void save(const std::string& file_path);

  /**
   * @brief Default virtual destructor.
   */
  virtual ~ConfigFile() = default;

  inline nlohmann::json& data() { return json_; }

 protected:
  nlohmann::json json_;
};

}  // namespace mllm
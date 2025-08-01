// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include <vector>
#include <string>
#include <sstream>
#include <fmt/base.h>
#include "mllm/utils/Common.hpp"

namespace mllm {

template<typename T>
class ArgArgument {
 public:
  T& get() { return static_cast<T&>(*this); }
};

class ArgumentBase {
 public:
  virtual ~ArgumentBase() = default;
  virtual void parse(const std::string& value) = 0;
  virtual void handleFlag() = 0;
  [[nodiscard]] virtual const std::vector<std::string>& flags() const = 0;
  [[nodiscard]] virtual bool isPositional() const = 0;
  [[nodiscard]] virtual bool isRequired() const = 0;
  [[nodiscard]] virtual bool isSet() const = 0;
  [[nodiscard]] virtual std::string help() const = 0;
  [[nodiscard]] virtual std::string meta() const = 0;
  [[nodiscard]] virtual bool isBoolean() const = 0;
  [[nodiscard]] virtual bool needsValue() const = 0;
};

template<typename T>
class Argument : public ArgumentBase, public ArgArgument<Argument<T>> {
 public:
  Argument& flag(const std::string& f) {
    flags_.push_back(f);
    return *this;
  }

  Argument& help(const std::string& text) {
    help_ = text;
    return *this;
  }

  Argument& required(bool val = true) {
    required_ = val;
    return *this;
  }

  Argument& positional(bool val = true) {
    positional_ = val;
    return *this;
  }

  Argument& meta(const std::string& m) {
    meta_ = m;
    return *this;
  }

  Argument& def(T val) {
    default_ = val;
    return *this;
  }

  void parse(const std::string& value) override {
    std::istringstream iss(value);
    if constexpr (std::is_same_v<T, bool>) {
      // Handle boolean values specially
      if (value == "true" || value == "1") {
        value_ = true;
      } else if (value == "false" || value == "0") {
        value_ = false;
      } else {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Invalid boolean value");
      }
    } else {
      if constexpr (std::is_same<T, std::string>()) {
        value_ = iss.str();
      } else {
        iss >> value_;
      }
      if (iss.fail()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Invalid value type"); }
    }
    is_set_ = true;
  }

  void handleFlag() override {
    if constexpr (std::is_same_v<T, bool>) {
      value_ = true;
      is_set_ = true;
    } else {
      throw std::runtime_error("Non-boolean flag not supported");
    }
  }

  [[nodiscard]] const std::vector<std::string>& flags() const override { return flags_; }

  [[nodiscard]] bool isPositional() const override { return positional_; }

  [[nodiscard]] bool isRequired() const override { return required_; }

  [[nodiscard]] bool isSet() const override { return is_set_; }

  [[nodiscard]] std::string help() const override { return help_; }

  [[nodiscard]] std::string meta() const override { return meta_; }

  [[nodiscard]] bool isBoolean() const override { return std::is_same_v<T, bool>; }

  [[nodiscard]] bool needsValue() const override { return !std::is_same_v<T, bool>; }

  T get() const {
    if (is_set_) return value_;
    return default_;
  }

 private:
  std::vector<std::string> flags_;
  std::string help_;
  std::string meta_;
  T value_;
  T default_;
  bool required_ = false;
  bool positional_ = false;
  bool is_set_ = false;
};

class Argparse {
 public:
  Argparse() = default;

  template<typename T>
  static Argument<T>& add(const std::string& flags) {
    auto& inst = instance();
    auto arg = std::make_unique<Argument<T>>();
    auto flag_list = splitFlags(flags);
    for (const auto& f : flag_list) { arg->flag(f); }
    if (flag_list.empty()) arg->positional();
    Argument<T>* ptr = arg.get();
    inst.args_.push_back(std::move(arg));
    return *ptr;
  }

  static void parse(int argc, char* argv[]) {
    auto& inst = instance();
    std::vector<std::string> args(argv + 1, argv + argc);
    std::vector<std::string> positional_args;

    // First parse options
    for (size_t i = 0; i < args.size(); ++i) {
      const auto& arg = args[i];
      if (arg[0] == '-') {
        bool found = false;
        for (auto& param : inst.args_) {
          const auto& flags = param->flags();
          if (std::find(flags.begin(), flags.end(), arg) != flags.end()) {
            if (param->isBoolean()) {
              param->handleFlag();
            } else {
              if (i + 1 >= args.size()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing value for {}", arg); }
              param->parse(args[++i]);
            }
            found = true;
            break;
          }
        }
        if (!found) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unknown option: {}", arg); }
      } else {
        positional_args.push_back(arg);
      }
    }

    // Then parse positional arguments
    size_t pos_idx = 0;
    for (auto& param : inst.args_) {
      if (param->isPositional()) {
        if (pos_idx >= positional_args.size()) {
          if (param->isRequired()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing positional argument"); }
          continue;
        }
        param->parse(positional_args[pos_idx++]);
      }
    }

    // Check requirements
    for (auto& param : inst.args_) {
      if (param->isRequired() && !param->isSet()) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing required argument: {}", param->meta());
      }
    }
  }

  template<typename T>
  static T get(const std::string& flag) {
    auto& inst = instance();
    for (auto& param : inst.args_) {
      const auto& flags = param->flags();
      if (std::find(flags.begin(), flags.end(), flag) != flags.end()) {
        auto* p = dynamic_cast<Argument<T>*>(param.get());
        if (p) { return p->get(); }
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Type mismatch for {}", flag);
      }
    }
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Argument not found: {}", flag);
  }

  static void printHelp() {
    auto& inst = instance();
    fmt::println("Usage:");
    for (const auto& arg : inst.args_) {
      if (arg->isPositional()) {
        fmt::print(" <{}>", arg->meta());
      } else {
        fmt::print(" [");
        for (size_t i = 0; i < arg->flags().size(); ++i) {
          if (i > 0) { fmt::print("|"); }
          fmt::print("{}", arg->flags()[i]);
        }
        fmt::print("]");
      }
    }
    fmt::println("\n\nOptions:");
    for (const auto& arg : inst.args_) {
      if (arg->isPositional()) {
        fmt::println("  <{}>\t{}", arg->meta(), arg->help());
      } else {
        fmt::print("  ");
        for (size_t i = 0; i < arg->flags().size(); ++i) {
          if (i > 0) { fmt::print(", "); }
          fmt::print("{}", arg->flags()[i]);
        }
        fmt::println("\t{}", arg->help());
      }
    }
  }

 private:
  std::vector<std::unique_ptr<ArgumentBase>> args_;

  static std::vector<std::string> splitFlags(const std::string& flags) {
    std::vector<std::string> result;

    size_t start = 0;
    size_t pos = 0;

    while ((pos = flags.find('|', start)) != std::string::npos) {
      std::string token = flags.substr(start, pos - start);
      if (!token.empty()) { result.push_back(token); }
      start = pos + 1;
    }
    std::string lastToken = flags.substr(start);
    if (!lastToken.empty()) { result.push_back(lastToken); }

    if (!flags.empty() && result.empty()) result.push_back(flags);
    return result;
  }

  static Argparse& instance() {
    static Argparse inst;
    return inst;
  }
};

}  // namespace mllm

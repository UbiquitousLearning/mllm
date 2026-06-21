// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <type_traits>
#include "mllm/utils/Common.hpp"

namespace mllm {

template<typename KeyT>
struct DefaultSymbolTableKeyFormatter {
  std::string operator()(const KeyT& key) const {
    if constexpr (std::is_same_v<KeyT, std::string>) {
      return key;
    } else if constexpr (std::is_arithmetic_v<KeyT>) {
      return std::to_string(key);
    } else {
      return "<unknown>";
    }
  }
};

template<typename KeyT, typename ValueT, typename Formatter = DefaultSymbolTableKeyFormatter<KeyT>>
class SymbolTable {
 public:
  using const_iterator = typename std::unordered_map<KeyT, ValueT>::const_iterator;

  void reg(const KeyT& name, const ValueT& v) {
    if (has(name)) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "When registering new value, found symbol table already has a value key: {}",
                      formatter_(name));
    }
    symbol_table_.insert({name, v});
  }

  bool has(const KeyT& name) { return symbol_table_.count(name); }

  void rename(const KeyT& name, const std::string& new_name) {
    if (!has(name)) {
      MLLM_WARN("When renaming symbol in symbol table. Found symbol key: {} not in symbol table", formatter_(name));
    }
    auto sec = symbol_table_[name];
    symbol_table_.erase(name);
    symbol_table_.insert({new_name, sec});
  }

  void remove(const KeyT& name) {
    if (!has(name)) {
      MLLM_WARN("When removing symbol in symbol table. Found symbol key: {} not in symbol table", formatter_(name));
    }
    symbol_table_.erase(name);
  }

  ValueT& operator[](const KeyT& name) {
    if (!has(name)) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "When accessing symbol in symbol table. Found symbol key: {} not in symbol table",
                      formatter_(name));
    }
    return symbol_table_[name];
  }

  std::unordered_map<KeyT, ValueT> _raw_data() const { return symbol_table_; }

  std::unordered_map<KeyT, ValueT>& _ref_raw_data() { return symbol_table_; }

  const_iterator begin() const { return symbol_table_.begin(); }
  const_iterator end() const { return symbol_table_.end(); }
  const_iterator cbegin() const { return symbol_table_.begin(); }
  const_iterator cend() const { return symbol_table_.end(); }

 private:
  std::unordered_map<KeyT, ValueT> symbol_table_;
  Formatter formatter_;
};

}  // namespace mllm

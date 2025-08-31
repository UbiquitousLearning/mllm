// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <concepts>
#include <utility>

/*
Usage:

auto result = redirect("output.txt", []() {
  std::cout << "This string will be write to file" << std::endl;
  std::cout << "current: " << __TIME__ << std::endl;
  return "Success";
});

auto result = redirect(custom_ss, []() {
  std::cout << "code" << std::endl
  return 42;
});

 *
 */

namespace mllm {

template<typename T>
concept OutputStream = requires(T stream) {
  { stream.rdbuf() } -> std::convertible_to<std::streambuf*>;
  {
    stream << "test(this message only for compiler, will not appear in the real productive env)"
  } -> std::same_as<std::remove_reference_t<T>&>;
};

class ScopedRedirect {
 private:
  std::streambuf* original_buf;
  bool active = true;

 public:
  explicit ScopedRedirect(std::ostream& target) : original_buf(std::cout.rdbuf()) { std::cout.rdbuf(target.rdbuf()); }

  ~ScopedRedirect() {
    if (active) { std::cout.rdbuf(original_buf); }
  }

  ScopedRedirect(const ScopedRedirect&) = delete;

  ScopedRedirect& operator=(const ScopedRedirect&) = delete;

  ScopedRedirect(ScopedRedirect&& other) noexcept : original_buf(other.original_buf), active(other.active) {
    other.active = false;
  }

  ScopedRedirect& operator=(ScopedRedirect&& other) noexcept {
    if (this != &other) {
      original_buf = other.original_buf;
      active = other.active;
      other.active = false;
    }
    return *this;
  }

  template<OutputStream Stream, typename Callable>
  decltype(auto) redirect(Stream& stream, Callable&& callable) {
    ScopedRedirect redirector(stream);
    return std::forward<Callable>(callable)();
  }

  template<typename Callable>
  auto redirect(Callable&& callable) {
    std::stringstream ss;
    auto result = redirect(ss, std::forward<Callable>(callable));
    return std::make_pair(std::move(result), ss.str());
  }

  template<typename Callable>
  auto redirect(const std::string& filename, Callable&& callable) {
    std::ofstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Can't Open File: " + filename); }

    auto result = redirect(file, std::forward<Callable>(callable));
    return result;
  }
};

}  // namespace mllm

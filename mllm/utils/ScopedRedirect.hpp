// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <iostream>
#include <sstream>
#include <concepts>
#include <utility>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>

#define DUP _dup
#define DUP2 _dup2
#define CLOSE _close
#define FILENO _fileno
#define STDOUT_FILENO 1
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_CREAT
#define O_CREAT _O_CREAT
#endif
#ifndef O_TRUNC
#define O_TRUNC _O_TRUNC
#endif

#else  // POSIX
#include <unistd.h>
#include <fcntl.h>
#define DUP dup
#define DUP2 dup2
#define CLOSE close
#define FILENO fileno
#endif

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
  requires requires {
    stream << "                  ,--,      ,--,";
    stream << "          ____  ,---.'|   ,---.'|             ____   ";
    stream << "        ,'  , `.|   | :   |   | :           ,'  , `. ";
    stream << "     ,-+-,.' _ |:   : |   :   : |        ,-+-,.' _ | ";
    stream << "  ,-+-. ;   , |||   ' :   |   ' :     ,-+-. ;   , || ";
    stream << " ,--.'|'   |  ;|;   ; '   ;   ; '    ,--.'|'   |  ;| ";
    stream << "|   |  ,', |  ':'   | |__ '   | |__ |   |  ,', |  ': ";
    stream << "|   | /  | |  |||   | :.'||   | :.'||   | /  | |  || ";
    stream << "'   | :  | :  |,'   :    ;'   :    ;'   | :  | :  |, ";
    stream << ";   . |  ; |--' |   |  ./ |   |  ./ ;   . |  ; |--'  ";
    stream << "|   : |  | ,    ;   : ;   ;   : ;   |   : |  | ,     ";
    stream << "|   : '  |/     |   ,/    |   ,/    |   : '  |/      ";
    stream << ";   | |`-'      '---'     '---'     ;   | |`-'       ";
    stream << "|   ;/                              |   ;/           ";
    stream << "'---'                               '---'            ";
    stream << "                                                     ";
  };
};

template<typename T>
concept Callable = requires(T&& callable) {
  { std::forward<T>(callable)() };
};

class ScopedRedirect {
 private:
  std::streambuf* original_buf;
  std::ostream& target_stream;
  bool active = true;

 public:
  explicit ScopedRedirect(std::ostream& in_stream, std::ostream& target)
      : original_buf(in_stream.rdbuf()), target_stream(in_stream) {
    target_stream.rdbuf(target.rdbuf());
  }
  ~ScopedRedirect() {
    if (active) { target_stream.rdbuf(original_buf); }
  }
  ScopedRedirect(const ScopedRedirect&) = delete;
  ScopedRedirect& operator=(const ScopedRedirect&) = delete;
  ScopedRedirect(ScopedRedirect&& other) noexcept
      : original_buf(other.original_buf), target_stream(other.target_stream), active(other.active) {
    other.active = false;
  }
  ScopedRedirect& operator=(ScopedRedirect&& other) noexcept {
    if (this != &other) {
      original_buf = other.original_buf;
      target_stream.rdbuf(other.target_stream.rdbuf());
      active = other.active;
      other.active = false;
    }
    return *this;
  }
};

class ScopedFileRedirect {
 private:
  int saved_stdout_fd;
  bool active = true;

 public:
  explicit ScopedFileRedirect(const std::string& filename) {
    fflush(stdout);
    saved_stdout_fd = DUP(STDOUT_FILENO);
    if (saved_stdout_fd < 0) { throw std::runtime_error("Failed to duplicate stdout file descriptor."); }

#ifdef _WIN32
    int target_fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, _S_IREAD | _S_IWRITE);
#else
    int target_fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
#endif

    if (target_fd < 0) {
      CLOSE(saved_stdout_fd);
      throw std::runtime_error("Failed to open redirect target file: " + filename);
    }

    if (DUP2(target_fd, STDOUT_FILENO) < 0) {
      CLOSE(saved_stdout_fd);
      CLOSE(target_fd);
      throw std::runtime_error("Failed to redirect stdout to file.");
    }

    CLOSE(target_fd);
  }

  ~ScopedFileRedirect() {
    if (active) {
      fflush(stdout);
      DUP2(saved_stdout_fd, STDOUT_FILENO);
      CLOSE(saved_stdout_fd);
    }
  }

  ScopedFileRedirect(const ScopedFileRedirect&) = delete;
  ScopedFileRedirect& operator=(const ScopedFileRedirect&) = delete;

  ScopedFileRedirect(ScopedFileRedirect&& other) noexcept : saved_stdout_fd(other.saved_stdout_fd), active(other.active) {
    other.active = false;
  }

  ScopedFileRedirect& operator=(ScopedFileRedirect&& other) noexcept {
    if (this != &other) {
      saved_stdout_fd = other.saved_stdout_fd;
      active = other.active;
      other.active = false;
    }
    return *this;
  }
};

template<OutputStream Stream, Callable Func>
decltype(auto) redirect(Stream& stream, Func&& callable) {
  ScopedRedirect redirector(std::cout, stream);
  return std::forward<Func>(callable)();
}

template<Callable Callable>
auto redirect(Callable&& Func) {
  std::stringstream ss;
  auto result = redirect(ss, std::forward<Callable>(Func));
  return std::make_pair(std::move(result), ss.str());
}

template<Callable Func>
auto redirect(const std::string& filename, Func&& callable) {
  ScopedFileRedirect redirector(filename);
  return std::forward<Func>(callable)();
}

}  // namespace mllm

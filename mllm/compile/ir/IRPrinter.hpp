// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <fmt/base.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace mllm::ir {

class IRPrinter {
 public:
  void newline(bool with_indent = true) {
    fmt::print("\n");
    if (with_indent) printIndent();
  }
  inline void lbrace() {
    fmt::print("{}", "{");
    increaseIndent();
    newline();
  }
  inline void rbrace() {
    decreaseIndent();
    newline();
    fmt::print("{}", "}");
  }

  static void langle() { fmt::print("<"); }
  static void rangle() { fmt::print(">"); }
  static void assign() { fmt::print(" = "); }
  static void lparentheses() { fmt::print("("); }
  static void rparentheses() { fmt::print(")"); }
  static void comma() { fmt::print(", "); }
  static void to() { fmt::print(" -> "); }
  static void lsbracket() { fmt::print("["); }
  static void rsbracket() { fmt::print("]"); }

  static void colon() { fmt::print(":"); }

  static void blank() { fmt::print(" "); }

  template<typename FirstArg>
  inline void print(FirstArg&& first_arg) {
    fmt::print(fmt::runtime(std::forward<FirstArg>(first_arg)));
  }

  template<typename FirstArg, typename... Args>
  inline void print(FirstArg&& first_arg, Args&&... args) {
    fmt::print(fmt::runtime(std::forward<FirstArg>(first_arg)), std::forward<Args>(args)...);
  }

  template<typename... Args>
  inline void comment(Args&&... args) {
    fmt::print("// ");
    fmt::print(std::forward<Args>(args)...);
    newline();
  }

 public:
  inline void increaseIndent() { m_indent++; }
  inline void decreaseIndent() { m_indent--; }

 private:
  inline void printIndent() const {
    for (size_t i = 0; i < m_indent * 4; ++i) fmt::print(" ");
  }

 private:
  size_t m_indent = 0;
};

struct IRPrinterGuard {
  ~IRPrinterGuard() { m_printer.decreaseIndent(); }
  explicit IRPrinterGuard(IRPrinter& printer) : m_printer(printer) { printer.increaseIndent(); }

 private:
  IRPrinter& m_printer;
};
}  // namespace mllm::ir

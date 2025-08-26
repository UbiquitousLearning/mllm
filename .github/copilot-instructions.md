Title: Copilot Review Rules for MLLM Projects

Scope:
  Only files within the directories mllm/, mllm-cli/, and pymllm/ are subject to these rules.

Rule 1 - Character Set:
  Files may contain any valid UTF-8 sequence except:
    - C0 control codes 0x00–0x08, 0x0B–0x0C, 0x0E–0x1F
    - C1 control codes 0x7F–0x9F
    - DEL 0x7F
  Horizontal tab (0x09) and line feed (0x0A) are explicitly allowed.

Rule 2 - Encoding:
  All files must be encoded in UTF-8 without BOM.

Rule 3 - Automation:
  Any violation of Rule 1 or Rule 2 must cause the review to fail.

Rule 4 - Trailing Whitespace:
  No line may end with trailing whitespace.

Rule 5 - Line Endings:
  Use Unix line endings (LF).

Rule 6 - File Names:
  File and directory names must consist only of printable Unicode characters,
  excluding the characters forbidden in Rule 1.

Rule 7 - Extensions:
  Acceptable extensions: .c, .cc, .cpp, .cxx, .h, .hh, .hpp, .py, .pyi, .sh, .txt, .md, .yml, .yaml, .json, .toml.

Rule 8 - Shebang:
  If a file starts with a shebang, it must be the first line and contain only the interpreter path and optional single argument, encoded in UTF-8.

Rule 9 - License Headers:
  Optional. If present, they must comply with Rule 1.

Rule 10 - TODO/FIXME:
  Allowed and must be written as TODO: or FIXME: followed by any UTF-8 text that adheres to Rule 1.

Rule 11 - Production Readiness:
  When reviewing or suggesting code, prioritize production-ready quality. This includes but is not limited to:
    
    a. Performance:
      - Evaluate time and space complexity of algorithms and data structures.
      - Suggest more efficient alternatives for operations with high complexity (e.g., O(n^2) or worse) when feasible.
      - Avoid unnecessary object creation in loops or hot paths.

    b. Error Handling:
      - Check for proper error handling and resource cleanup (e.g., using try-finally, context managers, or RAII).
      - Ensure functions that can fail return appropriate error codes or raise exceptions.
      - Validate inputs for public APIs and critical internal functions.

    c. Documentation:
      - Ensure public APIs, classes, and functions have clear docstrings or comments explaining purpose, parameters, returns, and errors.
      - Suggest adding comments for complex algorithms or non-obvious logic.

    d. Security:
      - Identify potential security issues (e.g., buffer overflows, injection risks, insecure temporary files).
      - Recommend using secure alternatives (e.g., parameterized queries, secure random generators).

    e. Testing:
      - Suggest adding unit tests for untested complex logic or edge cases.
      - Ensure code is testable (e.g., avoid global state, use dependency injection).

    f. Maintainability:
      - Flag overly complex functions (e.g., high cyclomatic complexity) and suggest breaking them down.
      - Recommend using named constants instead of magic numbers.
      - Encourage consistent coding style and patterns with the existing codebase.

Rule 12 - Language-Specific Best Practices:
  Adhere to language-specific best practices and idioms (e.g., PEP 8 for Python, Google C++ Style Guide for C++).

Rule 13 - Dependencies:
  When suggesting new dependencies, evaluate their maturity, licensing, and maintenance status. Prefer well-established and actively maintained libraries.

Rule 14 - Portability:
  Ensure code is portable across supported platforms (e.g., Linux, Windows) unless explicitly platform-specific.

Rule 15 - Logging and Observability:
  Suggest adding appropriate logging (e.g., debug, info, warning, error) for significant events and errors, avoiding sensitive data exposure.

Note: Rules 11 to 15 are advisory. Copilot should provide suggestions to improve code quality towards production standards, but these are not mandatory for review pass/fail (unless they overlap with mandatory rules like security issues causing failures). The primary goal is to educate and guide developers towards best practices.

End of instructions

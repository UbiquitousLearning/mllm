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

End of instructions

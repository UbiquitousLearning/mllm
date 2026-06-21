"""Tool-call (function-calling) output parser.

Extracts structured tool calls from model output text.  Supports both
one-shot and incremental streaming modes.

Formats supported:

* **qwen25** — ``<tool_call>{"name":...,"arguments":...}</tool_call>``
* **llama3** — ``<|python_tag|>{"name":...,"parameters":...}``
* **hermes** — ``<tool_call>{"name":...,"arguments":...}</tool_call>`` (same tags, Hermes schema)

Usage::

    # Non-streaming
    parser = ToolCallParser("qwen25", tools=tools_list)
    content, tool_calls = parser.parse_non_stream(full_text)

    # Streaming
    parser = ToolCallParser("qwen25", tools=tools_list)
    for delta in deltas:
        content_delta, tool_call_deltas = parser.parse_stream_chunk(delta)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolCallItem:
    """A single parsed tool call."""

    name: Optional[str] = None
    arguments: str = ""
    tool_call_id: str = ""
    index: int = 0

    def to_openai_dict(self, streaming: bool = True) -> Dict[str, Any]:
        """Convert to OpenAI ``tool_calls[]`` element format.

        Parameters
        ----------
        streaming
            If True, include ``index`` (streaming delta format).
            If False, omit ``index`` (non-streaming message format).
        """
        d: Dict[str, Any] = {"type": "function", "function": {}}
        if streaming:
            d["index"] = self.index
        if self.tool_call_id:
            d["id"] = self.tool_call_id
        fn: Dict[str, Any] = d["function"]
        if self.name is not None:
            fn["name"] = self.name
        fn["arguments"] = self.arguments or ""
        return d


# ---------------------------------------------------------------------------
# Detector base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FormatConfig:
    bot_token: str
    end_token: str
    # Regex to extract individual call bodies between bot/end tokens.
    # If None, the entire text between bot and end tokens is one call.
    call_regex: Optional[str] = None


_FORMAT_MAP: Dict[str, _FormatConfig] = {
    "qwen25": _FormatConfig(
        bot_token="<tool_call>\n",
        end_token="\n</tool_call>",
    ),
    "qwen3_coder": _FormatConfig(
        bot_token="<tool_call>",
        end_token="</tool_call>",
    ),
    "hermes": _FormatConfig(
        bot_token="<tool_call>\n",
        end_token="\n</tool_call>",
    ),
    "llama3": _FormatConfig(
        bot_token="<|python_tag|>",
        end_token="",  # Llama3 uses EOT, detected via EOS
    ),
}


# ---------------------------------------------------------------------------
# ToolCallParser
# ---------------------------------------------------------------------------


class ToolCallParser:
    """Model-agnostic tool-call parser.

    Parameters
    ----------
    model_type
        Key into the format registry (e.g. ``"qwen25"``, ``"llama3"``).
    tools
        The ``tools`` list from the OpenAI chat request (used to resolve
        function names).
    """

    SUPPORTED = set(_FORMAT_MAP)

    def __init__(self, model_type: str, tools: Optional[List[Any]] = None):
        cfg = _FORMAT_MAP.get(model_type)
        if cfg is None:
            raise ValueError(
                f"Unknown tool-call parser {model_type!r}. "
                f"Supported: {sorted(_FORMAT_MAP)}"
            )
        self._bot = cfg.bot_token
        self._end = cfg.end_token
        self._model_type = model_type
        self._tools = tools or []

        # -- streaming state --
        self._buffer = ""
        self._in_call = False
        self._current_tool_idx = 0
        self._current_call_buf = ""
        self._prev_args_len = 0
        self._name_sent = False
        self._completed_calls: List[ToolCallItem] = []

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #

    def has_tool_call(self, text: str) -> bool:
        """Return True if *text* contains a tool-call pattern."""
        return self._bot in text

    def parse_non_stream(
        self, text: str
    ) -> Tuple[str, List[ToolCallItem]]:
        """Parse complete text.

        Returns ``(remaining_content, tool_calls)``.
        """
        if not self.has_tool_call(text):
            return text, []

        tool_calls: List[ToolCallItem] = []
        normal_parts: List[str] = []

        remaining = text
        idx = 0
        while True:
            bot_pos = remaining.find(self._bot)
            if bot_pos == -1:
                normal_parts.append(remaining)
                break
            normal_parts.append(remaining[:bot_pos])
            remaining = remaining[bot_pos + len(self._bot) :]

            if self._end:
                end_pos = remaining.find(self._end)
                if end_pos == -1:
                    call_body = remaining
                    remaining = ""
                else:
                    call_body = remaining[:end_pos]
                    remaining = remaining[end_pos + len(self._end) :]
            else:
                call_body = remaining
                remaining = ""

            parsed = self._parse_call_body(call_body.strip())
            if parsed is not None:
                parsed.index = idx
                parsed.tool_call_id = _make_tool_call_id()
                tool_calls.append(parsed)
                idx += 1

        content = "".join(normal_parts).strip()
        return content, tool_calls

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def parse_stream_chunk(
        self, delta: str
    ) -> Tuple[str, List[ToolCallItem]]:
        """Parse an incremental streaming delta.

        Returns ``(content_delta, tool_call_items)``.

        For tool call items:
        - First item for a call: ``name`` is set, ``arguments`` is ``""``.
        - Subsequent items: ``name`` is ``None``, ``arguments`` is the new
          characters appended (argument delta).
        """
        if not delta:
            return "", []

        self._buffer += delta
        content_out = ""
        items: List[ToolCallItem] = []

        while True:
            if not self._in_call:
                # --- look for bot token ---
                bot_pos = self._buffer.find(self._bot)
                if bot_pos != -1:
                    content_out += self._buffer[:bot_pos]
                    self._buffer = self._buffer[bot_pos + len(self._bot) :]
                    self._in_call = True
                    self._current_call_buf = ""
                    self._prev_args_len = 0
                    self._name_sent = False
                    continue  # try to process call content
                else:
                    # Check for partial bot token at tail
                    if self._bot and _could_be_partial(self._buffer, self._bot):
                        safe = len(self._buffer) - len(self._bot) + 1
                        if safe > 0:
                            content_out += self._buffer[:safe]
                            self._buffer = self._buffer[safe:]
                    else:
                        content_out += self._buffer
                        self._buffer = ""
                    break

            if self._in_call:
                # --- look for end token ---
                if self._end:
                    end_pos = self._buffer.find(self._end)
                    if end_pos != -1:
                        self._current_call_buf += self._buffer[:end_pos]
                        self._buffer = self._buffer[end_pos + len(self._end) :]
                        # Emit final tool call
                        item = self._finalize_call()
                        if item is not None:
                            items.append(item)
                        self._in_call = False
                        self._current_tool_idx += 1
                        continue  # there may be more calls
                    else:
                        # Accumulate and stream arguments
                        self._current_call_buf += self._buffer
                        self._buffer = ""
                        item = self._stream_partial_call()
                        if item is not None:
                            items.append(item)
                        break
                else:
                    # No end token (e.g. Llama3) — accumulate everything
                    self._current_call_buf += self._buffer
                    self._buffer = ""
                    item = self._stream_partial_call()
                    if item is not None:
                        items.append(item)
                    break

        return content_out, items

    def flush(self) -> List[ToolCallItem]:
        """Flush any remaining buffered tool call (call at request end)."""
        items: List[ToolCallItem] = []
        if self._in_call and self._current_call_buf.strip():
            item = self._finalize_call()
            if item is not None:
                items.append(item)
            self._in_call = False
        return items

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_call_body(self, body: str) -> Optional[ToolCallItem]:
        """Parse a single call body (JSON or qwen3_coder XML-style)."""
        if self._model_type == "qwen3_coder":
            return self._parse_qwen3_coder_body(body)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        name = obj.get("name")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        return ToolCallItem(name=name, arguments=args)

    @staticmethod
    def _parse_qwen3_coder_body(body: str) -> Optional[ToolCallItem]:
        """Parse qwen3_coder XML-style: ``<function=NAME><parameter=K>V</parameter>...</function>``."""
        # Extract function name
        func_m = re.search(r"<function=([^>]+)>", body)
        if func_m is None:
            return None
        name = func_m.group(1)
        # Extract parameters
        params: Dict[str, Any] = {}
        for pm in re.finditer(
            r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>))",
            body,
            re.DOTALL,
        ):
            key = pm.group(1)
            val = pm.group(2).strip()
            # Try to parse as JSON value, otherwise keep as string
            try:
                params[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                params[key] = val
        return ToolCallItem(
            name=name,
            arguments=json.dumps(params, ensure_ascii=False),
        )

    def _stream_partial_call(self) -> Optional[ToolCallItem]:
        """Try to extract streaming information from the partial call."""
        body = self._current_call_buf.strip()
        if not body:
            return None

        # Try to extract name first
        if not self._name_sent:
            name = self._try_extract_name(body)
            if name is not None:
                self._name_sent = True
                return ToolCallItem(
                    name=name,
                    arguments="",
                    tool_call_id=_make_tool_call_id(),
                    index=self._current_tool_idx,
                )
            return None

        # Stream argument characters
        args_str = self._try_extract_args_partial(body)
        if args_str is not None and len(args_str) > self._prev_args_len:
            new_chars = args_str[self._prev_args_len :]
            self._prev_args_len = len(args_str)
            return ToolCallItem(
                name=None,
                arguments=new_chars,
                index=self._current_tool_idx,
            )
        return None

    def _finalize_call(self) -> Optional[ToolCallItem]:
        """Finalize a complete call — emit any remaining argument chars."""
        parsed = self._parse_call_body(self._current_call_buf.strip())
        if parsed is None:
            return None

        if not self._name_sent:
            # Entire call came at once
            parsed.index = self._current_tool_idx
            parsed.tool_call_id = _make_tool_call_id()
            return parsed

        # Name was already sent — emit remaining arguments
        full_args = parsed.arguments
        new_chars = full_args[self._prev_args_len :]
        if new_chars:
            return ToolCallItem(
                name=None,
                arguments=new_chars,
                index=self._current_tool_idx,
            )
        return None

    def _try_extract_name(self, partial: str) -> Optional[str]:
        """Try to extract function name from partial call body."""
        if self._model_type == "qwen3_coder":
            m = re.search(r"<function=([^>]+)>", partial)
            return m.group(1) if m else None
        m = re.search(r'"name"\s*:\s*"([^"]+)"', partial)
        return m.group(1) if m else None

    def _try_extract_args_partial(self, partial: str) -> Optional[str]:
        """Try to extract partial arguments from call body."""
        if self._model_type == "qwen3_coder":
            # Build JSON incrementally from <parameter=K>V</parameter> tags
            params: Dict[str, Any] = {}
            for pm in re.finditer(
                r"<parameter=([^>]+)>(.*?)(?:</parameter>)",
                partial,
                re.DOTALL,
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    params[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    params[key] = val
            if params:
                return json.dumps(params, ensure_ascii=False)
            return None
        m = re.search(r'"arguments"\s*:\s*(\{.*)', partial, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r'"parameters"\s*:\s*(\{.*)', partial, re.DOTALL)
        if m:
            return m.group(1)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _could_be_partial(text: str, pattern: str) -> bool:
    for i in range(1, len(pattern)):
        if text.endswith(pattern[:i]):
            return True
    return False

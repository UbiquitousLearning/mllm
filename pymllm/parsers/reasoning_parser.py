"""Reasoning / thinking content parser.

Separates ``<think>...</think>`` (or model-specific markers) from normal
assistant content.  Supports both one-shot and incremental streaming modes.

Usage::

    # Non-streaming
    parser = ReasoningParser("qwen3")
    reasoning, content = parser.parse_non_stream(full_text)

    # Streaming
    parser = ReasoningParser("qwen3")
    for delta in deltas:
        reasoning_delta, content_delta = parser.parse_stream_chunk(delta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DetectorConfig:
    start: str
    end: str
    force: bool  # True = always assume reasoning at start


_DETECTOR_MAP: Dict[str, _DetectorConfig] = {
    # DeepSeek-R1: always starts in reasoning mode
    "deepseek-r1": _DetectorConfig("<think>", "</think>", force=True),
    # Qwen3: optional thinking (controlled by request)
    "qwen3": _DetectorConfig("<think>", "</think>", force=False),
    # Qwen3 forced thinking
    "qwen3-thinking": _DetectorConfig("<think>", "</think>", force=True),
    # GLM-4.5
    "glm45": _DetectorConfig("<think>", "</think>", force=False),
    # Kimi
    "kimi": _DetectorConfig("\u25c1think\u25b7", "\u25c1/think\u25b7", force=False),
}


# ---------------------------------------------------------------------------
# ReasoningParser
# ---------------------------------------------------------------------------


class ReasoningParser:
    """Model-agnostic reasoning content parser.

    Parameters
    ----------
    model_type
        Key into the detector registry (e.g. ``"qwen3"``, ``"deepseek-r1"``).
    stream_reasoning
        If ``True``, stream reasoning content incrementally as it arrives.
        If ``False``, buffer reasoning until the end tag is found.
    """

    SUPPORTED = set(_DETECTOR_MAP)

    def __init__(self, model_type: str, stream_reasoning: bool = True):
        cfg = _DETECTOR_MAP.get(model_type)
        if cfg is None:
            raise ValueError(
                f"Unknown reasoning parser {model_type!r}. "
                f"Supported: {sorted(_DETECTOR_MAP)}"
            )
        self._start = cfg.start
        self._end = cfg.end
        self._force = cfg.force
        self._stream_reasoning = stream_reasoning

        # -- streaming state --
        self._buffer = ""
        self._in_reasoning = cfg.force
        self._start_consumed = False  # True once start tag has been stripped
        self._done = False  # True once end tag has been seen

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #

    def parse_non_stream(self, text: str) -> Tuple[Optional[str], str]:
        """Parse complete text.

        Returns ``(reasoning_content, content)`` where either may be empty.
        """
        start_idx = text.find(self._start)
        end_idx = text.find(self._end)

        if start_idx == -1 and not self._force:
            return None, text

        # Determine boundaries
        if self._force and start_idx == -1:
            # Model didn't emit explicit start tag; treat prefix as reasoning
            reason_start = 0
        else:
            reason_start = start_idx + len(self._start)

        before = text[:start_idx] if start_idx != -1 else ""

        if end_idx != -1 and end_idx >= reason_start:
            reasoning = text[reason_start:end_idx]
            after = text[end_idx + len(self._end) :]
        else:
            reasoning = text[reason_start:]
            after = ""

        content = (before + after).strip()
        reasoning = reasoning.strip()
        return reasoning or None, content

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def parse_stream_chunk(self, delta: str) -> Tuple[str, str]:
        """Parse an incremental streaming delta.

        Returns ``(reasoning_delta, content_delta)``.  Either may be ``""``.
        """
        if not delta:
            return "", ""

        if self._done:
            return "", delta

        self._buffer += delta
        reasoning_out = ""
        content_out = ""

        # In forced reasoning mode, consume the start tag if it appears
        # (the model may or may not emit it explicitly).
        if self._in_reasoning and not self._start_consumed:
            idx = self._buffer.find(self._start)
            if idx != -1:
                # Start tag found — strip it and any text before it
                self._buffer = self._buffer[idx + len(self._start) :]
                self._start_consumed = True
            elif _could_be_partial(self._buffer, self._start):
                # Might be a partial start tag — hold the buffer
                return "", ""
            else:
                # No start tag coming — mark consumed and continue
                self._start_consumed = True

        if not self._in_reasoning:
            # --- look for start tag ---
            idx = self._buffer.find(self._start)
            if idx != -1:
                content_out += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self._start) :]
                self._in_reasoning = True
                self._start_consumed = True
            elif _could_be_partial(self._buffer, self._start):
                # Potential partial match at tail — hold the buffer
                safe = len(self._buffer) - len(self._start) + 1
                if safe > 0:
                    content_out += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                return "", content_out
            else:
                content_out += self._buffer
                self._buffer = ""
                return "", content_out

        if self._in_reasoning:
            # --- look for end tag ---
            idx = self._buffer.find(self._end)
            if idx != -1:
                reasoning_out += self._buffer[:idx]
                after = self._buffer[idx + len(self._end) :]
                self._buffer = ""
                self._in_reasoning = False
                self._done = True
                if after:
                    content_out += after
            elif _could_be_partial(self._buffer, self._end):
                safe = len(self._buffer) - len(self._end) + 1
                if safe > 0:
                    reasoning_out += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
            else:
                reasoning_out += self._buffer
                self._buffer = ""

        if not self._stream_reasoning:
            reasoning_out = ""

        return reasoning_out, content_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _could_be_partial(text: str, pattern: str) -> bool:
    """Return True if *text* ends with a prefix of *pattern*."""
    for i in range(1, len(pattern)):
        if text.endswith(pattern[:i]):
            return True
    return False

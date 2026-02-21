from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class BaseReq:
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)

    def regenerate_rid(self) -> Union[str, List[str]]:
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class BaseBatchReq:
    rids: List[str]

    def regenerate_rids(self) -> List[str]:
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class GenerateReqInput(BaseReq):
    text: Optional[Union[List[str], str]] = None
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    return_logprob: Optional[Union[List[bool], bool]] = None
    logprob_start_len: Optional[Union[List[int], int]] = None
    top_logprobs_num: Optional[Union[List[int], int]] = None
    stream: bool = False

    # Multimodal placeholders.
    image_data: Optional[Any] = None
    video_data: Optional[Any] = None
    audio_data: Optional[Any] = None

    # Runtime extension placeholders.
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    session_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    # Derived fields populated by normalization.
    is_single: bool = field(default=True, init=False)
    batch_size: int = field(default=1, init=False)

    def normalize_batch_and_arguments(self) -> None:
        self._validate_inputs()
        self._determine_batch_size()

    def _validate_inputs(self) -> None:
        has_text = self.text is not None
        has_input_ids = self.input_ids is not None
        if has_text == has_input_ids:
            raise ValueError("Exactly one of `text` or `input_ids` must be provided.")

    def _determine_batch_size(self) -> None:
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                if len(self.text) == 0:
                    raise ValueError("`text` cannot be an empty list.")
                self.is_single = False
                self.batch_size = len(self.text)
            return

        assert self.input_ids is not None
        if len(self.input_ids) == 0:
            raise ValueError("`input_ids` cannot be empty.")
        if isinstance(self.input_ids[0], int):
            self.is_single = True
            self.batch_size = 1
        else:
            self.is_single = False
            self.batch_size = len(self.input_ids)

    def __getitem__(self, i: int) -> "GenerateReqInput":
        if i < 0 or i >= self.batch_size:
            raise IndexError(f"index {i} out of range for batch size {self.batch_size}")
        if self.batch_size == 1:
            return self
        return GenerateReqInput(
            rid=self._pick(self.rid, i),
            text=self._pick(self.text, i),
            input_ids=self._pick(self.input_ids, i),
            sampling_params=self._pick(self.sampling_params, i),
            return_logprob=self._pick(self.return_logprob, i),
            logprob_start_len=self._pick(self.logprob_start_len, i),
            top_logprobs_num=self._pick(self.top_logprobs_num, i),
            stream=self.stream,
            image_data=self._pick(self.image_data, i),
            video_data=self._pick(self.video_data, i),
            audio_data=self._pick(self.audio_data, i),
            lora_path=self._pick(self.lora_path, i),
            session_params=self._pick(self.session_params, i),
            extra_options=self.extra_options.copy(),
        )

    @staticmethod
    def _pick(value: Any, i: int) -> Any:
        if isinstance(value, list):
            return value[i]
        return value

    def to_request_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in {
            "rid": self.rid,
            "text": self.text,
            "input_ids": self.input_ids,
            "sampling_params": self.sampling_params,
            "return_logprob": self.return_logprob,
            "logprob_start_len": self.logprob_start_len,
            "top_logprobs_num": self.top_logprobs_num,
            "stream": self.stream,
            "image_data": self.image_data,
            "video_data": self.video_data,
            "audio_data": self.audio_data,
            "lora_path": self.lora_path,
            "session_params": self.session_params,
        }.items():
            if value is not None:
                payload[key] = value
        payload.update(self.extra_options)
        return payload


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    input_text: str = ""
    input_ids: List[int] = field(default_factory=list)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    return_logprob: bool = False
    logprob_start_len: int = -1
    top_logprobs_num: int = 0
    lora_path: Optional[str] = None
    session_params: Optional[Dict[str, Any]] = None


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    reqs: List[TokenizedGenerateReqInput]

    def __len__(self) -> int:
        return len(self.reqs)

    def __getitem__(self, i: int) -> TokenizedGenerateReqInput:
        return self.reqs[i]

    def __iter__(self) -> Iterator[TokenizedGenerateReqInput]:
        return iter(self.reqs)


@dataclass
class BatchTokenIDOutput(BaseBatchReq):
    finished_reasons: List[Optional[str]]
    decode_ids: List[int]
    read_offsets: List[int]
    output_ids: Optional[List[int]]
    skip_special_tokens: List[bool]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    input_token_logprobs_val: List[float] = field(default_factory=list)
    input_token_logprobs_idx: List[int] = field(default_factory=list)
    output_token_logprobs_val: List[float] = field(default_factory=list)
    output_token_logprobs_idx: List[int] = field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = field(default_factory=list)


@dataclass
class BatchStrOutput(BaseBatchReq):
    finished_reasons: List[Optional[str]]
    output_strs: List[str]
    output_ids: Optional[List[int]]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    input_token_logprobs_val: List[float] = field(default_factory=list)
    input_token_logprobs_idx: List[int] = field(default_factory=list)
    output_token_logprobs_val: List[float] = field(default_factory=list)
    output_token_logprobs_idx: List[int] = field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = field(default_factory=list)

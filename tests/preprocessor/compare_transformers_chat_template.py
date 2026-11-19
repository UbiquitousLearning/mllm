#!/usr/bin/env python3
"""Chat-template parity gate against Hugging Face Transformers.

Stage A (renderer): render pinned cases with Mllm-ChatTemplate-Render and
compare rendered bytes, Transformers token ids, and mllm tokenizer ids.

Stage B (product path): run Mllm-ChatTemplate-Probe, which goes through the
model tokenizer's convertMessage with the builtin and the jinja_required
backends, and compare both against Transformers apply_chat_template ids.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from transformers import AutoTokenizer


MODELS = {
    "qwen3_5": {
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "template_sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config_sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    },
    "minicpm5": {
        "revision": "4e9de7a0778dc1c362e983e6858f0e77542cbdca",
        "template_sha256": "7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c",
        "tokenizer_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
        "tokenizer_config_sha256": "094efb3cf1ff412284cc5945fc99dff58673a912760d04483a04aa1c716f66fd",
    },
    # Qwen3 dense checkpoints ship the template inside tokenizer_config.json.
    # The runner sends enable_thinking=false; the official template emits the
    # empty thinking block for that value.
    "qwen3": {
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "oracle": "Qwen/Qwen3-1.7B",
        "template_file": "tokenizer_config.json",
        "template_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "product_thinking": False,
    },
    # Qwen3-MoE and the Qwen3 Ascend runner reuse the Qwen3 dense template and
    # tokenizer as the oracle; the MoE checkpoint itself is not pinned here.
    "qwen3_moe": {
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "oracle": "Qwen/Qwen3-1.7B (family template)",
        "template_file": "tokenizer_config.json",
        "template_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "product_thinking": None,
    },
    "qwen_ascend": {
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "oracle": "Qwen/Qwen3-1.7B (family template)",
        "template_file": "tokenizer_config.json",
        "template_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "product_thinking": False,
    },
}

MINICPM5_2B = {
    "revision": "0e9c66dce9fedde5ba8663bbcdd54b6810bb929a",
    "template_sha256": "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c",
    "tokenizer_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multiturn_cases():
    # Test text-only structured requests separately from the single-turn runner
    # interfaces. The omitted flag must stay absent: explicit False is not the
    # same as undefined in the MiniCPM5 templates.
    system = {"role": "system", "content": "请简洁回答。"}
    first_user = {"role": "user", "content": "  先介绍杭州。  "}
    plain = {"role": "assistant", "content": "杭州是浙江的省会。"}
    inline = {"role": "assistant", "content": "<think>\ninline history marker\n</think>\n\n杭州是浙江的省会。"}
    second_user = {"role": "user", "content": "有什么景点？"}
    structured = {
        "role": "assistant",
        "reasoning_content": "\nstructured history marker\n",
        "content": "西湖和灵隐寺。",
    }
    followup = {"role": "user", "content": "再推荐一个，带表情。🌿\n"}
    histories = {
        "plain": [system, first_user, plain, second_user],
        "reasoning": [system, first_user, inline, second_user, structured, followup],
    }
    for name, messages in histories.items():
        for mode, context in (
            ("thinking_on", {"enable_thinking": True}),
            ("thinking_off", {"enable_thinking": False}),
            ("thinking_omitted", {}),
        ):
            yield {
                "name": f"multiturn_{name}_{mode}",
                "messages": messages,
                "add_generation_prompt": True,
                "extra_context": context,
            }
        yield {
            "name": f"multiturn_{name}_no_generation_prompt",
            "messages": messages,
            "add_generation_prompt": False,
            "extra_context": {"enable_thinking": True},
        }
    # Exercise assistant answers after the last user as well as earlier
    # history. The checkpoints differ in which reasoning blocks they retain.
    final_inline = {"role": "assistant", "content": "<think>\ninline final marker\n</think>\n\n西湖。"}
    final_plain = {"role": "assistant", "content": "西湖。"}
    for name, messages in (
        ("inline", [system, first_user, plain, second_user, final_inline]),
        ("structured", [system, first_user, inline, second_user, structured]),
        ("plain", [system, first_user, inline, second_user, final_plain]),
    ):
        yield {
            "name": f"multiturn_completed_{name}",
            "messages": messages,
            "add_generation_prompt": False,
            "extra_context": {},
        }


def renderer_cases(model: str):
    tool_history = [
        {"role": "user", "content": "What is the weather in Hangzhou?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Hangzhou", "unit": "celsius"}},
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 30, "condition": "sunny"}'},
    ]
    common = [
        {
            "name": "unicode_text",
            "messages": [{"role": "user", "content": "你好，介绍一下 mllm。"}],
            "add_generation_prompt": True,
            "extra_context": {"enable_thinking": False},
        },
        {
            "name": "system_thinking",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is 6 * 7?"},
            ],
            "add_generation_prompt": True,
            "extra_context": {"enable_thinking": True},
        },
        {
            "name": "multi_turn_reasoning",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "<think>\nplan\n</think>\n\nHello!"},
                {"role": "user", "content": "Again?"},
            ],
            "add_generation_prompt": True,
            "extra_context": {"enable_thinking": False},
        },
        {
            "name": "tool_response_history",
            "messages": tool_history,
            "tools": [TOOL],
            "add_generation_prompt": True,
            "extra_context": {"enable_thinking": False},
        },
    ]
    if model in ("minicpm5", "qwen3_5"):
        common.extend(multiturn_cases())
    if model == "qwen3_5":
        common.append(
            {
                "name": "multimodal_content_blocks",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "unused-by-renderer"},
                            {"type": "video", "video": "unused-by-renderer"},
                            {"type": "text", "text": "Describe this."},
                        ],
                    }
                ],
                "add_generation_prompt": True,
                "extra_context": {"add_vision_id": True, "enable_thinking": False},
            }
        )
    if model == "minicpm5":
        common.append(
            {
                "name": "tool_sep_and_cdata",
                "messages": [
                    {"role": "system", "content": "Sys<tool_def_sep>tail"},
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": "Checking.<tool_sep>Done.",
                        "tool_calls": [
                            {"function": {"name": "get_weather", "arguments": {"city": "a<b", "note": "x\ny"}}},
                            {"function": {"name": "get_weather", "arguments": {"city": "B"}}},
                        ],
                    },
                    {"role": "tool", "content": {"temperature": 30}},
                    {"role": "tool", "content": "2"},
                ],
                "tools": [TOOL],
                "add_generation_prompt": True,
                "extra_context": {"enable_thinking": True},
            }
        )
    return common


def product_cases(model: str):
    """Runner-shaped requests: what the example CLI sends through convertMessage."""
    cases = [
        {"name": "prompt", "prompt": "你好，介绍一下 mllm。"},
        {"name": "prompt_ascii", "prompt": "What is 6 * 7?"},
    ]
    if model == "minicpm5":
        cases += [
            {"name": "system", "prompt": "Hello", "system": "Be concise."},
            {"name": "system_thinking", "prompt": "Hello", "system": "Be concise.", "enable_thinking": True},
            {"name": "thinking", "prompt": "Hello", "enable_thinking": True},
        ]
    return cases


def product_messages(case: dict):
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


def apply_chat_template_ids(tokenizer, messages, **kwargs):
    # Transformers 4.x can return a list, while 5.x can return a BatchEncoding.
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def report_difference(label, expected, actual):
    """Print the first differing byte or token, including prefix-length mismatches."""
    index = next(
        (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
        min(len(expected), len(actual)),
    )
    before = max(0, index - 3)
    after = index + 4
    print(
        f"  {label}: first_difference={index}, expected_length={len(expected)}, actual_length={len(actual)}, "
        f"expected[{before}:{after}]={expected[before:after]!r}, actual[{before}:{after}]={actual[before:after]!r}"
    )


def run(command, stdin: bytes = b"") -> str:
    completed = subprocess.run(command, input=stdin, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[0]} exited with {completed.returncode}; stderr={completed.stderr.decode('utf-8', 'replace')!r}"
        )
    return completed.stdout.decode("utf-8")


def run_json(command, stdin: bytes = b"") -> dict:
    """The probe prints one JSON object on its last stdout line; runtime logs may precede it."""
    lines = [line for line in run(command, stdin).splitlines() if line.strip()]
    if not lines or not lines[-1].startswith("{"):
        raise RuntimeError(f"{command[0]} did not end with a JSON object: {lines[-3:]!r}")
    return json.loads(lines[-1])


def render_cpp(renderer: Path, template: Path, case: dict) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as request:
        json.dump(case, request, ensure_ascii=False)
        request.flush()
        return run([str(renderer), str(template), request.name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--minicpm5-checkpoint", choices=("1b", "2b"), default="1b")
    parser.add_argument("--model-dir", type=Path, required=True, help="official checkpoint directory")
    parser.add_argument("--renderer", type=Path, required=True, help="Mllm-ChatTemplate-Render")
    parser.add_argument("--probe", type=Path, help="Mllm-ChatTemplate-Probe (enables tokenizer and product-path stages)")
    parser.add_argument("--config", type=Path, help="mllm config.json for the product-path stage")
    args = parser.parse_args()

    if args.model != "minicpm5" and args.minicpm5_checkpoint != "1b":
        parser.error("--minicpm5-checkpoint=2b requires --model minicpm5")
    if args.config and not args.probe:
        parser.error("--config requires --probe")
    pins = MINICPM5_2B if args.model == "minicpm5" and args.minicpm5_checkpoint == "2b" else MODELS[args.model]
    template = args.model_dir / pins.get("template_file", "chat_template.jinja")
    render_source = args.model_dir if pins.get("template_file") == "tokenizer_config.json" else template
    tokenizer_json = args.model_dir / "tokenizer.json"
    if sha256(template) != pins["template_sha256"]:
        raise RuntimeError(f"unexpected {args.model} template identity; expected revision {pins['revision']}")
    if sha256(tokenizer_json) != pins["tokenizer_sha256"]:
        raise RuntimeError(f"unexpected {args.model} tokenizer identity; expected revision {pins['revision']}")
    config_hash = pins.get("tokenizer_config_sha256")
    if config_hash and sha256(args.model_dir / "tokenizer_config.json") != config_hash:
        raise RuntimeError(f"unexpected {args.model} tokenizer config identity; expected revision {pins['revision']}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    failures = []
    print("oracle packages: " + ", ".join(f"{name}={version(name)}" for name in ("transformers", "jinja2", "tokenizers")))

    print(f"== stage A: renderer bytes and token ids ({args.model})")
    for case in renderer_cases(args.model):
        kwargs = dict(case.get("extra_context", {}))
        if "tools" in case:
            kwargs["tools"] = case["tools"]
        kwargs["add_generation_prompt"] = case["add_generation_prompt"]
        reference = tokenizer.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        candidate = render_cpp(args.renderer, render_source, case)
        reference_ids = apply_chat_template_ids(tokenizer, case["messages"], **kwargs)
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        reference_bytes = reference.encode("utf-8")
        candidate_bytes = candidate.encode("utf-8")
        byte_exact = candidate_bytes == reference_bytes
        token_exact = candidate_ids == reference_ids
        mllm_token_exact = None
        if args.probe:
            mllm_ids = run_json(
                [str(args.probe), "--model", args.model, "--tokenizer", str(tokenizer_json), "--tokenize"],
                candidate_bytes,
            )["token_ids"]
            mllm_token_exact = mllm_ids == reference_ids
        print(
            f"{case['name']}: byte_exact={byte_exact} token_exact={token_exact} "
            f"mllm_token_exact={mllm_token_exact} bytes={len(candidate_bytes)} tokens={len(reference_ids)}"
        )
        if not byte_exact:
            report_difference("rendered UTF-8", reference_bytes, candidate_bytes)
        if not token_exact:
            report_difference("Transformers encode of mllm render", reference_ids, candidate_ids)
        if mllm_token_exact is False:
            report_difference("mllm tokenizer", reference_ids, mllm_ids)
        if not byte_exact or not token_exact or mllm_token_exact is False:
            failures.append("A/" + case["name"])

    if args.probe and args.config:
        print(f"== stage B: product path builtin vs jinja_required vs Transformers ({args.model})")
        for case in product_cases(args.model):
            command = [str(args.probe), "--model", args.model, "--tokenizer", str(tokenizer_json),
                       "--config", str(args.config), "--model_dir", str(args.model_dir), "--prompt", case["prompt"]]
            if case.get("system"):
                command += ["--system", case["system"]]
            if case.get("enable_thinking"):
                command.append("--enable_thinking")
            builtin = run_json(command + ["--backend", "builtin"])
            jinja = run_json(command + ["--backend", "jinja_required"])
            # product_thinking: False -> the runner always sends enable_thinking=false;
            # None -> the runner leaves it undefined (official default); otherwise per case.
            thinking_kwargs = {}
            product_thinking = pins.get("product_thinking", "per-case")
            if product_thinking is False:
                thinking_kwargs["enable_thinking"] = False
            elif product_thinking == "per-case":
                thinking_kwargs["enable_thinking"] = bool(case.get("enable_thinking", False))
            reference_ids = apply_chat_template_ids(
                tokenizer, product_messages(case), add_generation_prompt=True, **thinking_kwargs
            )
            builtin_vs_jinja = builtin["token_ids"] == jinja["token_ids"]
            jinja_vs_ref = jinja["token_ids"] == reference_ids
            builtin_vs_ref = builtin["token_ids"] == reference_ids
            print(
                f"{case['name']}: builtin==jinja={builtin_vs_jinja} jinja==transformers={jinja_vs_ref} "
                f"builtin==transformers={builtin_vs_ref} tokens={len(reference_ids)} "
                f"backend={builtin['backend']}/{jinja['backend']}"
            )
            if not jinja_vs_ref:
                report_difference("jinja_required", reference_ids, jinja["token_ids"])
            if not builtin_vs_ref:
                report_difference("builtin", reference_ids, builtin["token_ids"])
            backends_match = (builtin["backend"], jinja["backend"]) == ("builtin", "jinja_required")
            if not (builtin_vs_jinja and jinja_vs_ref and builtin_vs_ref and backends_match):
                failures.append("B/" + case["name"])

    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    scope = "renderer + tokenizer + product" if args.config else "renderer + tokenizer" if args.probe else "renderer only"
    print(f"PASS: {args.model} against Transformers at revision {pins['revision']} ({scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

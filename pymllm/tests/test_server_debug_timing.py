import pytest

from pymllm.configs.global_config import GlobalConfig, make_args, read_args
from pymllm.configs.server_config import ServerConfig
from pymllm.server import launch


@pytest.fixture(autouse=True)
def reset_global_config():
    GlobalConfig.reset()
    yield
    GlobalConfig.reset()


def test_server_debug_timing_is_disabled_by_default():
    assert ServerConfig(model_path=None).enable_debug_timing is False


def test_server_debug_timing_can_be_enabled_from_cli():
    cfg = read_args(
        ["--server.enable_debug_timing"],
        parser=make_args(),
    )

    assert cfg.server.enable_debug_timing is True


def test_debug_timing_is_not_added_when_disabled():
    cfg = GlobalConfig.get_instance()
    cfg.server.enable_debug_timing = False
    payload = {"id": "chatcmpl-test"}

    assert hasattr(launch, "_maybe_add_debug_timing")
    launch._maybe_add_debug_timing(
        payload,
        result={
            "vit_prefill_ms": 12.5,
            "vit_prefill_tokens": 25,
            "llm_prefill_ms": 50.0,
            "llm_decode_ms": 200.0,
        },
        prompt_tokens=100,
        completion_tokens=20,
    )

    assert "timing" not in payload
    assert "debug_timing" not in payload


def test_debug_timing_uses_debug_field_names_when_enabled():
    cfg = GlobalConfig.get_instance()
    cfg.server.enable_debug_timing = True
    payload = {"id": "chatcmpl-test"}

    assert hasattr(launch, "_maybe_add_debug_timing")
    launch._maybe_add_debug_timing(
        payload,
        result={
            "vit_prefill_ms": 12.5,
            "vit_prefill_tokens": 25,
            "llm_prefill_ms": 50.0,
            "llm_decode_ms": 200.0,
        },
        prompt_tokens=100,
        completion_tokens=20,
    )

    assert "timing" not in payload
    assert payload["debug_timing"] == {
        "experimental_vit_prefill_ms": 12.5,
        "experimental_llm_prefill_ms": 50.0,
        "decode_phase_wall_ms": 200.0,
        "prefill_tokens": 100,
        "completion_tokens": 20,
        "experimental_vit_prefill_tps": pytest.approx(2000.0),
        "experimental_llm_prefill_tps": pytest.approx(2000.0),
        "decode_phase_output_tps": pytest.approx(100.0),
    }

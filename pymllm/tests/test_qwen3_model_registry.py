from pymllm.models import get_model_class


def test_registry_resolves_qwen3_causallm():
    cls = get_model_class("Qwen3ForCausalLM")
    assert cls is not None
    assert cls.__name__ == "Qwen3ForCausalLM"

import importlib
import sys
from types import SimpleNamespace


def _load_auxiliary_client(monkeypatch):
    fake_openai = SimpleNamespace(OpenAI=object)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    sys.modules.pop("agent.auxiliary_client", None)
    return importlib.import_module("agent.auxiliary_client")


def test_auxiliary_runtime_override_supplies_main_model(monkeypatch):
    ac = _load_auxiliary_client(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "global-model", "provider": "openrouter"}},
    )

    with ac.auxiliary_runtime_override(
        main_runtime={"model": "project-main", "provider": "minimax-cn"}
    ):
        assert ac._read_main_model() == "project-main"
        assert ac._read_main_provider() == "minimax-cn"


def test_auxiliary_runtime_override_supplies_task_provider(monkeypatch):
    ac = _load_auxiliary_client(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"auxiliary": {"vision": {"provider": "auto", "model": ""}}},
    )

    with ac.auxiliary_runtime_override(
        auxiliary_runtime={"provider": "openrouter", "model": "xiaomi/mimo-v2-pro"}
    ):
        provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model("vision")

    assert provider == "openrouter"
    assert model == "xiaomi/mimo-v2-pro"
    assert base_url is None
    assert api_key is None
    assert api_mode is None

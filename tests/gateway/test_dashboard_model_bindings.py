"""Regression tests for dashboard-driven model bindings in gateway runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _SessionStoreStub:
    def __init__(self):
        self.projects: dict[str, str] = {}
        self.session_id = "sess-1"

    def get_project_name(self, session_key: str):
        return self.projects.get(session_key)

    def set_project_name(self, session_key: str, project_name: str | None):
        if project_name:
            self.projects[session_key] = project_name
        else:
            self.projects.pop(session_key, None)
        return True

    def get_or_create_session(self, source):
        return SimpleNamespace(session_key="session-1", session_id=self.session_id)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = _SessionStoreStub()
    runner._session_model_overrides = {}
    return runner


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_dashboard_binding_hydrates_project_and_project_models_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-main")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"provider": "global-provider"})

    _write_yaml(
        tmp_path / "base" / "conversation_bindings.yaml",
        {
            "bindings": [
                {
                    "platform": "telegram",
                    "chat_id": "chat-1",
                    "thread_id": "thread-9",
                    "project": "proj-a",
                }
            ]
        },
    )

    config = {
        "model": {"default": "global-main", "provider": "global-provider"},
        "auxiliary": {
            "vision": {"model": "global-aux", "provider": "global-provider"},
        },
        "projects": [
            {
                "name": "proj-a",
                "path": str(tmp_path / "projects" / "proj-a"),
                "model": {"default": "project-main", "provider": "project-provider"},
                "auxiliary": {
                    "vision": {"model": "project-aux", "provider": "project-provider"},
                },
            }
        ],
    }

    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        thread_id="thread-9",
        chat_type="group",
        user_id="user-1",
    )
    session_key = "agent:main:telegram:group:chat-1:thread-9"

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config=config,
    )

    assert runner.session_store.get_project_name(session_key) == "proj-a"
    assert model == "project-main"
    assert runtime["provider"] == "project-provider"

    aux = runner._effective_auxiliary_override(session_key, config)
    assert aux["model"] == "project-aux"
    assert aux["provider"] == "project-provider"


def test_conversation_assignment_beats_project_and_global(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-main")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"provider": "global-provider"})

    _write_yaml(
        tmp_path / "base" / "model_assignments.yaml",
        {
            "conversations": [
                {
                    "session_key": "session-1",
                    "main": {"model": "conversation-main", "provider": "conversation-provider"},
                    "auxiliary": {"model": "conversation-aux", "provider": "conversation-provider"},
                }
            ]
        },
    )

    config = {
        "model": {"default": "global-main", "provider": "global-provider"},
        "auxiliary": {
            "vision": {"model": "global-aux", "provider": "global-provider"},
        },
        "projects": [
            {
                "name": "proj-a",
                "path": str(tmp_path / "projects" / "proj-a"),
                "model": {"default": "project-main", "provider": "project-provider"},
                "auxiliary": {
                    "vision": {"model": "project-aux", "provider": "project-provider"},
                },
            }
        ],
    }

    runner = _make_runner()
    runner.session_store.set_project_name("session-1", "proj-a")

    model, runtime = runner._resolve_session_agent_runtime(
        session_key="session-1",
        user_config=config,
    )

    assert model == "conversation-main"
    assert runtime["provider"] == "conversation-provider"

    aux = runner._effective_auxiliary_override("session-1", config)
    assert aux["model"] == "conversation-aux"
    assert aux["provider"] == "conversation-provider"


def test_project_command_use_with_path_updates_config_and_binds_session(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    saved_configs: list[dict] = []

    runner = _make_runner()
    runner._evict_cached_agent = lambda session_key: None
    runner._reset_project_runtime = lambda session_key, session_id=None: None
    runner._set_session_project_runtime_override = lambda session_key, session_id=None, user_config=None: str(
        user_config["projects"][0]["path"]
    )
    runner._load_user_config = lambda: {}
    runner._save_user_config = lambda cfg: saved_configs.append(cfg.copy())
    runner._render_model_current = lambda **kwargs: "rendered"
    runner._session_key_for_source = lambda source: "session-1"

    project_dir = tmp_path / "repo-a"
    project_dir.mkdir()
    event = MessageEvent(
        text=f'/project use repo-a "{project_dir}"',
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-1",
            chat_type="channel",
            user_id="user-1",
        ),
    )

    result = asyncio.run(runner._handle_project_command(event))

    assert "Bound this chat to project `repo-a`" in result
    assert runner.session_store.get_project_name("session-1") == "repo-a"
    assert saved_configs
    assert saved_configs[0]["projects"][0]["name"] == "repo-a"
    assert saved_configs[0]["projects"][0]["path"] == str(project_dir)


def test_project_command_list_shows_projects_and_current_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = _make_runner()
    runner.session_store.set_project_name("session-1", "beta")
    runner._load_user_config = lambda: {}
    runner._session_key_for_source = lambda source: "session-1"
    (tmp_path / "projects" / "alpha").mkdir(parents=True)
    (tmp_path / "projects" / "beta").mkdir(parents=True)
    event = MessageEvent(
        text="/project list",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-1",
            chat_type="channel",
            user_id="user-1",
        ),
    )

    result = asyncio.run(runner._handle_project_command(event))

    assert "Available projects" in result
    assert "alpha" in result
    assert "beta" in result
    assert "Current project: `beta`" in result


def test_project_command_use_rejects_unknown_project(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = _make_runner()
    runner._load_user_config = lambda: {}
    runner._session_key_for_source = lambda source: "session-1"
    runner._evict_cached_agent = lambda session_key: None
    event = MessageEvent(
        text="/project use missing-project",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-1",
            chat_type="channel",
            user_id="user-1",
        ),
    )

    result = asyncio.run(runner._handle_project_command(event))

    assert "not found" in result.lower()
    assert "missing-project" in result
    assert runner.session_store.get_project_name("session-1") is None


def test_set_session_project_runtime_override_registers_bound_project_path(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = _make_runner()
    runner.session_store.set_project_name("session-1", "repo-a")
    config = {
        "projects": [
            {
                "name": "repo-a",
                "path": str(tmp_path / "repo-a"),
            }
        ]
    }

    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: captured.setdefault(task_id, overrides),
    )

    result = runner._set_session_project_runtime_override("session-1", session_id="sess-1", user_config=config)

    assert result == str(tmp_path / "repo-a")
    assert captured["session-1"]["cwd"] == str(tmp_path / "repo-a")
    assert captured["sess-1"]["cwd"] == str(tmp_path / "repo-a")


def test_session_context_prompt_mentions_bound_project(tmp_path, monkeypatch):
    from datetime import datetime
    from gateway.config import GatewayConfig
    from gateway.session import SessionEntry, build_session_context, build_session_context_prompt

    monkeypatch.setattr("gateway.session.get_hermes_home", lambda: tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-2",
        chat_name="strategy",
        chat_type="group",
        thread_id="topic-3",
        user_id="user-2",
        user_name="test-user",
    )
    entry = SessionEntry(
        session_key="telegram:chat-2:topic-3",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="group",
        project_name="sample-project",
    )

    context = build_session_context(source, GatewayConfig(platforms={}), entry)
    prompt = build_session_context_prompt(context)

    assert context.project_name == "sample-project"
    assert context.project_path == str(tmp_path / "projects" / "sample-project")
    assert "Bound Project" in prompt
    assert "sample-project" in prompt

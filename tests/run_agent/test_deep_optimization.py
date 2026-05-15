from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _build_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("read_file")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def test_maybe_refine_final_response_skips_when_disabled():
    agent = _build_agent()
    agent._deep_optimization_enabled = False
    agent._deep_optimization_min_user_chars = 10

    agent._run_text_refinement_request = MagicMock(return_value="SHOULD_NOT_RUN")

    result = agent._maybe_refine_final_response(
        final_response="draft",
        original_user_message="这是一个足够长的请求",
    )

    assert result == "draft"
    agent._run_text_refinement_request.assert_not_called()


def test_maybe_refine_final_response_skips_when_user_prompt_too_short():
    agent = _build_agent()
    agent._deep_optimization_enabled = True
    agent._deep_optimization_min_user_chars = 50

    agent._run_text_refinement_request = MagicMock(return_value="SHOULD_NOT_RUN")

    result = agent._maybe_refine_final_response(
        final_response="draft",
        original_user_message="太短",
    )

    assert result == "draft"
    agent._run_text_refinement_request.assert_not_called()


def test_maybe_refine_final_response_applies_single_pass_and_strips_think_blocks():
    agent = _build_agent()
    agent._deep_optimization_enabled = True
    agent._deep_optimization_passes = 1
    agent._deep_optimization_min_user_chars = 1

    agent._run_text_refinement_request = MagicMock(
        return_value="<think>internal</think>\n优化后的最终答案"
    )

    result = agent._maybe_refine_final_response(
        final_response="初稿",
        original_user_message="请你把这份方案做深度优化并补齐风险和验收标准",
    )

    assert result == "优化后的最终答案"
    assert agent._run_text_refinement_request.call_count == 1


def test_maybe_refine_final_response_respects_pass_count():
    agent = _build_agent()
    agent._deep_optimization_enabled = True
    agent._deep_optimization_passes = 3
    agent._deep_optimization_min_user_chars = 1

    agent._run_text_refinement_request = MagicMock(
        side_effect=["v1", "v2", "v3"]
    )

    result = agent._maybe_refine_final_response(
        final_response="初稿",
        original_user_message="这是一个很长很长的复杂需求，需要持续迭代优化",
    )

    assert result == "v3"
    assert agent._run_text_refinement_request.call_count == 3


def test_maybe_refine_final_response_short_prompt_with_optimize_keyword_triggers():
    agent = _build_agent()
    agent._deep_optimization_enabled = True
    agent._deep_optimization_passes = 1
    agent._deep_optimization_min_user_chars = 999

    agent._run_text_refinement_request = MagicMock(return_value="优化版")

    result = agent._maybe_refine_final_response(
        final_response="初稿",
        original_user_message="继续优化",
    )

    assert result == "优化版"
    assert agent._run_text_refinement_request.call_count == 1


def test_maybe_refine_final_response_rejects_clear_quality_regression():
    agent = _build_agent()
    agent._deep_optimization_enabled = True
    agent._deep_optimization_passes = 1
    agent._deep_optimization_min_user_chars = 1

    strong = """1. 步骤A\n2. 步骤B\n风险：接口超时\n验收：测试通过\n执行命令：pytest -q\n"""
    weak = "好。"
    agent._run_text_refinement_request = MagicMock(return_value=weak)

    result = agent._maybe_refine_final_response(
        final_response=strong,
        original_user_message="请持续优化并给出可执行方案",
    )

    assert result == strong
    assert agent._last_deep_optimization_meta["rejected_candidates"] == 1

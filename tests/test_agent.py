"""Agent 装配与结果解析测试：用假模型验证工具调用链路，不联网。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.tools import get_waybill
from app.agents.waybill_agent import build_agent, extract_final_text
from tests.fakes import ScriptedChatModel


def _tool_call_response(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class TestBuildAgent:
    def test_完整工具调用链路(self):
        model = ScriptedChatModel(
            responses=[
                _tool_call_response("get_waybill", {"waybill_no": "DK20261021001"}),
                AIMessage(content="运单已签收，运费 380.00 元。"),
            ]
        )
        agent = build_agent(model, tools=[get_waybill])
        result = agent.invoke({"messages": [{"role": "user", "content": "查下 DK20261021001"}]})

        messages = result["messages"]
        assert isinstance(messages[0], HumanMessage)
        # 中间必须有 ToolMessage，说明工具真的被执行了
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "深圳南山起降点" in tool_msgs[0].content
        assert extract_final_text(result) == "运单已签收，运费 380.00 元。"

    def test_空工具列表应报错(self):
        model = ScriptedChatModel(responses=[AIMessage(content="hi")])
        with pytest.raises(ValueError, match="tools 不能为空"):
            build_agent(model, tools=[])


class TestExtractFinalText:
    def test_跳过带工具调用的中间态(self):
        result = {
            "messages": [
                HumanMessage(content="查运单"),
                _tool_call_response("get_waybill", {"waybill_no": "DK20261021001"}),
                ToolMessage(content="运单信息", tool_call_id="call_1"),
                AIMessage(content="最终答复"),
            ]
        }
        assert extract_final_text(result) == "最终答复"

    def test_无消息时报错(self):
        with pytest.raises(KeyError):
            extract_final_text({"messages": []})

    def test_仅有工具调用无最终文本时返回空串(self):
        result = {"messages": [_tool_call_response("get_waybill", {"waybill_no": "X"})]}
        assert extract_final_text(result) == ""

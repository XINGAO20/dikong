"""Agent 装配：LangChain 1.x 的 ``create_agent`` 用法。

注意版本差异：1.x 的入参是 ``model`` / ``tools`` / ``system_prompt``；
0.x 时代的 ``initialize_agent``、``AgentExecutor`` 已被取代。
"""

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.agents.tools import WAYBILL_TOOLS

SYSTEM_PROMPT = (
    "你是低空物流调度助手。回答业务问题时优先调用工具取数，"
    "工具未返回的信息一律说明「数据未提供」，不得编造运单号、金额或时效。"
    "涉及金额时保留两位小数并标注币种。"
)


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
):
    """构造可执行 agent（返回 CompiledStateGraph）。

    Args:
        model: ChatModel。测试时可注入支持 bind_tools 的假模型。
        tools: 工具列表，默认使用运单工具集。

    Returns:
        编译后的图，调用 ``invoke({"messages": [...]})``。
    """
    resolved = list(tools) if tools is not None else list(WAYBILL_TOOLS)
    if not resolved:
        raise ValueError("tools 不能为空：无用工具的 agent 不需要走 agent 路径")

    return create_agent(
        model=model,
        tools=resolved,
        system_prompt=SYSTEM_PROMPT,
    )


def extract_final_text(result: dict) -> str:
    """从 agent 返回状态中取最后一条 AI 消息的文本。

    Args:
        result: ``agent.invoke(...)`` 的返回值。

    Raises:
        KeyError: 返回值缺少 ``messages`` 时。
    """
    messages = result.get("messages")
    if not messages:
        raise KeyError("agent 返回值中缺少 messages")

    for message in reversed(messages):
        # 只取最终答复；带 tool_calls 的中间态不算
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""

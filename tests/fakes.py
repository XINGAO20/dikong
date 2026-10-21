"""测试用假模型：支持 bind_tools 的剧本式 ChatModel。

为什么需要它：LangChain 1.x 的 ``create_agent`` 会对模型调用 ``bind_tools``，
而库里自带的 ``GenericFakeChatModel`` 未实现该方法会直接抛 NotImplementedError。
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """按预设剧本依次返回 AIMessage 的假模型，不发起任何网络请求。

    Attributes:
        responses: 依次返回的消息列表；用尽后重复最后一条。
    """

    responses: list[AIMessage] = []
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        """接受工具绑定但不做实际处理——剧本已固定。"""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise ValueError("responses 不能为空")
        index = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[index])]
        )

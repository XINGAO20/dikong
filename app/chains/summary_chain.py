"""LCEL 链示例：把运单原始记录压成结构化摘要。

提示词约束了「信息缺失写未提供」，避免模型对运单数据编造字段。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是低空物流运营助手。根据运单原始记录输出不超过 {max_words} 字的中文摘要，"
            "按「航线—载荷—时效—异常」四段陈述。"
            "凡是原始记录中未出现的信息一律写「未提供」，禁止推测或补全。",
        ),
        ("human", "运单原始记录：\n{raw_record}"),
    ]
)


def build_summary_chain(model: BaseChatModel, max_words: int = 80):
    """构造摘要链：prompt -> model -> 纯文本输出。

    Args:
        model: 任意 ChatModel（测试时可注入假模型）。
        max_words: 摘要字数上限。

    Returns:
        Runnable，调用 ``invoke({"raw_record": "..."})`` 得到字符串。
    """
    if max_words <= 0:
        raise ValueError("max_words 必须为正整数")

    return SUMMARY_PROMPT.partial(max_words=str(max_words)) | model | StrOutputParser()

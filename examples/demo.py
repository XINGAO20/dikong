"""最小可运行 demo：一次跑通 agent 工具调用、摘要链、RAG 三条链路。

用法（两种模式，默认离线）::

    python -m examples.demo              # 离线演示，不需要 API Key，不联网
    python -m examples.demo --live       # 真实模型，需 LLM_API_KEY
    python -m examples.demo --check      # 自检：三条链路跑通且返回非空，适合 CI

为什么默认离线：仓库克隆下来就能跑，不必先配 Key。真实调用只是把假模型换成
``build_chat_model()``，业务代码完全一致——这一点本身就是要演示的重点。
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.tools import calc_shipping_fee, days_until, get_waybill
from app.agents.waybill_agent import build_agent, extract_final_text
from app.chains.summary_chain import build_summary_chain
from app.core.fakes import ScriptedChatModel
from app.rag.qa import build_rag_chain, build_vector_store

# --------------------------------------------------------------------------- #
# 演示数据
# --------------------------------------------------------------------------- #
DEMO_QUESTION = "帮我查一下运单 DK20261021001 的状态和运费"
DEMO_RAW_RECORD = "DK20261021002 广州黄埔起降点→东莞松山湖起降点 5kg 运费 160.50 元 运输中"
DEMO_KB = [
    "低空物流起降点之间的最小水平间距不低于 50 米，以满足多旋翼无人机同时起降的安全间隔。",
    "单架次最大载荷 20 公斤，超出需拆分为多个架次并分别计费。",
    "运费按每公斤单价乘以实际载荷计算，不足 1 公斤按 1 公斤计。",
]
DEMO_RAG_QUESTION = "单架次最大载荷是多少"


def _rule(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# --------------------------------------------------------------------------- #
# 三条链路
# --------------------------------------------------------------------------- #
def run_agent_demo(model: BaseChatModel, offline: bool) -> str:
    """演示 1：agent 调用业务工具后作答。

    离线模式下用剧本模型先"要求调用工具"、再"给出结论"，
    因此可以在无网络、无 Key 的情况下完整验证工具链路。
    """
    _rule("演示 1 / 3 · Agent + 工具调用")

    if offline:
        tool_model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_waybill",
                            "args": {"waybill_no": "DK20261021001"},
                            "id": "call_demo_1",
                        }
                    ],
                ),
                AIMessage(content="运单 DK20261021001 已签收，运费 380.00 元。"),
            ]
        )
    else:
        tool_model = model

    print(f"提问：{DEMO_QUESTION}")
    agent = build_agent(tool_model)
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUESTION}]})

    # 逐条打印消息轨迹，让"工具真的被调用了"这件事可见
    print("\n消息轨迹：")
    for i, msg in enumerate(result["messages"], 1):
        kind = type(msg).__name__
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            names = ", ".join(tc["name"] for tc in msg.tool_calls)
            print(f"  {i}. {kind:<14} → 请求调用工具: {names}")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content).replace("\n", " ")[:60]
            print(f"  {i}. {kind:<14} → 工具返回: {preview}...")
        else:
            text = str(getattr(msg, "content", "")).replace("\n", " ")
            print(f"  {i}. {kind:<14} → {text[:60]}")

    answer = extract_final_text(result)
    print(f"\n最终答复：{answer}")
    return answer


def run_tools_demo() -> None:
    """演示 2：工具本身可独立调用，便于单测与排查。"""
    _rule("演示 2 / 3 · 业务工具直接调用")

    cases = [
        ("运单查询", get_waybill, {"waybill_no": "DK20261021002"}),
        ("运单不存在", get_waybill, {"waybill_no": "DK99999999999"}),
        ("运费计算", calc_shipping_fee, {"weight_kg": 12, "unit_price_cny": 31.6667}),
        ("参数非法", calc_shipping_fee, {"weight_kg": 0, "unit_price_cny": 10}),
        ("时效计算", days_until, {"target_date": "2026-11-01"}),
        ("日期格式错误", days_until, {"target_date": "2026/11/01"}),
    ]
    for label, tool, args in cases:
        print(f"  {label:<12} {tool.name:<18} → {tool.invoke(args)}")


def run_summary_demo(model: BaseChatModel, offline: bool) -> str:
    """演示 3a：LCEL 摘要链。"""
    _rule("演示 3 / 3 · 摘要链与 RAG")

    summary_model = (
        ScriptedChatModel(
            responses=[
                AIMessage(
                    content=(
                        "航线：广州黄埔起降点—东莞松山湖起降点；载荷：5kg；"
                        "时效：运输中；异常：未提供"
                    )
                )
            ]
        )
        if offline
        else model
    )
    print(f"原始记录：{DEMO_RAW_RECORD}")
    summary = build_summary_chain(summary_model, max_words=60).invoke(
        {"raw_record": DEMO_RAW_RECORD}
    )
    print(f"摘要：{summary}")
    return summary


def run_rag_demo(model: BaseChatModel, embeddings: Embeddings, offline: bool) -> str:
    """演示 3b：知识库检索问答。"""
    print()
    print(f"知识库条目：{len(DEMO_KB)} 条")
    print(f"提问：{DEMO_RAG_QUESTION}")

    store = build_vector_store(DEMO_KB, embeddings)
    rag_model = (
        ScriptedChatModel(responses=[AIMessage(content="单架次最大载荷 20 公斤。")])
        if offline
        else model
    )
    answer = build_rag_chain(store, rag_model, k=2).invoke(DEMO_RAG_QUESTION)
    print(f"答复：{answer}")
    return answer


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _build_live_components() -> tuple[BaseChatModel, Embeddings]:
    """构造真实模型与向量化实现；缺凭证时给出可操作提示，而非堆栈。"""
    from langchain_openai import OpenAIEmbeddings

    from app.config import get_settings
    from app.core.llm import MissingCredentialError, build_chat_model

    settings = get_settings()
    try:
        model = build_chat_model(settings)
    except MissingCredentialError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        print("提示：不加 --live 即可以离线模式运行，无需任何凭证。", file=sys.stderr)
        raise SystemExit(2) from exc

    assert settings.llm_api_key is not None  # 上面已校验
    embeddings = OpenAIEmbeddings(
        api_key=settings.llm_api_key.get_secret_value(),
        **({"base_url": settings.llm_base_url} if settings.llm_base_url else {}),
    )
    return model, embeddings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo",
        description="低空物流 LangChain 最小 demo：默认离线运行",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用真实模型（需 LLM_API_KEY）；默认用剧本模型离线运行",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="自检模式：跑完三条链路并校验返回值非空，失败时以退出码 1 结束",
    )
    args = parser.parse_args(argv)

    offline = not args.live

    if offline:
        # 离线模式：向量化也用固定替身，避免任何网络请求
        from langchain_core.embeddings import FakeEmbeddings

        model: BaseChatModel = ScriptedChatModel(responses=[AIMessage(content="离线占位")])
        embeddings: Embeddings = FakeEmbeddings(size=64)
        print("运行模式：离线（不联网、不需要 API Key）")
    else:
        model, embeddings = _build_live_components()
        print(f"运行模式：真实模型（{type(model).__name__}）")

    if args.check:
        # 自检：只验证"能跑通且返回非空"，不打印细节
        agent_out = run_agent_demo(model, offline)
        summary_out = run_summary_demo(model, offline)
        rag_out = run_rag_demo(model, embeddings, offline)
        failures = [
            name
            for name, value in (
                ("agent", agent_out),
                ("summary", summary_out),
                ("rag", rag_out),
            )
            if not str(value).strip()
        ]
        if failures:
            print(f"\n[自检失败] 以下链路返回空：{failures}", file=sys.stderr)
            return 1
        print("\n[自检通过] agent / summary / rag 三条链路均返回非空结果")
        return 0

    run_agent_demo(model, offline)
    run_tools_demo()
    run_summary_demo(model, offline)
    run_rag_demo(model, embeddings, offline)

    print()
    print("=" * 68)
    print("全部演示完成。真实调用请加 --live 并配置 LLM_API_KEY。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

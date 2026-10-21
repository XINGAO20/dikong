"""命令行入口：三种用法。

    python -m app.cli ask "帮我查运单 DK20261021001"
    python -m app.cli summary "运单 DK20261021002 已从广州起飞"
    python -m app.cli rag "起降点的最小间距是多少"   # 需先准备 kb 文本

需要环境变量 LLM_API_KEY；缺少时给出明确提示而非堆栈。
"""

import argparse
import sys

from app.config import get_settings
from app.core.llm import MissingCredentialError, build_chat_model


def _build_model():
    try:
        return build_chat_model(get_settings())
    except MissingCredentialError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        sys.exit(2)


def cmd_ask(question: str) -> None:
    from app.agents.waybill_agent import build_agent, extract_final_text

    agent = build_agent(_build_model())
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(extract_final_text(result) or "（模型未返回文本）")


def cmd_summary(raw_record: str) -> None:
    from app.chains.summary_chain import build_summary_chain

    chain = build_summary_chain(_build_model())
    print(chain.invoke({"raw_record": raw_record}))


def cmd_rag(question: str) -> None:
    from langchain_openai import OpenAIEmbeddings

    from app.rag.qa import build_rag_chain, build_vector_store

    settings = get_settings()
    sample_kb = [
        "低空物流起降点之间的最小水平间距不低于 50 米，以满足多旋翼无人机同时起降的安全间隔。",
        "单架次最大载荷 20 公斤，超出需拆分为多个架次并分别计费。",
        "运费按每公斤单价乘以实际载荷计算，不足 1 公斤按 1 公斤计。",
    ]
    embeddings = OpenAIEmbeddings(
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    )
    store = build_vector_store(sample_kb, embeddings)
    print(build_rag_chain(store, _build_model()).invoke(question))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dikong", description="低空物流 LangChain 示例")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="带工具的 agent 问答")
    p_ask.add_argument("question")

    p_sum = sub.add_parser("summary", help="运单摘要")
    p_sum.add_argument("raw_record")

    p_rag = sub.add_parser("rag", help="知识库检索问答")
    p_rag.add_argument("question")

    args = parser.parse_args(argv)
    {"ask": cmd_ask, "summary": cmd_summary, "rag": cmd_rag}[args.command](
        args.question if args.command != "summary" else args.raw_record
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

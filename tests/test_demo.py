"""demo 的离线测试：确保示例代码本身可运行，不会随主代码改动而失效。

这些用例全部离线，不需要 API Key——与 demo 的默认模式一致。
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage

from app.core.fakes import ScriptedChatModel
from examples.demo import (
    DEMO_KB,
    DEMO_QUESTION,
    DEMO_RAG_QUESTION,
    main,
    run_agent_demo,
    run_rag_demo,
    run_summary_demo,
    run_tools_demo,
)


def _offline_model() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[AIMessage(content="离线占位")])


class TestAgentDemo:
    def test_返回非空答复(self):
        assert run_agent_demo(_offline_model(), offline=True).strip()

    def test_答复内容与工具数据一致(self):
        """离线剧本里的结论必须与工具返回的运单事实相符，否则演示就是误导。"""
        answer = run_agent_demo(_offline_model(), offline=True)
        # get_waybill("DK20261021001") 的运费为 380.00，状态为已签收
        assert "380.00" in answer
        assert "已签收" in answer


class TestToolsDemo:
    def test_可正常运行(self, capsys):
        run_tools_demo()
        out = capsys.readouterr().out
        # 正常与异常两类分支都应被演示到
        assert "未找到运单" in out
        assert "错误：重量必须大于 0" in out


class TestSummaryDemo:
    def test_返回非空摘要(self):
        assert run_summary_demo(_offline_model(), offline=True).strip()


class TestRagDemo:
    def test_返回非空答复(self):
        answer = run_rag_demo(_offline_model(), FakeEmbeddings(size=32), offline=True)
        assert answer.strip()

    def test_知识库非空(self):
        assert len(DEMO_KB) >= 3


class TestMainEntry:
    def test_离线完整运行返回0(self):
        assert main([]) == 0

    def test_check模式返回0(self):
        assert main(["--check"]) == 0

    def test_live模式无凭证时退出码为2(self, monkeypatch):
        """缺少 LLM_API_KEY 时必须明确报错退出，而不是抛堆栈。"""
        for key in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LLM_API_KEY", "")

        from app.config import get_settings

        get_settings.cache_clear()
        try:
            with pytest.raises(SystemExit) as exc:
                main(["--live"])
            assert exc.value.code == 2
        finally:
            get_settings.cache_clear()


class TestDemoConstants:
    def test_提问与知识库主题一致(self):
        """RAG 问题必须在知识库中能找到依据，否则演示结果没有意义。"""
        assert "载荷" in DEMO_RAG_QUESTION
        assert any("载荷" in item for item in DEMO_KB)

    def test_agent提问包含运单号(self):
        assert "DK20261021001" in DEMO_QUESTION

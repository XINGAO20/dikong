"""链与 RAG 测试：使用假模型，全程离线。"""

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chains.summary_chain import SUMMARY_PROMPT, build_summary_chain
from app.rag.qa import build_rag_chain, build_vector_store


class TestSummaryPrompt:
    """直接校验提示词本身，不经过模型。"""

    def test_包含缺失信息约束(self):
        rendered = SUMMARY_PROMPT.invoke(
            {"raw_record": "DK20261021001 深圳→珠海 12kg", "max_words": "80"}
        )
        text = rendered.to_string()
        assert "未提供" in text
        assert "禁止推测" in text

    def test_渲染出原始记录(self):
        rendered = SUMMARY_PROMPT.invoke({"raw_record": "测试运单内容", "max_words": "50"})
        assert "测试运单内容" in rendered.to_string()
        assert "50" in rendered.to_string()


class TestSummaryChain:
    def test_返回纯文本(self):
        model = FakeListChatModel(responses=["航线：深圳—珠海；载荷：12kg"])
        chain = build_summary_chain(model, max_words=50)
        out = chain.invoke({"raw_record": "DK20261021001 深圳南山至珠海金湾 12kg"})
        assert isinstance(out, str)
        assert "深圳" in out

    def test_max_words_非法值报错(self):
        model = FakeListChatModel(responses=["x"])
        with pytest.raises(ValueError, match="max_words"):
            build_summary_chain(model, max_words=0)


class TestRag:
    def test_建库与检索(self):
        embeddings = FakeEmbeddings(size=16)
        kb = [
            "起降点之间最小水平间距不低于 50 米。",
            "单架次最大载荷 20 公斤。",
        ]
        store = build_vector_store(kb, embeddings)
        hits = store.similarity_search("间距", k=1)
        assert len(hits) == 1

    def test_空文本不报错(self):
        store = build_vector_store(["", "   "], FakeEmbeddings(size=8))
        assert store.similarity_search("任意", k=1) == []

    def test_问答链可运行(self):
        embeddings = FakeEmbeddings(size=16)
        store = build_vector_store(["单架次最大载荷 20 公斤。"], embeddings)
        model = FakeListChatModel(responses=["最大载荷 20 公斤。"])
        out = build_rag_chain(store, model, k=1).invoke("最大载荷是多少")
        assert isinstance(out, str)

    def test_k_非法值报错(self):
        store = build_vector_store(["内容"], FakeEmbeddings(size=8))
        model = FakeListChatModel(responses=["x"])
        with pytest.raises(ValueError, match="k 必须为正整数"):
            build_rag_chain(store, model, k=0)

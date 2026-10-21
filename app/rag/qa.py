"""RAG 最小实现：内存向量库 + 文本切分 + 检索问答链。

生产环境把 ``InMemoryVectorStore`` 换成 pgvector / Milvus 等持久化后端即可，
接口保持一致（``add_documents`` / ``similarity_search``）。
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "仅依据下列资料回答问题。资料中没有的内容，直接回答「资料中未提及」，"
            "禁止使用外部知识补充。\n\n资料：\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)


def build_vector_store(texts: list[str], embeddings: Embeddings) -> InMemoryVectorStore:
    """切分文本并写入内存向量库。

    Args:
        texts: 原始文档片段。
        embeddings: 向量化实现（测试用 FakeEmbeddings）。
    """
    docs = [Document(page_content=t) for t in texts if t and t.strip()]
    chunks = _splitter.split_documents(docs)

    store = InMemoryVectorStore(embedding=embeddings)
    if chunks:
        store.add_documents(chunks)
    return store


def build_rag_chain(store: InMemoryVectorStore, model: BaseChatModel, k: int = 3):
    """构造检索问答链：question -> 检索 -> prompt -> model -> text。

    Args:
        store: 已灌入数据的向量库。
        model: ChatModel。
        k: 召回的文档片段数。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")

    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )

# dikong · 低空物流 AI 应用

基于 **LangChain 1.x** 的低空物流业务应用骨架，提供三条可运行的链路：带工具的运单问答 agent、运单摘要链、知识库检索问答（RAG）。

> 定位说明：本仓库当前为**骨架**，业务数据使用内置演示数据，尚未接入真实运单库。

## 环境要求

- Python **3.11+**（CI 覆盖 3.11 / 3.12）
- `langchain>=1.0,<2.0`（本项目按 1.x API 编写，**与 0.x 教程不兼容**）

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env               # 然后填入 LLM_API_KEY
```

> **密钥纪律**：`LLM_API_KEY` 只写在 `.env`（已被 `.gitignore` 排除）或平台的 Secret 中。
> 任何时候不要把密钥写进源码、提交信息或 issue。

## 先跑 demo（不需要 Key）

克隆下来直接跑，**离线、不联网、不需要任何凭证**：

```bash
python -m examples.demo          # 完整演示，打印三条链路的执行过程
python -m examples.demo --check  # 自检：跑通即返回 0，失败返回 1，适合 CI
python -m examples.demo --live   # 真实模型调用，需已配置 LLM_API_KEY
```

离线模式用 `ScriptedChatModel`（剧本模型）替代真实模型，业务代码与线上完全一致——
换的只是模型实例。这正是 demo 要演示的重点：**用可替换的模型实现离线可测**。

输出示例（节选）：

```
演示 1 / 3 · Agent + 工具调用
提问：帮我查一下运单 DK20261021001 的状态和运费
消息轨迹：
  1. HumanMessage   → 帮我查一下运单 DK20261021001 的状态和运费
  2. AIMessage      → 请求调用工具: get_waybill
  3. ToolMessage    → 工具返回: 运单号 DK20261021001：深圳南山起降点 → 珠海金湾起降点...
  4. AIMessage      → 运单 DK20261021001 已签收，运费 380.00 元。
```

## 用法

```bash
# 1) 带工具的 agent：会调用 get_waybill 查库，不会凭空编造运单
python -m app.cli ask "帮我查一下运单 DK20261021001 的状态和运费"

# 2) LCEL 摘要链
python -m app.cli summary "DK20261021002 广州黄埔→东莞松山湖 5kg 运输中"

# 3) RAG 检索问答（演示用内存向量库，不持久化）
python -m app.cli rag "单架次最大载荷是多少"
```

## 目录结构

```
app/
├── config.py               # pydantic-settings 配置，密钥用 SecretStr
├── core/
│   ├── llm.py              # 模型工厂，缺凭证时明确报错
│   └── fakes.py            # 剧本模型替身，供 demo 与测试共用
├── chains/summary_chain.py # LCEL：prompt | model | parser
├── agents/
│   ├── tools.py            # 业务工具（只读、无副作用）
│   └── waybill_agent.py    # LangChain 1.x create_agent 装配
├── rag/qa.py               # 切分 + 向量库 + 检索问答链
└── cli.py                  # 命令行入口
examples/demo.py            # 最小可运行 demo（默认离线）
tests/                      # 42 个用例，全部离线，无需 API Key
```

## 测试

```bash
pytest -q            # 42 passed，不联网、不需要 API Key
ruff check .         # 代码风格
```

测试用 `app/core/fakes.py` 的 `ScriptedChatModel` 替身（`tests/fakes.py` 仅做转发）：LangChain 1.x 的 `create_agent` 会强制调用模型的 `bind_tools`，而内置的 `GenericFakeChatModel` 未实现该方法（会抛 `NotImplementedError`），因此需要自定义替身。

## 设计约定

| 约定 | 理由 |
|---|---|
| 密钥一律 `SecretStr` | 防止 `repr()` / 日志 / 异常栈打印明文 |
| 金额用 `Decimal` 计算 | `float` 在 `1.15*0.1` 这类输入上会算成 `0.11` 而非 `0.12`，属资金类缺陷 |
| 工具全部只读 | agent 不持有写权限，避免误改业务数据 |
| 提示词强制「未提供」 | 模型不得对运单号、金额、时效进行推测补全 |
| CI `permissions: contents: read` | 最小权限，PR 流水线不需要写仓库 |

## 已知限制

- 业务数据是 `app/agents/tools.py` 里的静态字典，**生产需替换为数据库查询层**。
- `InMemoryVectorStore` 不持久化，重启即失效；生产应换 pgvector / Milvus。
- 未实现鉴权、限流、审计日志——对外提供 API 前必须补齐。

## 许可

内部项目，未开放外部许可。

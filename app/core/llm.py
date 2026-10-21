"""模型工厂：把配置翻译成 LangChain 的 ChatModel 实例。

要点：
- 统一走 ``init_chat_model``，换供应商只改配置，不改业务代码。
- 缺少 API Key 时明确报错，不做静默降级（避免把空 key 发到线上）。
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings


class MissingCredentialError(RuntimeError):
    """缺少模型凭证时抛出，便于上层给出可操作的提示。"""


def _resolve_api_key(settings: Settings) -> str:
    """取出可用的 API Key 明文。

    Raises:
        MissingCredentialError: 未配置或配置为空字符串时。

    说明：``.env.example`` 里 ``LLM_API_KEY=`` 就是空值，用户照抄后
    若只判断 ``None``，空字符串会一路传到第三方 SDK，最终抛出库内部的
    堆栈（如 ``OpenAIError: Missing credentials``），对使用者毫无指引价值。
    """
    secret = settings.llm_api_key
    plain = secret.get_secret_value() if secret is not None else ""
    if not plain.strip():
        raise MissingCredentialError(
            "缺少模型凭证：请设置环境变量 LLM_API_KEY（本地写入 .env，"
            "线上交由平台密钥管理下发）。当前值为空。"
        )
    return plain


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """按配置构造 ChatModel。

    Args:
        settings: 配置对象，为空则读取全局配置。

    Raises:
        MissingCredentialError: 未配置 ``LLM_API_KEY`` 或配置为空时。
    """
    cfg = settings or get_settings()
    api_key = _resolve_api_key(cfg)

    kwargs: dict = {
        "model": cfg.llm_model,
        "model_provider": cfg.llm_provider,
        "temperature": cfg.llm_temperature,
        "timeout": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
        # SecretStr 在此处才解开，避免日志/异常栈里出现明文
        "api_key": api_key,
    }
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url

    return init_chat_model(**kwargs)

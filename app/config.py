"""应用配置：统一从环境变量 / .env 读取，禁止在代码里硬编码密钥。"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行期配置。所有字段均可通过环境变量覆盖（大写，如 LLM_MODEL）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 模型 ----
    llm_provider: str = Field(default="openai", description="模型提供方，如 openai / anthropic")
    llm_model: str = Field(default="gpt-4o-mini", description="模型名")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_api_key: SecretStr | None = Field(default=None, description="模型 API Key，禁止提交到仓库")
    llm_base_url: str | None = Field(default=None, description="自定义网关地址，留空用官方默认")

    # ---- 网络 ----
    timeout_seconds: int = Field(default=60, gt=0, description="单次调用超时（秒）")
    max_retries: int = Field(default=2, ge=0, le=10, description="失败重试次数")


@lru_cache
def get_settings() -> Settings:
    """进程内单例，避免重复解析 .env。"""
    return Settings()

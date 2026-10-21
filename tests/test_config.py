"""配置与模型工厂测试：不联网，只验证装配逻辑与缺凭证时的报错。"""

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.core.llm import MissingCredentialError, build_chat_model


class TestSettings:
    def test_默认值(self):
        cfg = Settings(_env_file=None)
        assert cfg.llm_provider == "openai"
        assert cfg.llm_temperature == 0.0
        assert cfg.llm_api_key is None

    def test_温度越界被拒(self):
        # 用 pydantic 的 ValidationError，而不是裸 Exception——
        # 裸断言会把「任何错误」都算通过，掩盖真正的失败原因
        with pytest.raises(ValidationError, match="llm_temperature"):
            Settings(_env_file=None, llm_temperature=3.0)

    def test_温度为负也被拒(self):
        with pytest.raises(ValidationError, match="llm_temperature"):
            Settings(_env_file=None, llm_temperature=-0.1)

    def test_密钥不会出现在_repr(self):
        cfg = Settings(_env_file=None, llm_api_key=SecretStr("sk-super-secret-value"))
        # SecretStr 的 repr 应为掩码，防止日志泄漏
        assert "sk-super-secret-value" not in repr(cfg)
        assert cfg.llm_api_key.get_secret_value() == "sk-super-secret-value"


class TestBuildChatModel:
    def test_缺密钥时报明确错误(self):
        cfg = Settings(_env_file=None, llm_api_key=None)
        with pytest.raises(MissingCredentialError, match="LLM_API_KEY"):
            build_chat_model(cfg)

    def test_有密钥时能构造出模型对象(self):
        cfg = Settings(
            _env_file=None,
            llm_api_key=SecretStr("sk-test-dummy-key"),
            llm_model="gpt-4o-mini",
        )
        model = build_chat_model(cfg)
        # 只验证类型与绑定参数，不发起请求
        assert type(model).__name__ == "ChatOpenAI"
        assert model.model_name == "gpt-4o-mini"

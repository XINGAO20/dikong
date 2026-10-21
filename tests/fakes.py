"""测试用假模型。

实现已上移到 ``app/core/fakes.py``，供 demo 与测试共用，避免两处各写一份导致行为漂移。
本模块只做转发，保留原有导入路径，既有测试无需改动。
"""

from app.core.fakes import ScriptedChatModel

__all__ = ["ScriptedChatModel"]

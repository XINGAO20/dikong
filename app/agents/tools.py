"""工具集：低空物流业务里可被 agent 调用的只读工具。

设计约束：
- 全部为纯函数或只读查询，副作用为零，避免 agent 误改数据。
- 金额、重量等数值用 Decimal/整型，不用 float 累积误差。
"""

from datetime import date
from decimal import Decimal, InvalidOperation

from langchain_core.tools import tool

# 演示用的静态运单表；生产环境应替换为数据库查询层
_WAYBILLS: dict[str, dict] = {
    "DK20261021001": {
        "origin": "深圳南山起降点",
        "dest": "珠海金湾起降点",
        "weight_kg": 12,
        "fee_cny": "380.00",
        "status": "已签收",
    },
    "DK20261021002": {
        "origin": "广州黄埔起降点",
        "dest": "东莞松山湖起降点",
        "weight_kg": 5,
        "fee_cny": "160.50",
        "status": "运输中",
    },
}


@tool
def get_waybill(waybill_no: str) -> str:
    """按运单号查询运单概要。运单号形如 DK20261021001。"""
    record = _WAYBILLS.get(waybill_no.strip().upper())
    if record is None:
        # 明确返回「未找到」，不伪造数据
        return f"未找到运单 {waybill_no}"
    return (
        f"运单号 {waybill_no}：{record['origin']} → {record['dest']}，"
        f"载荷 {record['weight_kg']}kg，运费 {record['fee_cny']} 元，状态 {record['status']}"
    )


@tool
def calc_shipping_fee(weight_kg: float, unit_price_cny: float) -> str:
    """按重量与单价计算运费，返回两位小数的金额字符串。

    用 Decimal 计算，避免浮点误差导致的费用偏差。

    Args:
        weight_kg: 重量（公斤），必须为正。
        unit_price_cny: 每公斤单价（元），必须非负。
    """
    if weight_kg <= 0:
        return "错误：重量必须大于 0"
    if unit_price_cny < 0:
        return "错误：单价不能为负"
    try:
        fee = Decimal(str(weight_kg)) * Decimal(str(unit_price_cny))
    except InvalidOperation:
        return "错误：参数格式不合法"
    return f"{fee.quantize(Decimal('0.01'))} 元"


@tool
def days_until(target_date: str) -> str:
    """计算从今天到目标日期还有多少天。target_date 格式 YYYY-MM-DD。"""
    try:
        target = date.fromisoformat(target_date.strip())
    except ValueError:
        return "错误：日期格式应为 YYYY-MM-DD"
    delta = (target - date.today()).days
    if delta < 0:
        return f"目标日期已过 {abs(delta)} 天"
    return f"距今还有 {delta} 天"


WAYBILL_TOOLS = [get_waybill, calc_shipping_fee, days_until]

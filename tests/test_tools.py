"""工具层单元测试：纯函数，不涉及模型。"""

from app.agents.tools import calc_shipping_fee, days_until, get_waybill


class TestGetWaybill:
    def test_命中运单(self):
        out = get_waybill.invoke({"waybill_no": "DK20261021001"})
        assert "深圳南山起降点" in out
        assert "珠海金湾起降点" in out
        assert "12kg" in out

    def test_大小写与空白归一(self):
        out = get_waybill.invoke({"waybill_no": "  dk20261021001  "})
        assert "未找到" not in out

    def test_未找到时不编造数据(self):
        out = get_waybill.invoke({"waybill_no": "DK00000000000"})
        assert "未找到运单" in out
        assert "运费" not in out


class TestCalcShippingFee:
    def test_正常计算(self):
        assert calc_shipping_fee.invoke({"weight_kg": 12, "unit_price_cny": 32.5}) == "390.00 元"

    def test_浮点误差被消除(self):
        """必须选 float 会算错的输入，否则测不出 Decimal 的价值。

        1.15 * 0.1：float 得 0.11499999999999999，进位成 0.11；
        Decimal 精确得 0.115，进位成 0.12。本用例能区分两条实现路径。
        """
        assert calc_shipping_fee.invoke({"weight_kg": 1.15, "unit_price_cny": 0.1}) == "0.12 元"

    def test_浮点误差被消除_第二例(self):
        """1.005 * 7：float 得 7.034999999999999，Decimal 得 7.035 -> 7.04。"""
        assert calc_shipping_fee.invoke({"weight_kg": 1.005, "unit_price_cny": 7}) == "7.04 元"

    def test_浮点误差被消除_第三例(self):
        """0.285 * 7：float 得 1.9949999999999999，Decimal 得 1.995 -> 2.00。"""
        assert calc_shipping_fee.invoke({"weight_kg": 0.285, "unit_price_cny": 7}) == "2.00 元"

    def test_零重量报错(self):
        assert "重量必须大于 0" in calc_shipping_fee.invoke({"weight_kg": 0, "unit_price_cny": 10})

    def test_负重量报错(self):
        assert "重量必须大于 0" in calc_shipping_fee.invoke({"weight_kg": -1, "unit_price_cny": 10})

    def test_负单价报错(self):
        assert "单价不能为负" in calc_shipping_fee.invoke({"weight_kg": 1, "unit_price_cny": -0.01})


class TestDaysUntil:
    def test_格式错误(self):
        assert "YYYY-MM-DD" in days_until.invoke({"target_date": "2026/01/01"})

    def test_过去日期(self):
        assert "已过" in days_until.invoke({"target_date": "2000-01-01"})

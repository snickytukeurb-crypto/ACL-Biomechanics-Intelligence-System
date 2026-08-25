"""ตรึงแบบจำลองไว้กับชุดค่าอ้างอิงที่กำหนดไว้ล่วงหน้า

ค่าคาดหวังทุกตัวมาจากภายนอกไฟล์นี้ ไม่ได้คำนวณย้อนจากโค้ด ถ้ากรณีใดไม่ผ่าน
ให้ถือว่าแบบจำลองเปลี่ยนไป ไม่ใช่ค่าคาดหวังผิด และห้ามแก้ตัวเลขให้เข้ากับโค้ด
"""

import pytest

from acl.risk import BAND_HIGH, BAND_LOW, WEIGHTS, clamp01, risk_index, risk_level

# (θ, γ, ω, α, R ที่ต้องได้, ระดับ) — ค่า R ปัดทศนิยม 1 ตำแหน่ง
REFERENCE_CASES = [
    (171.2, 3.1, 36.5, -82.4, 8.4, "Low"),
    (109.4, 10.5, 145.8, -468.2, 38.0, "Moderate"),
    (81.6, 18.4, 242.7, -1087.5, 65.8, "High"),
    (160.1, 5.8, 54.1, -243.1, 16.4, "Low"),
    (120.1, 14.5, 169.2, -795.7, 47.1, "Moderate"),
    (89.3, 20.2, 236.0, -1137.5, 67.1, "High"),
    (114.1, 15.4, 181.0, -914.1, 51.4, "High"),
    (109.8, 15.6, 190.6, -880.3, 52.4, "High"),
    (95.6, 20.8, 228.4, -1024.1, 64.9, "High"),
]


@pytest.mark.parametrize("theta,valgus,omega,alpha,expected,level", REFERENCE_CASES)
def test_matches_reference_values(theta, valgus, omega, alpha, expected, level):
    result = risk_index(theta, valgus, omega, alpha)
    assert result == pytest.approx(expected, abs=0.05)
    assert risk_level(result) == level


def test_weights_sum_to_one():
    """ผลรวมน้ำหนักต้องเท่ากับ 1.00 มิฉะนั้นดัชนีจะหลุดช่วง 0-100"""
    assert sum(WEIGHTS) == pytest.approx(1.0)


def test_band_boundaries_come_from_per_variable_criteria():
    """แทนขอบเกณฑ์รายตัวแปรลงในสมการแล้วต้องได้จุดแบ่งใกล้ 25 และ 50"""
    assert risk_index(160, 8, 120, 400) == pytest.approx(25.44, abs=0.01)
    assert risk_index(120, 15, 200, 800) == pytest.approx(49.83, abs=0.01)


@pytest.mark.parametrize(
    "risk,level",
    [(0.0, "Low"), (24.9, "Low"), (BAND_LOW, "Moderate"), (49.9, "Moderate"),
     (BAND_HIGH, "High"), (100.0, "High")],
)
def test_level_boundaries_are_inclusive_upward(risk, level):
    assert risk_level(risk) == level


def test_index_stays_in_range_for_absurd_input():
    """ค่าที่หลุดกรอบต้องถูก clamp ไม่ใช่ทำให้เปอร์เซ็นต์เกิน 100 หรือติดลบ"""
    assert risk_index(-500, 900, 9e9, -9e9) == pytest.approx(100.0)
    assert risk_index(180, 0, 0, 0) == pytest.approx(0.0)
    assert risk_index(400, -50, 0, 0) == pytest.approx(0.0)


def test_negative_omega_counts_by_magnitude():
    """การงอเข่าทำให้ ω ติดลบ แต่ความเสี่ยงต้องเท่ากับกรณีเหยียดด้วยอัตราเดียวกัน"""
    assert risk_index(120, 10, -150, -500) == pytest.approx(risk_index(120, 10, 150, 500))


@pytest.mark.parametrize("value,expected", [(-0.5, 0.0), (0.0, 0.0), (0.4, 0.4), (1.0, 1.0), (7.0, 1.0)])
def test_clamp01(value, expected):
    assert clamp01(value) == expected

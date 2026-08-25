"""แบบจำลอง ACL Risk Index

ไฟล์นี้เป็นแหล่งอ้างอิงเดียวของค่าน้ำหนักและเกณฑ์แบ่งระดับทั้งระบบ
ห้ามคัดลอกค่าเหล่านี้ไปไว้ที่อื่น มิฉะนั้นสองที่จะเพี้ยนออกจากกันโดยไม่มีใครรู้
tests/test_risk.py ตรึงค่าทั้งหมดไว้กับชุดค่าอ้างอิงภายนอก
"""

from __future__ import annotations

# ค่าน้ำหนักเรียงตาม (θ, γ, ω, α) รวมกันได้ 1.00 พอดี
# γ ถ่วงมากที่สุดเพราะสัมพันธ์กับการบาดเจ็บมากที่สุดตามงานวิจัยที่ใช้อ้างอิง
WEIGHTS = (0.25, 0.35, 0.20, 0.20)

# ค่าอ้างอิงสำหรับปรับสเกลแต่ละตัวแปรให้อยู่ในช่วง [0, 1]
THETA_REF = 180.0  # องศา — ค่าเมื่อขาเหยียดตรง
VALGUS_REF = 30.0  # องศา
OMEGA_REF = 300.0  # องศา/วินาที
ALPHA_REF = 1500.0  # องศา/วินาที²

# เกณฑ์แบ่งระดับความเสี่ยง: Low < 25 ≤ Moderate < 50 ≤ High
BAND_LOW = 25.0
BAND_HIGH = 50.0


def clamp01(value: float) -> float:
    """จำกัดค่าให้อยู่ในช่วง [0, 1] ตามข้อกำหนดของแบบจำลอง"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalize(theta: float, valgus: float, omega: float, alpha: float):
    """ปรับสเกลตัวแปรทั้งสี่ให้อยู่ในช่วง [0, 1]

    ω และ α ใช้ขนาด (absolute value) เพราะการงอเข่าทำให้ θ ลดลง ค่า ω ที่ได้
    จากอนุพันธ์จึงติดลบ ถ้าไม่คิดขนาด จังหวะงอเข่าจะถูกนับเป็นความเสี่ยงศูนย์
    """
    return (
        clamp01((THETA_REF - theta) / THETA_REF),
        clamp01(valgus / VALGUS_REF),
        clamp01(abs(omega) / OMEGA_REF),
        clamp01(abs(alpha) / ALPHA_REF),
    )


def risk_index(
    theta: float, valgus: float, omega: float, alpha: float, weights=WEIGHTS
) -> float:
    """ค่าดัชนีความเสี่ยง R_ACL ในช่วง 0-100"""
    factors = normalize(theta, valgus, omega, alpha)
    return 100.0 * sum(w * f for w, f in zip(weights, factors))


def risk_level(risk: float) -> str:
    """จัดระดับความเสี่ยงเป็น Low / Moderate / High"""
    if risk < BAND_LOW:
        return "Low"
    if risk < BAND_HIGH:
        return "Moderate"
    return "High"

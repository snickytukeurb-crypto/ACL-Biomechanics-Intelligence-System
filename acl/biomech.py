"""ตัวแปรทางชีวกลศาสตร์ที่คำนวณจากพิกัดข้อต่อ"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def knee_angle(hip, knee, ankle) -> float:
    """มุมข้อเข่า θ (องศา)

    สร้างเวกเตอร์ต้นขาและหน้าแข้งจากข้อเข่าไปยังสะโพกและข้อเท้า
    แล้วหามุมระหว่างเวกเตอร์ด้วยผลคูณเชิงจุดร่วมกับฟังก์ชันตรีโกณมิติผกผัน
    ขาเหยียดตรงได้ 180 องศา
    """
    thigh = np.asarray(hip, dtype=float) - np.asarray(knee, dtype=float)
    shank = np.asarray(ankle, dtype=float) - np.asarray(knee, dtype=float)
    denominator = float(np.linalg.norm(thigh) * np.linalg.norm(shank))
    if denominator == 0.0:
        return float("nan")
    cosine = float(np.dot(thigh, shank)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def knee_valgus(hip, knee, ankle) -> float:
    """มุม Knee Valgus γ (องศา)

    γ คือการเบี่ยงของข้อเข่าเข้าด้านในเมื่อเทียบกับแนวของสะโพกและข้อเท้า
    บนระนาบหน้า ฟังก์ชันนี้จึงฉายทั้งสามจุดลงบนระนาบ xy แล้ววัด
    มุมที่ข้อเข่าเบี่ยงออกจากแนวสะโพก-ข้อเท้า โดยใช้ระยะตั้งฉากจากข้อเข่าถึงแนวนั้น
    ร่วมกับความยาวสะโพก-ข้อเท้าผ่านฟังก์ชัน atan2

    หมายเหตุสำคัญ: สูตรตามตัวอักษรในเล่ม (γ = 180 − |atan2| ระหว่างเวกเตอร์
    ต้นขากับหน้าแข้ง) ใช้ได้เฉพาะตอนขาเกือบเหยียดตรง เพราะเมื่อเข่างอลึกจนสะโพก
    ลดต่ำมาใกล้ระดับเข่า ภาพฉายของทั้งต้นขาและหน้าแข้งจะชี้ลงทางเดียวกัน
    มุมระหว่างกันจึงเข้าใกล้ศูนย์ และ γ พุ่งเข้าใกล้ 180 องศา
    วัดจากคลิปจริงพบว่า γ แบบนั้นแปรผันตาม (180 − θ) ที่สหสัมพันธ์ 0.95
    คือนับการงอเข่าซ้ำอีกครั้งแทนที่จะวัดการเบี่ยงเข้าด้านใน และให้ค่าถึง 165 องศา
    ซึ่งเกินพิสัย 0-30 องศาที่แบบจำลองใช้ปรับสเกล

    นิยามที่ใช้ที่นี่ให้ 0 องศาเมื่อข้อเข่าอยู่บนแนวสะโพก-ข้อเท้าพอดี
    และเพิ่มขึ้นตามขนาดการเบี่ยงจริง โดยไม่ขึ้นกับว่าเข่างอลึกเท่าใด
    """
    hip_xy = np.asarray(hip, dtype=float)[:2]
    knee_xy = np.asarray(knee, dtype=float)[:2]
    ankle_xy = np.asarray(ankle, dtype=float)[:2]

    axis = ankle_xy - hip_xy
    span = float(np.linalg.norm(axis))
    if span == 0.0:
        return float("nan")

    to_knee = knee_xy - hip_xy
    # ระยะตั้งฉากจากข้อเข่าถึงแนวสะโพก-ข้อเท้า หาจากขนาดผลคูณเชิงเวกเตอร์
    offset = abs(axis[0] * to_knee[1] - axis[1] * to_knee[0]) / span
    return math.degrees(math.atan2(offset, span))


def derivative(current: float, previous: float, dt: float) -> float:
    """ผลต่างสืบเนื่องแบบย้อนหลัง (Backward Finite Difference) — ขั้นที่ 5

    ใช้ทั้งกับความเร็วเชิงมุม (ω = Δθ/Δt) และความเร่งเชิงมุม (α = Δω/Δt)
    """
    if dt <= 0.0:
        return 0.0
    return (current - previous) / dt


class AngleSmoother:
    """ค่าเฉลี่ยเคลื่อนที่ของมุม เพื่อลดสัญญาณรบกวนก่อนนำไปหาอนุพันธ์

    การหาอนุพันธ์เชิงตัวเลขขยายสัญญาณรบกวนของ MediaPipe อย่างมาก
    ทำให้ ω และ α แกว่งจนค่าดัชนีความเสี่ยงกระโดดไปมาในแต่ละเฟรม
    ตั้ง window = 1 เพื่อปิดการกรอง จะได้อนุพันธ์ดิบที่ไม่ผ่านตัวกรองเลย
    """

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window ต้องมีค่าอย่างน้อย 1")
        self._values: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._values.append(value)
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()

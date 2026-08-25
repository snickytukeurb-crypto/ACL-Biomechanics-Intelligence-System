"""ติดตามผลรายเฟรม เลือก Critical Frame และส่งออกเป็นตาราง

Critical Frame คือเฟรมที่มุมข้อเข่ามีค่าน้อยที่สุดในช่วงที่ผู้ทดสอบลงสู่พื้นและรับน้ำหนัก
ซึ่งเป็นจังหวะที่ข้อเข่ารับแรงมากที่สุด จึงใช้ค่าดัชนี ณ เฟรมนั้นเป็นผลของการทดสอบหนึ่งครั้ง
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, fields

import pandas as pd

from .biomech import AngleSmoother, derivative, knee_angle, knee_valgus
from .risk import risk_index, risk_level

NAN = float("nan")


@dataclass(frozen=True)
class FrameRecord:
    """ผลการคำนวณของหนึ่งเฟรม ค่าเป็น NaN เมื่อมองไม่เห็นขาข้างนั้น"""

    frame_index: int
    time_s: float
    left_theta: float
    left_valgus: float
    left_omega: float
    left_alpha: float
    left_risk: float
    right_theta: float
    right_valgus: float
    right_omega: float
    right_alpha: float
    right_risk: float
    risk: float
    level: str
    worst_side: str

    @property
    def min_theta(self) -> float:
        """มุมข้อเข่าที่น้อยที่สุดของเฟรมนี้ ใช้คัดเลือก Critical Frame"""
        angles = [a for a in (self.left_theta, self.right_theta) if not math.isnan(a)]
        return min(angles) if angles else NAN


class LegTracker:
    """คำนวณ θ, γ, ω, α และดัชนีความเสี่ยงของขาหนึ่งข้างตามลำดับเฟรม"""

    def __init__(self, smoothing_window: int = 5):
        # กรองทั้งมุมและความเร็วเชิงมุม เพราะการหาอนุพันธ์สองครั้งขยายสัญญาณรบกวน
        # แบบทวีคูณ วัดจากคลิปจริงขณะยืนนิ่ง การกรอง ω เพิ่มลด |α| ที่ควรเป็นศูนย์
        # จากราว 218 เหลือราว 104 องศา/วินาที² (เปอร์เซ็นไทล์ที่ 95)
        self._theta_smoother = AngleSmoother(smoothing_window)
        self._omega_smoother = AngleSmoother(smoothing_window)
        self._previous_theta: float | None = None
        self._previous_omega = 0.0

    def reset(self) -> None:
        """ล้างสถานะเมื่อขาหายไปจากภาพ กันไม่ให้อนุพันธ์ข้ามช่วงที่ขาดหาย"""
        self._theta_smoother.reset()
        self._omega_smoother.reset()
        self._previous_theta = None
        self._previous_omega = 0.0

    def update(self, points, dt: float):
        """คืนค่า (θ, γ, ω, α, R) ของเฟรมนี้ หรือ None เมื่อไม่มีพิกัด"""
        if points is None:
            self.reset()
            return None

        hip, knee, ankle = points
        theta = self._theta_smoother.update(knee_angle(hip, knee, ankle))
        valgus = knee_valgus(hip, knee, ankle)

        omega = 0.0
        alpha = 0.0
        if self._previous_theta is not None:
            omega = self._omega_smoother.update(
                derivative(theta, self._previous_theta, dt)
            )
            alpha = derivative(omega, self._previous_omega, dt)
        self._previous_theta = theta
        self._previous_omega = omega

        return theta, valgus, omega, alpha, risk_index(theta, valgus, omega, alpha)


class Session:
    """รวบรวมผลของทั้งคลิปหรือทั้งช่วงที่เปิดกล้อง"""

    def __init__(self, fps: float, smoothing_window: int = 5):
        if fps <= 0:
            raise ValueError("fps ต้องมากกว่า 0")
        self.fps = fps
        self.dt = 1.0 / fps
        self._trackers = {
            "left": LegTracker(smoothing_window),
            "right": LegTracker(smoothing_window),
        }
        self.records: list[FrameRecord] = []
        self._elapsed = 0.0

    def update(self, frame_index: int, left_points, right_points, dt: float | None = None):
        """ประมวลผลหนึ่งเฟรม คืนค่า FrameRecord หรือ None เมื่อไม่เห็นขาทั้งสองข้าง

        dt มาจากอัตราเฟรมของวิดีโอ (Δt = 1/FPS) ส่วนโหมดกล้องสด
        ให้ส่งเวลาจริงที่วัดได้ระหว่างเฟรมเข้ามาแทน
        """
        step = self.dt if dt is None else dt
        time_s = self._elapsed
        self._elapsed += step  # ต้องเดินหน้าแม้เฟรมนี้จะมองไม่เห็นขา ไม่งั้นเฟรมถัดไปเวลาจะเพี้ยน
        left = self._trackers["left"].update(left_points, step)
        right = self._trackers["right"].update(right_points, step)
        if left is None and right is None:
            return None

        risks = {
            side: metrics[4]
            for side, metrics in (("left", left), ("right", right))
            if metrics is not None
        }
        worst_side = max(risks, key=risks.__getitem__)

        record = FrameRecord(
            frame_index=frame_index,
            time_s=time_s,
            **_leg_fields("left", left),
            **_leg_fields("right", right),
            risk=risks[worst_side],
            level=risk_level(risks[worst_side]),
            worst_side=worst_side,
        )
        self.records.append(record)
        return record

    @property
    def critical_frame(self) -> FrameRecord | None:
        """เฟรมที่มุมข้อเข่าน้อยที่สุดตลอดการทดสอบ"""
        candidates = [r for r in self.records if not math.isnan(r.min_theta)]
        if not candidates:
            return None
        return min(candidates, key=lambda record: record.min_theta)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            # คงคอลัมน์ไว้เสมอ ผู้เรียกจะได้ไม่ต้องแยกกรณีตารางว่าง
            return pd.DataFrame(columns=[f.name for f in fields(FrameRecord)])
        return pd.DataFrame([asdict(record) for record in self.records])

    def to_csv_bytes(self) -> bytes:
        return self.to_dataframe().to_csv(index=False).encode("utf-8-sig")


def _leg_fields(side: str, metrics) -> dict:
    """แปลงผลของขาหนึ่งข้างเป็นคอลัมน์ที่มีคำนำหน้า left_ หรือ right_"""
    names = ("theta", "valgus", "omega", "alpha", "risk")
    values = metrics if metrics is not None else (NAN,) * len(names)
    return {f"{side}_{name}": value for name, value in zip(names, values)}


class LevelStabiliser:
    """ระดับความเสี่ยงที่ทั้งนิ่งพอจะอ่านได้ และไม่พลาดจังหวะเสี่ยงที่เกิดขึ้นสั้น ๆ

    ทำงานสองขั้น

    ขั้นที่ 1 — **มัธยฐานช่วงสั้น** ค่า R รายเฟรมแกว่งเพราะ ω และ α มาจากอนุพันธ์
    เชิงตัวเลข ถ้านำมาแสดงตรง ๆ ป้ายจะกะพริบสลับ Low/Moderate/High จนอ่านไม่ทัน
    มัธยฐานทนต่อค่าโดดเดี่ยวเพียงเฟรมเดียวได้ดีกว่าค่าเฉลี่ย แต่ต้องใช้หน้าต่างสั้น
    เพราะจังหวะเสี่ยงจริงกินเวลาไม่นาน

    ขั้นที่ 2 — **ค้างค่าสูงสุด** วัดจากคลิปจริงของกลุ่มตัวอย่าง ช่วงที่ R ≥ 50
    ติดต่อกันยาวเพียง 0.04-0.14 วินาที ซึ่งสั้นเกินกว่าคนจะอ่านป้ายทัน และถ้าใช้
    มัธยฐานหน้าต่าง 0.5 วินาที ป้ายจะไม่ขึ้น High เลยแม้แต่ครั้งเดียวทั้งคลิป
    จึงค้างค่าสูงสุดของสัญญาณที่กรองแล้วไว้ hold_seconds เพื่อให้จังหวะลงพื้น
    ปรากฏบนจอนานพอจะเห็น

    ค้างค่าจากสัญญาณที่ผ่านมัธยฐานแล้ว ไม่ใช่ค่าดิบ มิฉะนั้นสัญญาณรบกวนเฟรมเดียว
    จะค้างป้ายไว้ทั้งช่วง ตั้ง hold_seconds = 0 เพื่อปิดการค้างค่า

    เก็บตามเวลาไม่ใช่ตามจำนวนเฟรม เพราะอัตราเฟรมของกล้องเว็บแคมไม่คงที่
    ค่าที่คืนมาใช้เฉพาะการแสดงผล ส่วน Session.records และ CSV ยังเก็บค่ารายเฟรมดิบไว้
    """

    def __init__(self, window_seconds: float = 0.25, hold_seconds: float = 2.0):
        if window_seconds <= 0:
            raise ValueError("window_seconds ต้องมากกว่า 0")
        if hold_seconds < 0:
            raise ValueError("hold_seconds ต้องไม่ติดลบ")
        self.window_seconds = window_seconds
        self.hold_seconds = hold_seconds
        self._samples: deque[tuple[float, float]] = deque()
        self._filtered: deque[tuple[float, float]] = deque()

    def update(self, risk: float, timestamp_s: float) -> tuple[float, str]:
        """คืนค่า (R ที่ใช้แสดงผล, ระดับความเสี่ยง) สำหรับป้ายบนหน้าจอ"""
        self._samples.append((timestamp_s, risk))
        while self._samples and self._samples[0][0] < timestamp_s - self.window_seconds:
            self._samples.popleft()
        median = statistics.median(value for _, value in self._samples)

        self._filtered.append((timestamp_s, median))
        while self._filtered and self._filtered[0][0] < timestamp_s - self.hold_seconds:
            self._filtered.popleft()
        display = max(value for _, value in self._filtered)

        return display, risk_level(display)

    def reset(self) -> None:
        self._samples.clear()
        self._filtered.clear()

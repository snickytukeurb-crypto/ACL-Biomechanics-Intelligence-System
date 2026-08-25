"""ห่อหุ้ม MediaPipe Pose Landmarker และดึงพิกัดข้อต่อของขา

MediaPipe รุ่นใหม่ถอด API เดิม mp.solutions.pose ออกไปแล้ว โมดูลนี้จึงใช้
MediaPipe Tasks (PoseLandmarker) ซึ่งเป็นทางที่ Google รองรับอยู่ในปัจจุบัน
และให้จุด landmark ชุดเดียวกัน 33 จุด พร้อมพิกัดสามมิติหน่วยเมตรและค่า visibility
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"

# จุด landmark หมายเลข 23-28 คือ สะโพก เข่า และข้อเท้า ทั้งสองข้าง
LEG_LANDMARKS = {"left": (23, 25, 27), "right": (24, 26, 28)}
SIDES = tuple(LEG_LANDMARKS)


class ModelNotFound(FileNotFoundError):
    """ยังไม่ได้ดาวน์โหลดไฟล์โมเดลของ MediaPipe"""


def create_pose(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_path=None,
):
    """สร้างตัวตรวจจับท่าทางในโหมด VIDEO ต้องเรียก close() เมื่อใช้เสร็จ"""
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise ModelNotFound(
            f"ไม่พบไฟล์โมเดลที่ {path}\nดาวน์โหลดด้วย:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {MODEL_URL}"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms: float):
    """ประมวลผลเฟรมจาก OpenCV (BGR) ด้วย MediaPipe ซึ่งรับภาพแบบ RGB

    โหมด VIDEO บังคับให้เวลาประทับ (มิลลิวินาที) ต้องเพิ่มขึ้นทุกเฟรม
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, int(timestamp_ms))


def world_leg(results, side: str, min_visibility: float = 0.5):
    """พิกัดสามมิติหน่วยเมตรของ (สะโพก, เข่า, ข้อเท้า) หรือ None ถ้ามองเห็นไม่ชัด

    ใช้ pose_world_landmarks สำหรับการคำนวณ ไม่ใช่ pose_landmarks เพราะพิกัด
    แบบหลังถูกนอร์มัลไลซ์ด้วยความกว้างและความสูงของภาพคนละค่า มุมที่คำนวณได้
    จึงผิดเพี้ยนทุกครั้งที่ภาพไม่ใช่จัตุรัส

    การคืนค่า None เมื่อ visibility ต่ำ ทำให้เฟรมที่ข้อต่อถูกบัง ถูกข้ามไป
    แทนที่จะปล่อยค่ามุมที่ไม่มีความหมายเข้าสู่แบบจำลอง
    """
    groups = getattr(results, "pose_world_landmarks", None)
    if not groups:
        return None
    landmarks = groups[0]
    points = []
    for index in LEG_LANDMARKS[side]:
        landmark = landmarks[index]
        if landmark.visibility < min_visibility:
            return None
        points.append(np.array([landmark.x, landmark.y, landmark.z], dtype=float))
    return tuple(points)


def pixel_leg(results, side: str, width: int, height: int):
    """พิกัดพิกเซลของขาข้างที่ระบุ ใช้สำหรับวาดภาพซ้อนเท่านั้น"""
    groups = getattr(results, "pose_landmarks", None)
    if not groups:
        return None
    landmarks = groups[0]
    return tuple(
        (int(landmarks[index].x * width), int(landmarks[index].y * height))
        for index in LEG_LANDMARKS[side]
    )


def draw_leg(frame, points, color) -> None:
    """วาดเส้นต้นขาและหน้าแข้งพร้อมจุดข้อต่อลงบนเฟรม"""
    hip, knee, ankle = points
    cv2.line(frame, hip, knee, color, 3)
    cv2.line(frame, knee, ankle, color, 3)
    for point in (hip, knee, ankle):
        cv2.circle(frame, point, 6, color, -1)

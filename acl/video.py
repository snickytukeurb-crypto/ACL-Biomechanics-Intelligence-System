"""เขียนวิดีโอผลการวิเคราะห์ และย่อขนาดด้วย ffmpeg

ใช้ตัวเข้ารหัส H.264 (avc1) เพราะเป็นรูปแบบเดียวที่เบราว์เซอร์เล่นได้แน่นอน
ตัวเลือกอย่าง mp4v เปิดไฟล์ได้ก็จริงแต่เบราว์เซอร์ส่วนใหญ่เล่นไม่ออก
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2

FFMPEG_CANDIDATES = (
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "ffmpeg",
)

# ป้ายที่ใช้บนหน้าเว็บ -> ความกว้างเป็นพิกเซล (None คือคงขนาดเดิมของคลิป)
SIZE_CHOICES: dict[str, int | None] = {
    "640px": 640,
    "960px": 960,
    "ต้นฉบับ": None,
}
DEFAULT_SIZE = "960px"


class EncoderNotAvailable(RuntimeError):
    """เครื่องนี้ไม่มีตัวเข้ารหัสที่จำเป็น"""


def find_ffmpeg() -> str | None:
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def open_writer(path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """เปิดไฟล์วิดีโอ H.264 สำหรับเขียนเฟรมที่วาดผลแล้ว

    size คือ (กว้าง, สูง) ตามที่ OpenCV ต้องการ
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        raise EncoderNotAvailable(
            "เปิดตัวเข้ารหัส H.264 (avc1) ไม่ได้ จึงบันทึกวิดีโอผลลัพธ์ไม่ได้"
        )
    return writer


def rescale(source, target, width: int, timeout: float = 300.0) -> None:
    """ย่อวิดีโอให้กว้างตามที่กำหนด โดยคงอัตราส่วนเดิม

    ใช้ -2 กับความสูงเพื่อให้ ffmpeg เลือกค่าที่หารด้วยสองลงตัว
    ซึ่ง H.264 แบบ yuv420p บังคับไว้
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise EncoderNotAvailable(
            "ไม่พบ ffmpeg ในเครื่อง จึงย่อขนาดวิดีโอไม่ได้ เลือก 'ต้นฉบับ' แทนได้"
        )
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-i", str(source),
            "-vf", f"scale={width}:-2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            str(target),
        ],
        check=True, capture_output=True, timeout=timeout,
    )

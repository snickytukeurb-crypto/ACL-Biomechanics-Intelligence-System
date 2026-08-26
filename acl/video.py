"""เขียนวิดีโอผลการวิเคราะห์ และย่อขนาดด้วย ffmpeg

ใช้ตัวเข้ารหัส H.264 (avc1) เพราะเป็นรูปแบบเดียวที่เบราว์เซอร์เล่นได้แน่นอน
ตัวเลือกอย่าง mp4v เปิดไฟล์ได้ก็จริงแต่เบราว์เซอร์ส่วนใหญ่เล่นไม่ออก
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    """หา ffmpeg ในเครื่อง ถ้าไม่มีจึงใช้ไบนารีที่มากับแพ็กเกจ imageio-ffmpeg

    ลำดับนี้สำคัญ: เครื่องที่ลง ffmpeg เองไว้แล้วต้องได้ตัวนั้น เพราะมันมี ffprobe
    อยู่ข้าง ๆ ให้ชุดทดสอบเรียกใช้ ส่วนไบนารีของ imageio-ffmpeg มีแต่ ffmpeg อย่างเดียว
    """
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return bundled_ffmpeg()


def bundled_ffmpeg() -> str | None:
    """ไบนารี ffmpeg ที่มากับแพ็กเกจ imageio-ffmpeg คืน None เมื่อไม่มีให้ใช้"""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # ไม่ได้ติดตั้งไว้ หรือแพลตฟอร์มนี้ไม่มีไบนารีให้
        return None


class FfmpegWriter:
    """เขียนวิดีโอ H.264 โดยส่งเฟรมดิบเข้า ffmpeg ทางไปป์

    ใช้แทน cv2.VideoWriter บน Linux เพราะตัว OpenCV ที่แจกเป็น wheel สำหรับ Linux
    ไม่มีตัวเข้ารหัส H.264 มาให้ (ติดเรื่องสิทธิบัตร) จะเขียน avc1 ไม่ได้เลย หรือได้ไฟล์
    ที่เบราว์เซอร์เล่นไม่ออก ส่วนบน macOS ตัว OpenCV เรียก VideoToolbox ได้ จึงใช้ทางเดิม

    รับ-ส่งหน้าตาเหมือน cv2.VideoWriter เท่าที่หน้าเว็บใช้จริง คือ write() กับ release()
    """

    def __init__(self, ffmpeg: str, path, fps: float, size: tuple[int, int]):
        width, height = size
        self._process = subprocess.Popen(
            [
                ffmpeg, "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{max(fps, 1.0):.6f}",
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                str(path),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        # ffmpeg ที่ตายกลางคันทำให้ไปป์ขาด ปล่อยผ่านไปเงียบ ๆ ดีกว่าล้มการวิเคราะห์ทั้งรอบ
        # ทิ้ง เพราะผลตัวเลขยังใช้ได้ ต่อให้ไฟล์วิดีโอไม่สมบูรณ์
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def release(self) -> None:
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        self._process.wait(timeout=120)


def open_writer(path, fps: float, size: tuple[int, int]):
    """เปิดไฟล์วิดีโอ H.264 สำหรับเขียนเฟรมที่วาดผลแล้ว

    size คือ (กว้าง, สูง) ตามที่ OpenCV ต้องการ
    """
    if sys.platform.startswith("linux"):
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise EncoderNotAvailable(
                "ไม่พบ ffmpeg ในเครื่อง จึงบันทึกวิดีโอผลลัพธ์ไม่ได้"
            )
        return FfmpegWriter(ffmpeg, path, fps, size)

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

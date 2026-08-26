"""ตรวจการเขียนวิดีโอผลลัพธ์และการย่อขนาด

วิดีโอต้องเป็น H.264 เท่านั้น เพราะเป็นรูปแบบเดียวที่เบราว์เซอร์เล่นได้แน่นอน
ชุดทดสอบจึงอ่านโคเดกจริงจากไฟล์ที่เขียนออกมา ไม่ใช่แค่ดูว่าไฟล์ถูกสร้าง
"""

import json
import subprocess

import numpy as np
import pytest

from acl import video


def _probe(path) -> dict:
    """อ่านคุณสมบัติของสตรีมวิดีโอด้วย ffprobe"""
    output = subprocess.run(
        [video.find_ffmpeg().replace("ffmpeg", "ffprobe"), "-v", "error",
         "-show_entries", "stream=codec_name,width,height,pix_fmt",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output)["streams"][0]


def _write_clip(path, frames=12, size=(320, 240)):
    writer = video.open_writer(path, 25.0, size)
    for index in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, : index * 10] = (40, 180, 60)
        writer.write(frame)
    writer.release()


def test_size_choices_cover_original():
    """ตัวเลือก 'ต้นฉบับ' ต้องเป็น None เพื่อบอกว่าไม่ต้องย่อ"""
    assert video.SIZE_CHOICES["ต้นฉบับ"] is None
    assert video.SIZE_CHOICES[video.DEFAULT_SIZE] == 960


def test_writer_produces_browser_playable_h264(tmp_path):
    target = tmp_path / "clip.mp4"
    _write_clip(target)
    stream = _probe(target)
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert target.stat().st_size > 0


@pytest.mark.skipif(video.find_ffmpeg() is None, reason="ต้องมี ffmpeg จึงจะย่อขนาดได้")
def test_rescale_keeps_aspect_ratio(tmp_path):
    source, target = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _write_clip(source, size=(640, 360))
    video.rescale(source, target, 320)
    stream = _probe(target)
    assert (stream["width"], stream["height"]) == (320, 180)
    assert stream["codec_name"] == "h264"


@pytest.mark.skipif(video.find_ffmpeg() is None, reason="ต้องมี ffmpeg จึงจะย่อขนาดได้")
def test_rescale_forces_even_height(tmp_path):
    """H.264 แบบ yuv420p บังคับให้ความกว้างและความสูงหารสองลงตัว"""
    source, target = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _write_clip(source, size=(640, 362))
    video.rescale(source, target, 300)
    stream = _probe(target)
    assert stream["height"] % 2 == 0


def test_missing_ffmpeg_reports_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "FFMPEG_CANDIDATES", ("/nonexistent/ffmpeg",))
    # ต้องปิดไบนารีสำรองของ imageio-ffmpeg ด้วย ไม่งั้นจะยังหา ffmpeg เจอเสมอ
    monkeypatch.setattr(video, "bundled_ffmpeg", lambda: None)
    with pytest.raises(video.EncoderNotAvailable):
        video.rescale(tmp_path / "a.mp4", tmp_path / "b.mp4", 320)

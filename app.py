"""ระบบวิเคราะห์ความเสี่ยงการบาดเจ็บเอ็นไขว้หน้าหัวเข่า (ACL)

ไฟล์นี้ทำหน้าที่เป็นส่วนติดต่อผู้ใช้เท่านั้น การคำนวณทั้งหมดอยู่ในแพ็กเกจ acl/
เรียกใช้ด้วย:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import cv2
import streamlit as st

from acl import pose as pose_module
from acl import report
from acl import video
from acl.risk import ALPHA_REF, OMEGA_REF, VALGUS_REF
from acl.session import LevelStabiliser, Session

# ---------------------------------------------------------------- ตั้งค่าหลังบ้าน
# ค่าเหล่านี้ตั้งใจไม่เปิดให้ปรับจากหน้าจอ เพื่อให้ทุกการทดสอบใช้เงื่อนไขเดียวกัน
# ที่มาของแต่ละค่าอธิบายไว้ใน README.md หัวข้อ "รายละเอียดการคำนวณที่ควรทราบ"
SMOOTHING_WINDOW = 5  # กรอง θ และ ω ก่อนหาอนุพันธ์; ตั้งเป็น 1 = ปิดการกรอง
MIN_VISIBILITY = 0.5  # ข้ามเฟรมที่ MediaPipe เห็นสะโพก เข่า หรือข้อเท้าไม่ชัดพอ
MODEL_CONFIDENCE = 0.5  # เกณฑ์ตรวจพบและติดตามท่าทางของ MediaPipe
FPS_FALLBACK = 30.0  # ใช้เมื่อไฟล์วิดีโอไม่แจ้งอัตราเฟรมมา
LEVEL_WINDOW_SECONDS = 0.25  # หามัธยฐานของ R ก่อนแสดงป้าย ลดการกะพริบของระดับ
LEVEL_HOLD_SECONDS = 2.0  # ค้างระดับสูงสุดไว้ เพราะจังหวะเสี่ยงจริงกินเวลาเพียง ~0.1 วินาที

# เซิร์ฟเวอร์ที่โฮสต์แอป (เช่น Streamlit Community Cloud) ไม่มีกล้องต่ออยู่ ปุ่มเว็บแคม
# จึงถูกซ่อนไปเลย ดีกว่าปล่อยให้ผู้ใช้กดแล้วเจอ error ตอนเปิดกล้องไม่ได้
# ตั้ง ACL_ENABLE_WEBCAM=1 เพื่อบังคับเปิดเมื่อรันบน Linux ที่มีกล้องจริง
if os.environ.get("ACL_ENABLE_WEBCAM"):
    WEBCAM_AVAILABLE = os.environ["ACL_ENABLE_WEBCAM"] == "1"
elif sys.platform.startswith("linux"):
    WEBCAM_AVAILABLE = os.path.exists("/dev/video0")
else:
    WEBCAM_AVAILABLE = True

PREVIEW_WIDTH = 720
# สัดส่วนความกว้าง (วิดีโอ : ตารางตัวเลข) ระหว่างวิเคราะห์
# วิดีโอยืดเต็มคอลัมน์ของตัวเอง ตัวเลขตัวแรกจึงคุมขนาดภาพโดยตรง
# ลดตัวแรกลง (เช่น 5, 5) ถ้าอยากให้ภาพเล็กลงอีก
LIVE_LAYOUT_RATIO = (6, 5)
# ความกว้างของเครื่องเล่นวิดีโอตอนสรุปผล (พิกเซล)
# st.video ตั้งต้นเป็น "stretch" คือเต็มความกว้างหน้าเว็บ ซึ่งใหญ่เกินจำเป็น
# คนละเรื่องกับตัวเลือก "ขนาดวิดีโอ" ที่คุมความละเอียดของไฟล์ ไม่ใช่ขนาดที่แสดง
RESULT_VIDEO_WIDTH = 560
LEVEL_COLORS = {"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"}
# OpenCV ใช้ลำดับสี BGR
LEG_COLORS = {"left": (241, 102, 99), "right": (94, 197, 34)}
LEVEL_BGR = {"Low": (94, 197, 34), "Moderate": (11, 158, 245), "High": (68, 68, 239)}
LEVEL_THAI = {"Low": "ความเสี่ยงต่ำ", "Moderate": "ความเสี่ยงปานกลาง", "High": "ความเสี่ยงสูง"}

st.set_page_config(page_title="ACL Biomechanics Intelligence System", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #0f172a; }
h1, h2, h3 { color: #f8fafc; }
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px;
}
section[data-testid="stSidebar"] { background: #111827; }
.critical-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px 26px;
}
.critical-score { font-size: 58px; font-weight: 700; line-height: 1.1; }
.critical-level { font-size: 20px; font-weight: 600; letter-spacing: 0.04em; }
.critical-note { color: #94a3b8; font-size: 14px; margin-top: 6px; }
/* ป้ายระดับความเสี่ยงระหว่างวิเคราะห์อยู่ในคอลัมน์แคบกว่าการ์ด Critical Frame ตอนสรุปผล
   จึงใช้ตัวเลขเล็กลง (34px แทน 58px) ไม่งั้นตัวเลขจะล้นคอลัมน์ */
.badge-compact { font-size: 34px; font-weight: 700; line-height: 1.1; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- ค่าเริ่มต้นของสถานะ
# ต้องตั้งก่อนบล็อกใด ๆ ที่อ่าน st.session_state.running เพื่อตัดสินใจว่าจะแสดงผลส่วนไหน
st.session_state.setdefault("running", False)
st.session_state.setdefault("session", None)
st.session_state.setdefault("critical_image", None)
st.session_state.setdefault("report_pdf", None)
st.session_state.setdefault("source_label", "")
st.session_state.setdefault("participant", None)
# เก็บแหล่งข้อมูลไว้ตอนกดเริ่ม เพราะวิดเจ็ตเลือกไฟล์/โหมดจะถูกซ่อนระหว่างวิเคราะห์ (ดูจุดกดเริ่มด้านล่าง)
st.session_state.setdefault("mode", None)
st.session_state.setdefault("upload_name", None)
st.session_state.setdefault("upload_bytes", None)
st.session_state.setdefault("video_path", None)
st.session_state.setdefault("video_variants", {})

# ---------------------------------------------------------------- แถบข้าง
# วางไว้นอกเงื่อนไข running เพราะเป็นแผงอ้างอิงที่ไม่กินพื้นที่แนวตั้งของเนื้อหาหลัก
# จึงไม่ขัดกับเป้าหมาย "จอเดียวจบ" ระหว่างวิเคราะห์ และเห็นเกณฑ์ได้ตลอด
st.sidebar.subheader("เกณฑ์แบ่งระดับ")
st.sidebar.markdown(
    "- **Low** — R < 25\n- **Moderate** — 25 ≤ R < 50\n- **High** — R ≥ 50\n\n"
    f"ค่าอ้างอิงปรับสเกล: γ {VALGUS_REF:.0f}°, ω {OMEGA_REF:.0f}°/s, α {ALPHA_REF:.0f}°/s²"
)

if not st.session_state.running:
    st.title("ACL Biomechanics Intelligence System")
    st.caption(
        "ประเมินความเสี่ยงการบาดเจ็บเอ็นไขว้หน้าหัวเข่าแบบเวลาจริง "
        "จากมุมข้อเข่า Knee Valgus ความเร็วเชิงมุม และความเร่งเชิงมุม"
    )

    # ------------------------------------------------------ ประตูข้อมูลผู้ทดสอบ
    # ต้องกรอกก่อนใช้งานหน้าจอส่วนอื่น เพื่อให้ทุกคลิปที่บันทึกผลมีเจ้าของ
    # ค่าทั้งสี่เป็นข้อมูลอ้างอิงในรายงาน/CSV เท่านั้น ห้ามไหลเข้าสูตรคำนวณความเสี่ยงเด็ดขาด
    if st.session_state.participant is None:
        with st.form("participant_form"):
            st.subheader("ข้อมูลผู้ทดสอบ")
            age = st.number_input("อายุ (ปี)", min_value=10, max_value=100, value=17, step=1)
            gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
            weight_kg = st.number_input(
                "น้ำหนัก (กก.)", min_value=20.0, max_value=200.0, value=60.0, step=0.5
            )
            height_cm = st.number_input(
                "ส่วนสูง (ซม.)", min_value=100.0, max_value=250.0, value=170.0, step=0.5
            )
            submitted = st.form_submit_button("เริ่มใช้งาน")
        if submitted:
            st.session_state.participant = report.Participant(
                age=age, gender=gender, weight_kg=weight_kg, height_cm=height_cm
            )
            st.rerun()
        st.stop()

    chip_column, edit_column = st.columns([5, 1])
    chip_column.markdown(f":material/person: **ผู้ทดสอบ:** {st.session_state.participant.label}")
    if edit_column.button("แก้ไขข้อมูลผู้ทดสอบ"):
        st.session_state.participant = None
        st.rerun()

    # ------------------------------------------------------------------ แหล่งข้อมูล
    st.subheader("แหล่งข้อมูล")
    if WEBCAM_AVAILABLE:
        mode = st.radio("เลือกวิธีรับข้อมูล", ["ไฟล์วิดีโอ", "กล้องเว็บแคม"], horizontal=True)
    else:
        mode = "ไฟล์วิดีโอ"
        st.caption("เครื่องที่รันแอปนี้ไม่มีกล้องต่ออยู่ จึงใช้ได้เฉพาะการอัปโหลดคลิป")

    uploaded_video = None
    if mode == "ไฟล์วิดีโอ":
        uploaded_video = st.file_uploader(
            "อัปโหลดคลิป", type=["mp4", "avi", "mov", "mkv", "m4v"]
        )

    start_column, _ = st.columns([1, 5])
    if start_column.button("เริ่มวิเคราะห์", type="primary"):
        st.session_state.running = True
        st.session_state.session = None
        st.session_state.critical_image = None
        st.session_state.report_pdf = None
        st.session_state.source_label = (
            uploaded_video.name if uploaded_video is not None else "กล้องเว็บแคม"
        )
        # จับค่าของวิดเจ็ตไว้ใน session_state เพราะบล็อกด้านล่างนี้จะถูกซ่อนระหว่างวิเคราะห์
        # (ต้องใช้ getvalue() ไม่ใช่ read() เพราะ read() คืนค่าว่างในรอบที่สอง
        # เมื่อตัวชี้ของบัฟเฟอร์อยู่ท้ายไฟล์แล้ว)
        st.session_state.mode = mode
        st.session_state.upload_name = uploaded_video.name if uploaded_video is not None else None
        st.session_state.upload_bytes = (
            uploaded_video.getvalue() if uploaded_video is not None else None
        )
        st.rerun()


def draw_risk_banner(frame, risk: float, level: str) -> None:
    """วาดแถบสีสัญญาณไฟจราจรบอกระดับความเสี่ยงลงบนเฟรม

    แถบนี้ติดไปกับภาพ ไม่ใช่วางข้างจอ ทำให้ผู้ทดสอบที่ยืนห่างจากจอ 2-3 เมตร
    อ่านผลได้เอง โดยไม่ต้องเดินเข้ามาดูตัวเลขใกล้ ๆ และติดไปกับวิดีโอที่บันทึกด้วย
    """
    height, width = frame.shape[:2]
    scale = width / 960.0  # ปรับขนาดตามความกว้างภาพให้อ่านออกทั้ง 720p และ 1080p
    cv2.rectangle(frame, (0, 0), (width, int(96 * scale)), LEVEL_BGR[level], -1)

    text = f"{level.upper()}   R {risk:.0f}"
    origin = (int(24 * scale), int(66 * scale))
    font_scale = 1.8 * scale
    # วาดขอบดำก่อนแล้วทับด้วยตัวอักษรขาว ให้อ่านออกบนพื้นทั้งเขียว เหลือง และแดง
    for color, thickness in (((0, 0, 0), int(9 * scale)), ((255, 255, 255), int(4 * scale))):
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, max(2, thickness), cv2.LINE_AA,
        )


def draw_metric_strip(frame, record) -> None:
    """เขียนค่าทั้งสี่ตัวแปรของทั้งสองขาลงท้ายเฟรม

    ทำให้วิดีโอที่บันทึกไว้อธิบายตัวเองได้ครบทุกเฟรม ดาวน์โหลดไปเปิดที่ไหน
    หรือหยุดดูเฟรมไหนก็เห็นค่าของเฟรมนั้นทันที โดยไม่ต้องเปิดตารางเทียบ

    วางไว้ใต้แถบสัญญาณไฟ ไม่ใช่ท้ายภาพ เพราะแถบควบคุมของเครื่องเล่นวิดีโอ
    ลอยทับขอบล่างอยู่ ถ้าวางไว้ล่างจะถูกบังตอนหยุดเล่นหรือเลื่อนเมาส์ไปโดน

    ใช้ป้ายภาษาอังกฤษเพราะ cv2.putText รองรับเฉพาะอักษร ASCII
    ตัวอักษรกรีก (θ γ ω α) และภาษาไทยจะกลายเป็นสี่เหลี่ยม
    """
    height, width = frame.shape[:2]
    scale = width / 960.0
    line_height = int(30 * scale)
    top = int(96 * scale)  # ความสูงของแถบสัญญาณไฟด้านบน
    bottom = top + line_height * 2 + int(12 * scale)

    # พื้นหลังทึบแสงบางส่วน ให้ตัวเลขอ่านออกไม่ว่าฉากหลังจะสว่างหรือมืด
    panel = frame[top:bottom, 0:width].copy()
    cv2.rectangle(panel, (0, 0), (width, bottom - top), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.55, frame[top:bottom, 0:width], 0.45, 0,
                    frame[top:bottom, 0:width])

    def value(name, side):
        number = getattr(record, f"{side}_{name}")
        return "  --  " if number != number else f"{number:7.1f}"

    for index, (side, tag) in enumerate((("left", "L"), ("right", "R"))):
        text = (
            f"{tag}  KNEE {value('theta', side)}  VALGUS {value('valgus', side)}"
            f"  VEL {value('omega', side)}  ACC {value('alpha', side)}"
        )
        cv2.putText(
            frame, text,
            (int(16 * scale), top + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, (255, 255, 255),
            max(1, int(2 * scale)), cv2.LINE_AA,
        )


def build_live_panel(frames_slot):
    """สร้างที่ว่างสำหรับอัปเดตค่าระหว่างวิเคราะห์ วางวิดีโอซ้าย ตัวเลขขวา

    ก่อนหน้านี้ทุกอย่างเรียงต่อกันแนวตั้ง (วิดีโอ แล้วค่อยตัวเลข) ซึ่งเกินความสูงจอ
    จอเดียวจึงดูวิดีโอกับตัวเลขพร้อมกันไม่ได้ ที่นี่ frames_slot มาจาก header
    แถบบนสุด (ดูจุดกดปุ่มหยุด) เพราะจำนวนเฟรมย้ายไปอยู่ที่นั่นแทนที่จะอยู่ใต้วิดีโอ
    """
    video_column, metrics_column = st.columns(list(LIVE_LAYOUT_RATIO))
    frame_slot = video_column.empty()
    with metrics_column:
        badge_slot = st.empty()
        # หนึ่งแถวต่อหนึ่งตัวแปร (θ, γ, ω, α) ขาซ้าย-ขวาอยู่คู่กัน เทียบกันได้ทันที
        # ต่างจากเดิมที่วางค่าขาซ้ายทั้งสี่ตัวก่อนแล้วค่อยขาขวา ซึ่งเทียบกันยาก
        rows = []
        for _ in range(4):
            left_column, right_column = st.columns(2)
            rows.append((left_column.empty(), right_column.empty()))
    return {
        "badge": badge_slot,
        "frame": frame_slot,
        "frames": frames_slot,
        "rows": rows,
        "progress": st.empty(),
    }


def update_live_panel(panel, record, display_risk, display_level, analysed_frames, progress):
    """อัปเดตป้ายระดับความเสี่ยงและค่าตัวแปรรายขา

    ป้ายใช้ค่าที่ผ่าน LevelStabiliser แล้ว ส่วนตัวเลขรายขาเป็นค่าของเฟรมปัจจุบันจริง ๆ
    """
    color = LEVEL_COLORS[display_level]
    panel["badge"].markdown(
        f"""
<div class="critical-card" style="border-color:{color};">
  <div style="color:#94a3b8;font-size:12px;letter-spacing:.08em;">
    ระดับความเสี่ยงขณะนี้ (ค้างค่าสูงสุด {LEVEL_HOLD_SECONDS:.0f} วินาที)
  </div>
  <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
    <div class="badge-compact" style="color:{color};">{display_level.upper()}</div>
    <div style="font-size:20px;font-weight:600;color:{color};">R {display_risk:.1f}</div>
  </div>
  <div class="critical-note">{LEVEL_THAI[display_level]}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    panel["frames"].caption(f"วิเคราะห์แล้ว {analysed_frames} เฟรม")
    variables = (
        ("θ", "theta", "°"),
        ("γ", "valgus", "°"),
        ("ω", "omega", "°/s"),
        ("α", "alpha", "°/s²"),
    )
    for (symbol, attribute, unit), (left_slot, right_slot) in zip(variables, panel["rows"]):
        for side, slot in (("left", left_slot), ("right", right_slot)):
            label = "ซ้าย" if side == "left" else "ขวา"
            value = getattr(record, f"{side}_{attribute}")
            slot.metric(
                f"{symbol} {label} ({unit})", "—" if value != value else f"{value:.1f}"
            )
    if progress is not None:
        panel["progress"].progress(progress, text=f"กำลังวิเคราะห์ {progress:.0%}")


def _discard_videos() -> None:
    """ลบวิดีโอของรอบก่อนทิ้ง กันไม่ให้ไฟล์ชั่วคราวค้างสะสมทุกครั้งที่วิเคราะห์ใหม่"""
    stale = [st.session_state.get("video_path")]
    stale += list((st.session_state.get("video_variants") or {}).values())
    for path in stale:
        if path and os.path.exists(path):
            os.unlink(path)
    st.session_state.video_variants = {}


def _new_video_path() -> str:
    _discard_videos()
    return _new_variant_path()


def _new_variant_path() -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="acl_result_")
    handle.close()
    return handle.name


@st.cache_resource(show_spinner="กำลังเตรียมไฟล์โมเดลท่าทาง (ครั้งแรกครั้งเดียว)")
def prepare_model():
    """ดึงไฟล์โมเดลของ MediaPipe มาเก็บไว้ก่อนเริ่มวิเคราะห์

    เครื่องที่ติดตั้งตาม README มีไฟล์นี้อยู่แล้ว ฟังก์ชันจึงคืนค่าทันที
    ส่วนเครื่องที่รันจากโค้ดสด (ไฟล์โมเดลไม่ได้อยู่ใน repo) จะโหลดครั้งเดียวแล้วแคชไว้
    """
    return pose_module.ensure_model()


def analyse(capture, session, panel, use_wall_clock, total_frames, fps, video_path):
    """วนอ่านทีละเฟรม อัปเดต session และบันทึกวิดีโอผลลัพธ์ไปพร้อมกัน

    การกดปุ่มหยุดจะทำให้ Streamlit สั่งรันสคริปต์ใหม่ ซึ่งขัดจังหวะลูปนี้เอง
    บล็อก finally จึงเป็นที่ปิดตัวเขียนวิดีโอ ไม่งั้นไฟล์จะเสียหายเปิดไม่ได้
    ส่วนบล็อก finally ของผู้เรียกเป็นที่ปิดกล้องและลบไฟล์ชั่วคราว
    """
    detector = pose_module.create_pose(
        min_detection_confidence=MODEL_CONFIDENCE,
        min_tracking_confidence=MODEL_CONFIDENCE,
    )
    stabiliser = LevelStabiliser(LEVEL_WINDOW_SECONDS, LEVEL_HOLD_SECONDS)
    writer = None
    analysed = 0
    lowest_theta = float("inf")
    started = time.perf_counter()
    previous_time = started
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # โหมด VIDEO ของ MediaPipe บังคับให้เวลาประทับเพิ่มขึ้นทุกเฟรม
            # คลิปใช้เวลาตามอัตราเฟรม ส่วนกล้องสดใช้เวลาจริงที่ผ่านไป
            now = time.perf_counter()
            timestamp_ms = (
                (now - started) * 1000.0 if use_wall_clock else frame_index / fps * 1000.0
            )
            results = pose_module.detect(detector, frame, timestamp_ms)

            dt = None
            if use_wall_clock:
                dt = max(now - previous_time, 1e-3)
                previous_time = now

            legs = {
                side: pose_module.world_leg(results, side, MIN_VISIBILITY)
                for side in pose_module.SIDES
            }
            record = session.update(frame_index, legs["left"], legs["right"], dt)

            height, width = frame.shape[:2]
            for side in pose_module.SIDES:
                if legs[side] is None:
                    continue
                pixels = pose_module.pixel_leg(results, side, width, height)
                if pixels is not None:
                    pose_module.draw_leg(frame, pixels, LEG_COLORS[side])

            if record is not None:
                analysed += 1
                display_risk, display_level = stabiliser.update(
                    record.risk, timestamp_ms / 1000.0
                )
                draw_risk_banner(frame, display_risk, display_level)
                draw_metric_strip(frame, record)

            # เปิดตัวเขียนวิดีโอเมื่อรู้ขนาดเฟรมจริงแล้ว คือหลังอ่านเฟรมแรกได้
            if writer is None:
                writer = video.open_writer(video_path, fps, (width, height))
            writer.write(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if record is not None:
                if record.min_theta < lowest_theta:
                    lowest_theta = record.min_theta
                    st.session_state.critical_image = rgb
                progress = (frame_index + 1) / total_frames if total_frames else None
                update_live_panel(
                    panel, record, display_risk, display_level, analysed, progress
                )

            panel["frame"].image(rgb, width="stretch")
            frame_index += 1
    finally:
        detector.close()
        if writer is not None:
            writer.release()


if st.session_state.running:
    # แถบหัวแบบย่อ: ชิปผู้ทดสอบ + จำนวนเฟรมทางซ้าย ปุ่มหยุดทางขวา แทนที่หัวข้อ/คำอธิบาย/
    # แหล่งข้อมูล/ปุ่มเริ่ม ที่ถูกซ่อนไป ให้เนื้อหาหลักเหลือพื้นที่พอสำหรับวิดีโอกับตัวเลข
    header_left, header_right = st.columns([5, 1])
    with header_left:
        participant = st.session_state.participant
        chip_text = participant.label if participant is not None else ""
        st.markdown(f":material/person: **ACL** · {chip_text}")
        frames_slot = st.empty()
    if header_right.button("หยุด", type="primary"):
        st.session_state.running = False
        st.rerun()

    capture = None
    temporary_path = None
    try:
        # อ่านจาก session_state ไม่ใช่วิดเจ็ต mode/uploaded_video ตรง ๆ เพราะวิดเจ็ตเหล่านั้น
        # อยู่ในบล็อก "if not running" ซึ่งถูกซ่อนระหว่างวิเคราะห์ จึงไม่มีตัวแปรให้อ่านในรอบนี้
        run_mode = st.session_state.mode
        if run_mode == "กล้องเว็บแคม":
            capture = cv2.VideoCapture(0)
            use_wall_clock = True
            total_frames = 0
        else:
            if st.session_state.upload_bytes is None:
                st.warning("กรุณาอัปโหลดคลิปก่อนเริ่มวิเคราะห์")
                st.session_state.running = False
                st.stop()
            suffix = os.path.splitext(st.session_state.upload_name or "")[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(st.session_state.upload_bytes)
                temporary_path = handle.name
            capture = cv2.VideoCapture(temporary_path)
            use_wall_clock = False
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if not capture.isOpened():
            st.error(
                "เปิดกล้องไม่ได้ กรุณาตรวจสอบสิทธิ์การเข้าถึงกล้องใน System Settings"
                if run_mode == "กล้องเว็บแคม"
                else "อ่านไฟล์วิดีโอนี้ไม่ได้ ลองแปลงเป็น .mp4 (H.264) แล้วอัปโหลดใหม่"
            )
            st.session_state.running = False
            st.stop()

        fps = capture.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        session = Session(fps=fps, smoothing_window=SMOOTHING_WINDOW)
        # เก็บไว้ก่อนวิเคราะห์ เพราะ Session ถูกแก้ไขในตัว (append ทีละเฟรม)
        # และปุ่มหยุดจะทำให้ Streamlit สั่งรันสคริปต์ใหม่กลางลูป ข้ามโค้ดหลัง analyse()
        st.session_state.session = session
        # วิดีโอผลลัพธ์เขียนที่ความละเอียดเดิมของแหล่งข้อมูล แล้วค่อยย่อตอนแสดงผล
        # ถ้าย่อตั้งแต่ตอนเขียน จะย้อนกลับไปเอาภาพคมชัดเดิมไม่ได้อีก
        st.session_state.video_path = _new_video_path()
        try:
            prepare_model()
            analyse(
                capture, session, build_live_panel(frames_slot),
                use_wall_clock, total_frames, fps, st.session_state.video_path,
            )
        except pose_module.ModelNotFound as error:
            st.error(str(error))
            st.session_state.running = False
            st.stop()
        except video.EncoderNotAvailable as error:
            st.error(str(error))
            st.session_state.running = False
            st.stop()

        st.session_state.running = False
        st.rerun()
    finally:
        if capture is not None:
            capture.release()
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

# ---------------------------------------------------------------- ผลลัพธ์
session = st.session_state.session
if session is not None and session.records:
    critical = session.critical_frame
    st.divider()
    st.subheader("ผลการวิเคราะห์")

    if critical is not None:
        summary_column, image_column = st.columns([3, 2])
        with summary_column:
            color = LEVEL_COLORS[critical.level]
            st.markdown(
                f"""
<div class="critical-card">
  <div style="color:#94a3b8;font-size:14px;letter-spacing:.08em;">CRITICAL FRAME</div>
  <div class="critical-score" style="color:{color};">{critical.risk:.1f}</div>
  <div class="critical-level" style="color:{color};">
    {critical.level} — {LEVEL_THAI[critical.level]}
  </div>
  <div class="critical-note">
    เฟรมที่ {critical.frame_index} (วินาทีที่ {critical.time_s:.2f}) มุมข้อเข่าน้อยที่สุด
    {critical.min_theta:.1f}° — ค่าดัชนีมาจากขา{"ซ้าย" if critical.worst_side == "left" else "ขวา"}
    ซึ่งเสี่ยงกว่า
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            side = critical.worst_side
            for column, (label, value) in zip(
                st.columns(4),
                [
                    ("θ มุมข้อเข่า (°)", getattr(critical, f"{side}_theta")),
                    ("γ Knee Valgus (°)", getattr(critical, f"{side}_valgus")),
                    ("ω ความเร็วเชิงมุม (°/s)", getattr(critical, f"{side}_omega")),
                    ("α ความเร่งเชิงมุม (°/s²)", getattr(critical, f"{side}_alpha")),
                ],
            ):
                column.metric(label, f"{value:.1f}")

        if st.session_state.critical_image is not None:
            image_column.image(
                st.session_state.critical_image, caption="เฟรมวิกฤต", width=PREVIEW_WIDTH
            )

        left_column, right_column = st.columns(2)
        for column, side, label in (
            (left_column, "left", "ขาซ้าย"),
            (right_column, "right", "ขาขวา"),
        ):
            value = getattr(critical, f"{side}_risk")
            column.metric(
                f"{label} ณ เฟรมวิกฤต",
                "—" if value != value else f"{value:.1f}",
                f"θ {getattr(critical, f'{side}_theta'):.1f}°",
            )

    st.markdown("**ตารางสรุป**")
    st.dataframe(
        report.summarise(session).style.format("{:.1f}", na_rep="—"),
        width="stretch",
    )

    # ------------------------------------------------------------ วิดีโอผลลัพธ์
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.markdown("**วิดีโอผลการวิเคราะห์**")
        st.caption(
            "ทุกเฟรมมีค่า θ γ ω α ของทั้งสองขาและระดับความเสี่ยงฝังอยู่ในภาพ "
            "หยุดดูเฟรมไหนก็อ่านค่าของเฟรมนั้นได้ทันที"
        )
        # ปล่อยให้ key จัดการสถานะเอง ไม่ตั้งค่าเริ่มต้นซ้ำใน session_state
        # มิฉะนั้น Streamlit จะเตือนว่าวิดเจ็ตถูกกำหนดค่าจากสองทาง
        size_label = st.segmented_control(
            "ขนาดวิดีโอ",
            list(video.SIZE_CHOICES),
            default=video.DEFAULT_SIZE,
            key="video_size",
        ) or video.DEFAULT_SIZE

        target_width = video.SIZE_CHOICES[size_label]
        playable = st.session_state.video_path
        if target_width is not None:
            # ย่อครั้งเดียวต่อหนึ่งขนาด แล้วเก็บไว้ใช้ซ้ำ ไม่ต้องเข้ารหัสใหม่ทุกครั้งที่หน้าเว็บรีเฟรช
            cached = st.session_state.video_variants.get(size_label)
            if cached is None or not os.path.exists(cached):
                with st.spinner(f"กำลังย่อวิดีโอเป็น {size_label}..."):
                    try:
                        cached = _new_variant_path()
                        video.rescale(st.session_state.video_path, cached, target_width)
                        st.session_state.video_variants[size_label] = cached
                    except (video.EncoderNotAvailable, subprocess.SubprocessError) as error:
                        st.warning(f"ย่อขนาดไม่สำเร็จ ใช้ต้นฉบับแทน ({error})")
                        cached = None
            playable = cached or st.session_state.video_path

        st.video(playable, width=RESULT_VIDEO_WIDTH)
        with open(playable, "rb") as handle:
            st.download_button(
                "ดาวน์โหลดวิดีโอผลการวิเคราะห์",
                data=handle.read(),
                file_name="acl_analysed.mp4",
                mime="video/mp4",
            )

    frame_table = session.to_dataframe().set_index("time_s")
    st.markdown("**ดัชนีความเสี่ยงตลอดช่วงที่วิเคราะห์**")
    st.line_chart(frame_table[["left_risk", "right_risk"]])
    st.markdown("**มุมข้อเข่าตลอดช่วงที่วิเคราะห์**")
    st.line_chart(frame_table[["left_theta", "right_theta"]])

    # ตารางรายเฟรมครบทุกแถว คู่กับวิดีโอด้านบน ใช้ค้นหาเฟรมที่สนใจแล้วเทียบตัวเลขได้ละเอียด
    st.markdown("**ค่าทุกตัวแปรรายเฟรม**")
    all_frames = session.to_dataframe()
    # จัดรูปเฉพาะคอลัมน์ทศนิยม คอลัมน์ข้อความอย่าง level และ worst_side จัดรูปแบบตัวเลขไม่ได้
    # ส่วน frame_index เป็นจำนวนเต็ม ปล่อยไว้ตามเดิมอ่านง่ายกว่าเติม .0
    decimals = all_frames.select_dtypes("float").columns
    st.dataframe(
        all_frames.style.format({name: "{:.1f}" for name in decimals}, na_rep="—"),
        width="stretch",
        height=320,
    )

    csv_column, pdf_column = st.columns(2)
    csv_column.download_button(
        "ดาวน์โหลดผลรายเฟรม (CSV)",
        # แปะคอลัมน์ผู้ทดสอบไว้ทุกแถว เพื่อให้นำ CSV ของหลายคนมาต่อกันเป็นตารางเดียวได้
        # เป็นแค่คอลัมน์ข้อมูลอ้างอิง ไม่กระทบค่าที่คำนวณในคอลัมน์อื่น
        data=report.csv_bytes(session, st.session_state.participant),
        file_name="acl_session.csv",
        mime="text/csv",
        width="stretch",
    )

    # สร้าง PDF เมื่อผู้ใช้ขอเท่านั้น เพราะต้องเรียก Chrome ซึ่งใช้เวลาสองสามวินาที
    # แล้วเก็บผลไว้ใน session_state กันไม่ให้สร้างซ้ำทุกครั้งที่หน้าเว็บรีเฟรช
    if st.session_state.report_pdf is None:
        if pdf_column.button("สร้างรายงาน PDF", width="stretch"):
            with st.spinner("กำลังสร้างรายงาน..."):
                try:
                    st.session_state.report_pdf = report.build_pdf(
                        session,
                        st.session_state.source_label or "ไม่ระบุ",
                        st.session_state.critical_image,
                        participant=st.session_state.participant,
                    )
                except report.ChromeNotFound as error:
                    st.error(str(error))
                except subprocess.SubprocessError as error:
                    st.error(f"สร้างรายงานไม่สำเร็จ: {error}")
            st.rerun()
    else:
        pdf_column.download_button(
            "ดาวน์โหลดรายงาน (PDF)",
            data=st.session_state.report_pdf,
            file_name="acl_report.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
elif session is not None and not st.session_state.running:
    st.warning(
        "ไม่พบข้อต่อที่ชัดพอในคลิปนี้ "
        "ถ่ายให้เห็นสะโพก เข่า และข้อเท้าครบทั้งตัว ในที่ที่แสงเพียงพอ"
    )
elif not st.session_state.running:
    st.info(
        "เลือกแหล่งข้อมูลแล้วกดเริ่มวิเคราะห์ "
        "ถ่ายให้เห็นสะโพก เข่า และข้อเท้าครบทั้งตัว ระยะประมาณ 2-3 เมตร"
    )

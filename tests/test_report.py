"""ตรวจตารางสรุปและการประกอบรายงาน"""

import io
import math

import pandas as pd
import pytest

from acl import report
from acl.session import Session


def _bend(theta_degrees, side_offset=0.0):
    """พิกัดขาที่งอเข่าในระนาบข้าง โดยเลื่อนเข่าออกด้านข้างได้เพื่อสร้าง valgus"""
    angle = math.radians(180.0 - theta_degrees)
    return (
        (0.0, -0.4, 0.0),
        (side_offset, 0.0, 0.0),
        (0.0, 0.4 * math.cos(angle), 0.4 * math.sin(angle)),
    )


def _session(angles=(170.0, 120.0, 90.0, 140.0)):
    session = Session(fps=50, smoothing_window=1)
    for index, theta in enumerate(angles):
        session.update(index, _bend(theta), _bend(theta))
    return session


def test_summary_has_three_rows():
    summary = report.summarise(_session())
    assert list(summary.index) == ["ขาซ้าย", "ขาขวา", "รวม"]


def test_summary_reports_both_angle_extremes():
    """รายงานทั้งมุมน้อยสุดและสูงสุด เพราะมุมน้อย = งอลึก = เสี่ยงสูง"""
    summary = report.summarise(_session((170.0, 120.0, 90.0, 140.0)))
    assert summary.loc["ขาซ้าย", "มุมข้อเข่าน้อยสุด (°)"] == pytest.approx(90.0, abs=0.5)
    assert summary.loc["ขาซ้าย", "มุมข้อเข่าสูงสุด (°)"] == pytest.approx(170.0, abs=0.5)


def test_summary_average_risk_matches_frames():
    session = _session()
    frame = session.to_dataframe()
    summary = report.summarise(session)
    assert summary.loc["รวม", "ความเสี่ยงเฉลี่ย"] == pytest.approx(frame["risk"].mean())
    assert summary.loc["รวม", "ความเสี่ยงสูงสุด"] == pytest.approx(frame["risk"].max())


def test_summary_uses_magnitude_for_acceleration():
    """α ที่ติดลบคือการชะลอ ต้องนับขนาด ไม่งั้นค่าสูงสุดจะกลายเป็นค่าที่เล็กที่สุด"""
    session = _session()
    frame = session.to_dataframe()
    assert summary_alpha(session) == pytest.approx(frame["left_alpha"].abs().max())


def summary_alpha(session):
    return report.summarise(session).loc["ขาซ้าย", "ความเร่งเชิงมุมสูงสุด (°/s²)"]


def test_missing_leg_becomes_nan_not_zero():
    """ขาที่ไม่เคยตรวจพบต้องเป็น NaN เพื่อให้รายงานแสดงขีดกลาง ไม่ใช่เลข 0 ที่ชวนเข้าใจผิด"""
    session = Session(fps=50, smoothing_window=1)
    for index, theta in enumerate((170.0, 120.0, 95.0)):
        session.update(index, _bend(theta), None)
    summary = report.summarise(session)
    assert math.isnan(summary.loc["ขาขวา", "ความเสี่ยงเฉลี่ย"])
    assert not math.isnan(summary.loc["ขาซ้าย", "ความเสี่ยงเฉลี่ย"])


def test_html_contains_summary_and_charts():
    page = report.build_html(_session(), "C0009.MP4")
    assert "ตารางสรุป" in page
    assert "เฟรมวิกฤต" in page
    assert "C0009.MP4" in page
    assert page.count("data:image/png;base64,") == 2  # กราฟสองรูป ยังไม่มีภาพเฟรมวิกฤต
    assert 'lang="th"' in page


def test_html_escapes_the_source_name():
    """ชื่อไฟล์มาจากผู้ใช้ ต้องไม่หลุดเป็น HTML"""
    page = report.build_html(_session(), "<script>alert(1)</script>.mp4")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_html_survives_a_session_with_no_critical_frame():
    session = Session(fps=50, smoothing_window=1)
    for index in range(3):  # คลิปที่ไม่มีคนอยู่ในภาพเลย
        session.update(index, None, None)
    assert session.records == []
    page = report.build_html(session, "empty.mp4")
    assert "ไม่พบเฟรมที่คำนวณมุมข้อเข่าได้" in page


@pytest.mark.skipif(report.find_chrome() is None, reason="ต้องมี Chrome จึงจะสร้าง PDF ได้")
def test_pdf_is_produced():
    data = report.build_pdf(_session(), "C0009.MP4")
    assert data.startswith(b"%PDF-")
    assert len(data) > 10_000


def test_participant_bmi_known_case():
    participant = report.Participant(age=17, gender="ชาย", weight_kg=60.0, height_cm=170.0)
    assert participant.bmi == pytest.approx(20.8, abs=0.05)


def test_participant_bmi_guards_against_zero_height():
    """ส่วนสูงเป็น 0 ต้องคืน NaN ไม่ใช่โยน ZeroDivisionError"""
    participant = report.Participant(age=17, gender="ชาย", weight_kg=60.0, height_cm=0.0)
    assert math.isnan(participant.bmi)


def test_build_html_includes_participant_label_when_given():
    participant = report.Participant(age=17, gender="ชาย", weight_kg=60.0, height_cm=170.0)
    page = report.build_html(_session(), "C0009.MP4", participant=participant)
    assert participant.label in page


def test_build_html_omits_participant_line_when_none():
    page = report.build_html(_session(), "C0009.MP4", participant=None)
    assert "ผู้ทดสอบ:" not in page


def test_build_html_escapes_participant_label():
    """เพศหรือข้อความในผู้ทดสอบมาจากผู้ใช้ ต้องไม่หลุดเป็น HTML เหมือน source_label"""
    participant = report.Participant(
        age=17, gender="<script>alert(1)</script>", weight_kg=60.0, height_cm=170.0
    )
    page = report.build_html(_session(), "C0009.MP4", participant=participant)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_csv_bytes_prepends_participant_columns():
    session = _session()
    participant = report.Participant(age=17, gender="ชาย", weight_kg=60.0, height_cm=170.0)
    data = report.csv_bytes(session, participant)
    frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    assert list(frame.columns[:5]) == [
        "participant_age", "participant_gender",
        "participant_weight_kg", "participant_height_cm", "participant_bmi",
    ]
    assert (frame["participant_age"] == 17).all()
    assert (frame["participant_gender"] == "ชาย").all()
    assert (frame["participant_weight_kg"] == 60.0).all()
    assert (frame["participant_height_cm"] == 170.0).all()
    assert frame["participant_bmi"].tolist() == pytest.approx([participant.bmi] * len(frame))


def test_csv_bytes_without_participant_matches_session_export():
    session = _session()
    assert report.csv_bytes(session, None) == session.to_csv_bytes()

"""ตรวจการคำนวณมุมและอนุพันธ์ด้วยรูปทรงที่รู้คำตอบล่วงหน้า"""

import math

import pytest

from acl.biomech import AngleSmoother, derivative, knee_angle, knee_valgus
from acl.session import LevelStabiliser, Session

# ขาเหยียดตรงในแนวดิ่ง: สะโพกอยู่เหนือเข่า เข่าอยู่เหนือข้อเท้า (แกน y ชี้ลงตาม MediaPipe)
STRAIGHT = ((0.0, -0.4, 0.0), (0.0, 0.0, 0.0), (0.0, 0.4, 0.0))


def test_straight_leg_is_180_degrees():
    assert knee_angle(*STRAIGHT) == pytest.approx(180.0)


def test_straight_leg_has_no_valgus():
    assert knee_valgus(*STRAIGHT) == pytest.approx(0.0, abs=1e-9)


def test_right_angle_knee():
    """สะโพกอยู่เหนือเข่า ข้อเท้าอยู่ด้านหน้าเข่า ได้มุม 90 องศา"""
    assert knee_angle((0.0, -0.4, 0.0), (0.0, 0.0, 0.0), (0.4, 0.0, 0.0)) == pytest.approx(90.0)


def test_valgus_grows_with_deviation():
    """เข่าเบี่ยงเข้าด้านในมากขึ้น ค่า γ ต้องเพิ่มขึ้น"""
    slight = knee_valgus((0.0, -0.4, 0.0), (0.02, 0.0, 0.0), (0.0, 0.4, 0.0))
    heavy = knee_valgus((0.0, -0.4, 0.0), (0.08, 0.0, 0.0), (0.0, 0.4, 0.0))
    assert 0.0 < slight < heavy


def test_valgus_ignores_depth_axis():
    """γ วัดบนระนาบหน้า การเลื่อนตามแกน z จึงต้องไม่เปลี่ยนค่า"""
    flat = knee_valgus((0.0, -0.4, 0.0), (0.05, 0.0, 0.0), (0.0, 0.4, 0.0))
    deep = knee_valgus((0.0, -0.4, 0.3), (0.05, 0.0, -0.2), (0.0, 0.4, 0.5))
    assert flat == pytest.approx(deep)


def test_degenerate_points_give_nan():
    assert math.isnan(knee_angle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.4, 0.0)))


def test_derivative_is_backward_difference():
    assert derivative(150.0, 160.0, 1 / 30) == pytest.approx(-300.0)


def test_derivative_guards_against_zero_dt():
    assert derivative(150.0, 160.0, 0.0) == 0.0


def test_smoother_window_one_is_passthrough():
    smoother = AngleSmoother(window=1)
    assert [smoother.update(v) for v in (10.0, 90.0, 50.0)] == [10.0, 90.0, 50.0]


def test_smoother_averages_over_window():
    smoother = AngleSmoother(window=3)
    assert smoother.update(30.0) == pytest.approx(30.0)
    assert smoother.update(60.0) == pytest.approx(45.0)
    assert smoother.update(90.0) == pytest.approx(60.0)
    assert smoother.update(120.0) == pytest.approx(90.0)  # ค่าแรกหลุดหน้าต่างไปแล้ว


def test_smoother_rejects_zero_window():
    with pytest.raises(ValueError):
        AngleSmoother(window=0)


def _bend(theta_degrees):
    """พิกัดขาที่งอเข่าตามมุมที่กำหนด โดยหมุนหน้าแข้งในระนาบข้าง (sagittal)

    การงอเข่าจริงเกิดในระนาบ y-z ไม่ใช่ระนาบหน้า จึงหมุนตามแกน z
    เพื่อให้ค่า γ ที่วัดบนระนาบหน้ายังคงเป็นศูนย์
    """
    angle = math.radians(180.0 - theta_degrees)
    return (
        (0.0, -0.4, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.4 * math.cos(angle), 0.4 * math.sin(angle)),
    )


def test_pure_sagittal_flexion_produces_no_valgus():
    """งอเข่าตรง ๆ ในระนาบข้าง ต้องไม่ทำให้ γ เพิ่ม มิฉะนั้นจะนับการงอเข่าซ้ำสองครั้ง"""
    for theta in (170.0, 140.0, 110.0, 80.0, 50.0):
        assert knee_valgus(*_bend(theta)) == pytest.approx(0.0, abs=1e-9)


def test_session_picks_frame_with_smallest_knee_angle():
    """Critical Frame คือเฟรมที่มุมข้อเข่าน้อยที่สุดตลอดการทดสอบ"""
    session = Session(fps=30, smoothing_window=1)
    for index, theta in enumerate([175.0, 140.0, 95.0, 130.0]):
        session.update(index, _bend(theta), _bend(theta))

    critical = session.critical_frame
    assert critical is not None
    assert critical.frame_index == 2
    assert critical.min_theta == pytest.approx(95.0, abs=0.5)


def test_session_reports_the_more_at_risk_leg():
    session = Session(fps=30, smoothing_window=1)
    record = session.update(0, _bend(170.0), _bend(100.0))
    assert record.worst_side == "right"
    assert record.risk == pytest.approx(record.right_risk)


def test_session_skips_frames_with_no_visible_leg():
    session = Session(fps=30, smoothing_window=1)
    assert session.update(0, None, None) is None
    assert session.records == []
    assert session.critical_frame is None


def test_session_uses_video_timebase_not_wall_clock():
    """ω ต้องมาจาก Δt = 1/FPS ของคลิป ไม่ใช่เวลาจริงของเครื่อง"""
    session = Session(fps=30, smoothing_window=1)
    session.update(0, _bend(160.0), None)
    record = session.update(1, _bend(150.0), None)
    assert record.left_omega == pytest.approx(-10.0 * 30.0, abs=0.5)
    assert record.time_s == pytest.approx(1 / 30)


def test_session_accumulates_variable_dt():
    """โหมดกล้องสดส่ง dt จริงที่วัดได้ ซึ่งไม่คงที่ นาฬิกาจึงต้องสะสมจาก dt เหล่านั้น"""
    session = Session(fps=30, smoothing_window=1)
    times = [session.update(0, _bend(160.0), None, dt).time_s for dt in (0.1, 0.2, 0.1)]
    assert times == pytest.approx([0.0, 0.1, 0.3])


def test_session_advances_clock_when_both_legs_undetected():
    """เฟรมที่มองไม่เห็นขาทั้งสองข้างต้องเดินนาฬิกาต่อ มิฉะนั้นเฟรมถัดไปเวลาจะเพี้ยน"""
    session = Session(fps=30, smoothing_window=1)
    session.update(0, None, None, 0.1)
    record = session.update(1, _bend(160.0), None, 0.2)
    assert record.time_s == pytest.approx(0.1)


def _run_theta_series(angles, window):
    """ป้อนมุมชุดหนึ่งเข้า Session แล้วคืนค่า α ของขาซ้ายทุกเฟรม"""
    session = Session(fps=50, smoothing_window=window)
    for index, theta in enumerate(angles):
        session.update(index, _bend(theta), None)
    return [record.left_alpha for record in session.records]


def test_still_leg_has_no_angular_acceleration():
    assert max(abs(a) for a in _run_theta_series([160.0] * 20, window=5)) == pytest.approx(0.0, abs=1e-6)


def test_smoothing_suppresses_jitter_in_acceleration():
    """สัญญาณรบกวนสลับขึ้นลงต้องถูกกรองก่อนเข้าสู่ α มิฉะนั้นดัชนีจะแกว่ง

    วัดจากคลิปจริงขณะยืนนิ่ง การกรองลด |α| ที่ควรเป็นศูนย์ลงราวครึ่งหนึ่ง
    """
    jitter = [160.0 + (0.3 if index % 2 else -0.3) for index in range(40)]
    unfiltered = max(abs(a) for a in _run_theta_series(jitter, window=1))
    filtered = max(abs(a) for a in _run_theta_series(jitter, window=5))
    assert filtered < unfiltered / 2


# ---------------------------------------------------------------- LevelStabiliser


def _steady(stabiliser, risk, count, start=0.0, step=0.05):
    """ป้อนค่า R เดิมซ้ำ ๆ แล้วคืนผลของครั้งสุดท้าย"""
    result = None
    for index in range(count):
        result = stabiliser.update(risk, start + index * step)
    return result


def test_stabiliser_returns_median_of_window():
    stabiliser = LevelStabiliser(window_seconds=1.0, hold_seconds=0.0)
    for index, risk in enumerate([10.0, 30.0, 20.0]):
        median, _ = stabiliser.update(risk, index * 0.1)
    assert median == pytest.approx(20.0)


def test_stabiliser_drops_samples_older_than_window():
    """ค่าเก่าที่หลุดหน้าต่างต้องไม่ถ่วงป้ายไว้"""
    stabiliser = LevelStabiliser(window_seconds=0.5, hold_seconds=0.0)
    _steady(stabiliser, 90.0, count=10)
    median, level = _steady(stabiliser, 5.0, count=10, start=1.0)
    assert median == pytest.approx(5.0)
    assert level == "Low"


def test_single_frame_spike_does_not_flip_the_badge():
    """ค่าโดดเดี่ยวเฟรมเดียวคือสัญญาณรบกวน ไม่ใช่ความเสี่ยงที่เกิดขึ้นจริง"""
    stabiliser = LevelStabiliser(window_seconds=0.5)
    _steady(stabiliser, 10.0, count=10)
    _, level = stabiliser.update(95.0, 0.5)
    assert level == "Low"


def test_sustained_high_risk_does_flip_the_badge():
    """ถ้าความเสี่ยงสูงจริงต่อเนื่อง ป้ายต้องขึ้น High ภายในหนึ่งหน้าต่าง"""
    stabiliser = LevelStabiliser(window_seconds=0.5)
    _steady(stabiliser, 10.0, count=10)
    _, level = _steady(stabiliser, 80.0, count=10, start=0.5)
    assert level == "High"


def test_first_sample_is_reported_immediately():
    assert LevelStabiliser().update(62.0, 0.0) == (62.0, "High")


def test_stabiliser_rejects_zero_window():
    with pytest.raises(ValueError):
        LevelStabiliser(window_seconds=0.0)


def test_brief_peak_stays_visible_long_enough_to_read():
    """จังหวะ R สูงจริงกินเวลาเพียงราว 0.1 วินาที ต้องค้างบนป้ายให้อ่านทัน

    วัดจากคลิปจริง ช่วงที่ R >= 50 ติดต่อกันยาว 0.04-0.14 วินาที
    ถ้าไม่ค้างค่า ป้ายจะวูบขึ้นแล้วหายไปก่อนที่คนจะมองเห็น
    """
    stabiliser = LevelStabiliser(window_seconds=0.2, hold_seconds=2.0)
    _steady(stabiliser, 10.0, count=20, step=0.02)          # ยืนนิ่ง 0.4 วินาที
    _steady(stabiliser, 80.0, count=7, start=0.4, step=0.02)  # ลงพื้น 0.14 วินาที
    _, level = _steady(stabiliser, 10.0, count=25, start=0.54, step=0.02)
    assert level == "High"  # ผ่านไปครึ่งวินาทีแล้วยังค้างอยู่


def test_hold_expires_after_its_window():
    stabiliser = LevelStabiliser(window_seconds=0.2, hold_seconds=1.0)
    _steady(stabiliser, 80.0, count=10, step=0.02)
    _, level = _steady(stabiliser, 5.0, count=20, start=3.0, step=0.02)
    assert level == "Low"


def test_noise_spike_is_not_latched_by_the_hold():
    """ค้างค่าจากสัญญาณที่กรองแล้ว สัญญาณรบกวนเฟรมเดียวจึงไม่ค้างป้ายไว้สองวินาที"""
    stabiliser = LevelStabiliser(window_seconds=0.2, hold_seconds=2.0)
    _steady(stabiliser, 10.0, count=20, step=0.02)
    stabiliser.update(99.0, 0.40)
    _, level = _steady(stabiliser, 10.0, count=10, start=0.42, step=0.02)
    assert level == "Low"


def test_stabiliser_rejects_negative_hold():
    with pytest.raises(ValueError):
        LevelStabiliser(hold_seconds=-1.0)

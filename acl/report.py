"""สร้างรายงานสรุปผลการวิเคราะห์เป็นไฟล์ PDF

แปลง HTML เป็น PDF ด้วย Google Chrome แบบ headless เพราะการวางสระบน-ล่าง
และวรรณยุกต์ของภาษาไทยต้องใช้เอนจินจัดรูปอักษร (text shaping) ที่ครบถ้วน
ไลบรารีสร้าง PDF ทั่วไปวางรูปสระผิดตำแหน่ง ส่วน Chrome ใช้เอนจินเดียวกับที่
เรนเดอร์หน้าเว็บ จึงได้ผลตรงกับที่เห็นบนจอ

ข้อความบนกราฟใช้ภาษาอังกฤษ เพราะ matplotlib ไม่มีเอนจินจัดรูปอักษรไทย
"""

from __future__ import annotations

import base64
import html
import io
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # ไม่มีหน้าต่างกราฟิก ต้องเรนเดอร์ลงไฟล์อย่างเดียว

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .risk import BAND_HIGH, BAND_LOW  # noqa: E402

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "chromium",
)

# บนเครื่องแม่ข่าย Linux (เช่น Streamlit Cloud) Chromium เปิด sandbox ไม่ได้เพราะ
# คอนเทนเนอร์ปิด user namespace ไว้ และ /dev/shm เล็กเกินกว่าจะเรนเดอร์หน้าใหญ่ ๆ ได้
# บน macOS ไม่ต้องใส่ธงพวกนี้ พฤติกรรมเดิมจึงไม่เปลี่ยน
SERVER_FLAGS = ("--no-sandbox", "--disable-dev-shm-usage") if sys.platform.startswith("linux") else ()

SIDE_LABELS = {"left": "ขาซ้าย", "right": "ขาขวา"}


@dataclass(frozen=True)
class Participant:
    """ข้อมูลผู้ทดสอบ — ใช้เป็นข้อมูลอ้างอิงในรายงานและ CSV เท่านั้น

    ห้ามให้ค่าพวกนี้ไหลเข้าสูตร risk_index เด็ดขาด แบบจำลองของรายงานมีตัวแปรอิสระ
    เพียงสี่ตัวคือ θ, γ, ω, α การใส่อายุหรือเพศเข้าไปในการคำนวณเป็นบั๊กเดิมที่แก้ไปแล้ว
    """

    age: int
    gender: str  # "ชาย" / "หญิง"
    weight_kg: float
    height_cm: float

    @property
    def bmi(self) -> float:
        """ดัชนีมวลกาย คืน NaN แทนการโยน error เมื่อส่วนสูงไม่สมเหตุสมผล"""
        if self.height_cm <= 0:
            return float("nan")
        height_m = self.height_cm / 100.0
        return self.weight_kg / (height_m * height_m)

    @property
    def label(self) -> str:
        """ข้อความสรุปหนึ่งบรรทัด ใช้แสดงเป็นชิปบนหน้าเว็บและในหัวรายงาน"""
        bmi = self.bmi
        bmi_text = "—" if math.isnan(bmi) else f"{bmi:.1f}"
        return (
            f"{self.gender} · {self.age} ปี · {self.weight_kg:.1f} กก. · "
            f"{self.height_cm:.1f} ซม. · BMI {bmi_text}"
        )


class ChromeNotFound(RuntimeError):
    """ไม่พบเบราว์เซอร์ที่ใช้แปลง HTML เป็น PDF ได้"""


def _stats(risk, theta, valgus, omega, alpha) -> dict:
    """ค่าสรุปของขาหนึ่งข้าง ค่า NaN จากเฟรมที่มองไม่เห็นขาถูกข้ามโดยอัตโนมัติ"""
    return {
        "ความเสี่ยงเฉลี่ย": risk.mean(),
        "ความเสี่ยงสูงสุด": risk.max(),
        "มุมข้อเข่าน้อยสุด (°)": theta.min(),
        "มุมข้อเข่าสูงสุด (°)": theta.max(),
        "Knee Valgus สูงสุด (°)": valgus.max(),
        "ความเร็วเชิงมุมสูงสุด (°/s)": omega.abs().max(),
        "ความเร่งเชิงมุมสูงสุด (°/s²)": alpha.abs().max(),
    }


def summarise(session) -> pd.DataFrame:
    """ตารางสรุปสามแถว: ขาซ้าย ขาขวา และรวม

    แถว "รวม" ใช้คอลัมน์ risk ซึ่งเป็นค่าของขาที่เสี่ยงกว่าในแต่ละเฟรม
    จึงเป็นตัวเลขชุดเดียวกับที่ Critical Frame รายงาน
    """
    frame = session.to_dataframe()
    both = lambda name: pd.concat([frame[f"left_{name}"], frame[f"right_{name}"]])
    rows = {
        SIDE_LABELS["left"]: _stats(*(frame[f"left_{n}"] for n in
                                      ("risk", "theta", "valgus", "omega", "alpha"))),
        SIDE_LABELS["right"]: _stats(*(frame[f"right_{n}"] for n in
                                       ("risk", "theta", "valgus", "omega", "alpha"))),
        "รวม": _stats(frame["risk"], both("theta"), both("valgus"),
                      both("omega"), both("alpha")),
    }
    return pd.DataFrame(rows).T


def _chart_png(frame, columns, labels, ylabel, bands=False) -> str:
    """เรนเดอร์กราฟหนึ่งรูปเป็น PNG ที่ฝังใน HTML ได้เลย"""
    figure, axes = plt.subplots(figsize=(9.0, 3.0), dpi=150)
    for column, label in zip(columns, labels):
        axes.plot(frame["time_s"], frame[column], linewidth=1.1, label=label)
    if bands:
        # เส้นเกณฑ์ เพื่อให้อ่านกราฟในไฟล์ PDF ได้โดยไม่ต้องเปิดหน้าเว็บดูคำอธิบาย
        for level, color in ((BAND_LOW, "#f59e0b"), (BAND_HIGH, "#ef4444")):
            axes.axhline(level, color=color, linewidth=0.9, linestyle="--", alpha=0.8)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.25)
    axes.legend(loc="upper right", fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


FRAME_MAX_WIDTH = 900


def _image_png(image_rgb) -> str:
    """เข้ารหัสภาพเฟรมวิกฤตเป็น PNG ฝังในหน้า

    ย่อภาพก่อน เพราะภาพ 1080p ที่เข้ารหัส base64 ทำให้ไฟล์ PDF ใหญ่เกินจำเป็น
    ขณะที่ความกว้างระดับนี้ก็เกินพอสำหรับพิมพ์ลงกระดาษ A4
    """
    image = image_rgb
    if image.shape[1] > FRAME_MAX_WIDTH:
        scale = FRAME_MAX_WIDTH / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _format(value) -> str:
    return "—" if value != value else f"{value:.1f}"


def _summary_table_html(summary: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in summary.columns)
    body = ""
    for name, row in summary.iterrows():
        cells = "".join(f"<td>{_format(v)}</td>" for v in row)
        body += f"<tr><th class='row'>{html.escape(name)}</th>{cells}</tr>"
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(session, source_label: str, critical_image=None,
               generated_at: datetime | None = None,
               participant: Participant | None = None) -> str:
    """ประกอบรายงานทั้งฉบับเป็น HTML หน้าเดียวที่ฝังภาพและกราฟไว้ในตัว"""
    frame = session.to_dataframe()
    critical = session.critical_frame
    stamp = (generated_at or datetime.now()).strftime("%d/%m/%Y %H:%M น.")
    duration = frame["time_s"].iloc[-1] if len(frame) else 0.0

    risk_chart = _chart_png(frame, ["left_risk", "right_risk"],
                            ["Left", "Right"], "ACL Risk Index", bands=True)
    angle_chart = _chart_png(frame, ["left_theta", "right_theta"],
                             ["Left", "Right"], "Knee angle (deg)")

    # บรรทัดผู้ทดสอบเป็นข้อมูลอ้างอิงเพิ่มเติมเท่านั้น ไม่มีผลต่อค่าที่คำนวณด้านล่าง
    participant_line = (
        f"<div class=\"muted\">ผู้ทดสอบ: {html.escape(participant.label)}</div>"
        if participant is not None else ""
    )

    critical_block = "<p class='muted'>ไม่พบเฟรมที่คำนวณมุมข้อเข่าได้</p>"
    if critical is not None:
        side = critical.worst_side
        items = "".join(
            f"<div class='cell'><div class='key'>{html.escape(k)}</div>"
            f"<div class='val'>{_format(v)}</div></div>"
            for k, v in (
                ("θ มุมข้อเข่า (°)", getattr(critical, f"{side}_theta")),
                ("γ Knee Valgus (°)", getattr(critical, f"{side}_valgus")),
                ("ω ความเร็วเชิงมุม (°/s)", getattr(critical, f"{side}_omega")),
                ("α ความเร่งเชิงมุม (°/s²)", getattr(critical, f"{side}_alpha")),
            )
        )
        picture = ""
        if critical_image is not None:
            encoded = _image_png(critical_image)
            if encoded:
                picture = f"<img class='frame' src='data:image/png;base64,{encoded}'>"
        critical_block = f"""
<div class="critical">
  <div class="score">{critical.risk:.1f}</div>
  <div>
    <div class="level">{critical.level} — ความเสี่ยง{
        {"Low": "ต่ำ", "Moderate": "ปานกลาง", "High": "สูง"}[critical.level]}</div>
    <div class="muted">เฟรมที่ {critical.frame_index} · วินาทีที่ {critical.time_s:.2f}
      · มุมข้อเข่าน้อยที่สุด {critical.min_theta:.1f}°
      · ค่าดัชนีมาจาก{SIDE_LABELS[side]}ซึ่งเสี่ยงกว่า</div>
  </div>
</div>
<div class="grid">{items}</div>
{picture}"""

    return f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><title>ACL Report</title>
<style>
@page {{ size: A4; margin: 14mm; }}
body {{ font-family: "Thonburi", "Sarabun", "Leelawadee UI", "Tahoma",
       "Noto Sans Thai", "Helvetica Neue", sans-serif;
       color: #0f172a; font-size: 12px; line-height: 1.55; }}
h1 {{ font-size: 21px; margin: 0 0 2px; }}
h2 {{ font-size: 14px; margin: 18px 0 7px; border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px; }}
.muted {{ color: #64748b; font-size: 11px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 5px 7px; text-align: right; }}
thead th {{ background: #f1f5f9; text-align: center; font-weight: 600; }}
th.row {{ text-align: left; background: #f8fafc; white-space: nowrap; }}
.critical {{ display: flex; align-items: center; gap: 18px;
             border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px 16px; }}
.score {{ font-size: 40px; font-weight: 700; }}
.level {{ font-size: 15px; font-weight: 600; }}
.grid {{ display: flex; gap: 8px; margin-top: 9px; }}
.cell {{ flex: 1; border: 1px solid #e2e8f0; border-radius: 8px; padding: 7px 9px; }}
.key {{ color: #64748b; font-size: 10px; }}
.val {{ font-size: 16px; font-weight: 600; }}
img.frame {{ width: 62%; margin-top: 10px; border-radius: 8px; }}
img.chart {{ width: 100%; margin-top: 4px; }}
footer {{ margin-top: 20px; color: #64748b; font-size: 10px;
          border-top: 1px solid #cbd5e1; padding-top: 7px; }}
/* แถบปุ่มสั่งพิมพ์ สำหรับคนที่เปิดไฟล์ HTML นี้เอง เพราะเป็นทางเดียวที่ได้ PDF
   เมื่อเครื่องที่รันแอปไม่มี Chrome ให้เรียก ซ่อนตอนพิมพ์ จึงไม่ติดไปในไฟล์ PDF */
.printbar {{ position: fixed; top: 14px; right: 14px; }}
.printbar button {{ font: inherit; padding: 9px 15px; border-radius: 9px;
       border: 0; background: #0f172a; color: #f8fafc; cursor: pointer; }}
@media print {{ .printbar {{ display: none; }} }}
/* จอแคบ ๆ ปุ่มลอยจะทับหัวเรื่องที่ตัดบรรทัด จึงให้ไหลไปตามเนื้อหาแทน */
@media (max-width: 700px) {{ .printbar {{ position: static; margin-bottom: 10px; }} }}
</style></head><body>

<div class="printbar"><button onclick="window.print()">พิมพ์ / บันทึกเป็น PDF</button></div>

<h1>รายงานการวิเคราะห์ความเสี่ยงการบาดเจ็บเอ็นไขว้หน้าหัวเข่า (ACL)</h1>
<div class="muted">แหล่งข้อมูล: {html.escape(source_label)} ·
  ความยาว {duration:.1f} วินาที · วิเคราะห์ได้ {len(frame)} เฟรม ·
  อัตราเฟรม {session.fps:.0f} · ออกรายงาน {stamp}</div>
{participant_line}

<h2>ตารางสรุป</h2>
{_summary_table_html(summarise(session))}

<h2>เฟรมวิกฤต (Critical Frame)</h2>
{critical_block}

<h2>ดัชนีความเสี่ยงตลอดช่วงที่วิเคราะห์</h2>
<img class="chart" src="data:image/png;base64,{risk_chart}">

<h2>มุมข้อเข่าตลอดช่วงที่วิเคราะห์</h2>
<img class="chart" src="data:image/png;base64,{angle_chart}">

<footer>
R<sub>ACL</sub> = 100 · [0.25·f(θ) + 0.35·f(γ) + 0.20·f(ω) + 0.20·f(α)] ·
เกณฑ์ Low &lt; {BAND_LOW:.0f} ≤ Moderate &lt; {BAND_HIGH:.0f} ≤ High ·
เส้นประบนกราฟคือเกณฑ์แบ่งระดับ
</footer>
</body></html>"""


def find_chrome() -> str | None:
    """หาเบราว์เซอร์ที่ใช้แปลงเป็น PDF ได้ คืน None เมื่อไม่พบ"""
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def render_pdf(page_html: str, timeout: float = 60.0) -> bytes:
    """แปลง HTML เป็น PDF ด้วย Chrome แบบ headless"""
    browser = find_chrome()
    if browser is None:
        raise ChromeNotFound(
            "ไม่พบ Google Chrome ในเครื่อง จึงสร้างไฟล์ PDF ไม่ได้\n"
            "ติดตั้ง Chrome แล้วลองใหม่ หรือดาวน์โหลดผลรายเฟรมเป็น CSV แทน"
        )

    with tempfile.TemporaryDirectory() as workspace:
        source = Path(workspace) / "report.html"
        target = Path(workspace) / "report.pdf"
        source.write_text(page_html, encoding="utf-8")
        subprocess.run(
            # ห้ามใส่ --user-data-dir: โหมด headless ใช้โปรไฟล์ชั่วคราวอยู่แล้ว
            # และการบังคับโปรไฟล์ใหม่ทำให้ Chrome ค้างเมื่อผู้ใช้เปิด Chrome ไว้อยู่
            [
                browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={target}",
                "--virtual-time-budget=8000",
                *SERVER_FLAGS,
                source.as_uri(),
            ],
            check=True, capture_output=True, timeout=timeout,
        )
        if not target.exists():
            raise ChromeNotFound("Chrome ทำงานแล้วแต่ไม่ได้สร้างไฟล์ PDF ออกมา")
        return target.read_bytes()


def build_pdf(session, source_label: str, critical_image=None,
              participant: Participant | None = None) -> bytes:
    """ทางลัดที่ใช้จริงในหน้าเว็บ: ประกอบ HTML แล้วแปลงเป็น PDF"""
    return render_pdf(build_html(session, source_label, critical_image,
                                  participant=participant))


def csv_bytes(session, participant: Participant | None = None) -> bytes:
    """ส่งออกผลรายเฟรมเป็น CSV โดยเลือกแปะคอลัมน์ผู้ทดสอบซ้ำทุกแถว

    ทำให้นำ CSV ของหลายคนมาต่อกันเป็นตารางเดียวเพื่อวิเคราะห์รวมได้ทันที
    โดยไม่ต้องเปิดไฟล์แยกดูว่าแถวไหนเป็นของใคร เมื่อไม่มีผู้ทดสอบ ผลลัพธ์ต้อง
    เหมือนกับ session.to_csv_bytes() ทุกไบต์ จึงปล่อยให้ฟังก์ชันนั้นยังเป็น
    ทางลัดเดิมสำหรับกรณีไม่มีผู้ทดสอบ
    """
    frame = session.to_dataframe()
    if participant is not None:
        frame = frame.copy()
        frame.insert(0, "participant_bmi", participant.bmi)
        frame.insert(0, "participant_height_cm", participant.height_cm)
        frame.insert(0, "participant_weight_kg", participant.weight_kg)
        frame.insert(0, "participant_gender", participant.gender)
        frame.insert(0, "participant_age", participant.age)
    return frame.to_csv(index=False).encode("utf-8-sig")

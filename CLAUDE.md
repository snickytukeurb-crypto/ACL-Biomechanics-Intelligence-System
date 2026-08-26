# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit tool that scores ACL (anterior cruciate ligament) injury risk from video or
webcam, using MediaPipe pose landmarks. The scoring model is fixed and externally
specified — the code exists to implement it faithfully, not to redefine it. Treat the
weights, reference values and band thresholds in `acl/risk.py` as settled.

`README.md` is written in Thai and covers install/usage; this file covers what an agent
needs that the README does not.

## Commands

```bash
.venv/bin/streamlit run app.py            # run the app
.venv/bin/python -m pytest tests/ -q      # full suite (75 tests)
.venv/bin/python -m pytest tests/test_risk.py -q          # one file
.venv/bin/python -m pytest tests/ -q -k valgus            # one topic
.venv/bin/python -m pytest tests/test_biomech.py::test_straight_leg_is_180_degrees
```

There is no linter or formatter configured.

## Environment constraints

- **Python 3.12 only.** MediaPipe publishes no wheels for 3.13+. The venv is built from
  `/opt/homebrew/Cellar/python@3.12/3.12.14/bin/python3.12` because Homebrew's post-install
  step failed, so 3.12 is not linked into `PATH`.
- **Renaming the project directory breaks the venv.** Every script in `.venv/bin` hardcodes
  an absolute shebang. After a move, `sed -i '' "s|<old path>|<new path>|g"` across the text
  files in `.venv/bin` and `.venv/pyvenv.cfg` restores it. This has already happened once
  (the repo was `githubrepo`).
- **MediaPipe uses the Tasks API, not `mp.solutions`.** The legacy `mp.solutions.pose` was
  removed upstream; `acl/pose.py` uses `PoseLandmarker` in VIDEO mode, which requires
  monotonically increasing timestamps and a `models/pose_landmarker_full.task` file
  (gitignored, ~9 MB — download command is in `README.md`).
- **PDF export shells out to headless Google Chrome**; **video export needs `ffmpeg`**
  (`/opt/homebrew/bin/ffmpeg`). Both degrade with a clear message when missing.
- Sample footage lives **outside the repo** at `../Video/*.MP4`. `C0006`/`C0007`/`C0009`
  contain real landings; `C0004`/`C0010`/`C0011` have no detectable person.

## Deployment (Streamlit Community Cloud)

`packages.txt` and `.streamlit/config.toml` exist for the hosted copy. The deploy needs
**Python 3.12 picked in Advanced settings**; the default is newer and MediaPipe has no
wheel for it.

**`packages.txt` holds exactly one line, `libgl1`, and adding to it is how the deploy
breaks.** The image mixes Debian trixie with a bullseye-security repo, so apt cannot solve
`chromium`, `ffmpeg`, or `libglib2.0-0` (bullseye's `libglib2.0-0` conflicts with the
`libglib2.0-0t64` trixie needs). Any unsolvable line fails the whole install step and the
app never boots — it is not a per-feature degradation. It is also fed straight to apt-get,
so it must stay a bare list: no comments, no blank lines. Solve things with Python
packages instead, the way `imageio-ffmpeg` ships its own ffmpeg binary. The consequence: **the hosted copy cannot
render PDFs itself** (no Chromium), so the export row swaps the "สร้างรายงาน PDF" button
for an HTML download whenever `report.find_chrome()` is None — the viewer prints it from
their own browser. `build_html` carries a `.printbar` button that `@media print` hides, so
the same HTML serves both routes and the button never reaches the paper.

Three things differ on a server and are handled in code, not by hand:
`pose.ensure_model()` downloads the gitignored `.task` file on first use (called from
`app.py` behind `st.cache_resource`); the webcam option is hidden when no camera exists
(`WEBCAM_AVAILABLE`, override with `ACL_ENABLE_WEBCAM=1`); and `report.SERVER_FLAGS` adds
`--no-sandbox --disable-dev-shm-usage` to Chromium on Linux only, so macOS behaviour is
untouched.

`video.open_writer` also forks by platform: on Linux it returns `FfmpegWriter`, which pipes
raw BGR frames into ffmpeg, because the Linux OpenCV wheels ship no H.264 encoder and
`avc1` either fails outright or writes a file browsers refuse to play. macOS keeps the
`cv2.VideoWriter` path (VideoToolbox). `find_ffmpeg()` falls back to the `imageio-ffmpeg`
binary last, so local runs still get Homebrew's ffmpeg — which matters because
`tests/test_video.py` derives `ffprobe` from that path and the bundled binary has no
`ffprobe` beside it.

## Architecture

`app.py` is presentation only — every calculation lives in `acl/`. Keep it that way.

| Module | Role |
|---|---|
| `acl/risk.py` | The ACL Risk Index. Weights, reference values, bands. No dependencies. |
| `acl/biomech.py` | Knee angle, knee valgus, backward finite difference, moving-average smoother. Pure geometry. |
| `acl/pose.py` | MediaPipe wrapper; landmark extraction and overlay drawing. |
| `acl/session.py` | Per-frame records, critical-frame selection, CSV, live badge stabiliser. |
| `acl/report.py` | Summary table, matplotlib charts, participant value object, HTML→PDF. |
| `acl/video.py` | H.264 writing and ffmpeg rescaling. |

Data flow per frame: `pose.world_leg` → `Session.update` → `LegTracker` (smooth θ →
differentiate to ω → smooth ω → differentiate to α) → `risk.risk_index` → `FrameRecord`.

Two design points that are easy to break:

- **`Session` is mutated in place and stored in `st.session_state` *before* `analyse()`
  runs.** Pressing "หยุด" makes Streamlit raise a rerun exception inside the frame loop, so
  anything after the `analyse()` call never executes. Moving that assignment back below the
  call silently discards every webcam result.
- **`Session.update` accumulates elapsed time from `dt`**, rather than deriving it from the
  frame index, so webcam sessions (variable frame rate) get a truthful time axis.

## Rules specific to this project

**The risk model is pinned to an external reference set.** `tests/test_risk.py` asserts
against a fixed set of reference values that live outside this repo. Never weaken those
assertions; if one fails, the model drifted rather than the fixture being wrong.

**Never commit anything under `Source/` or `TestResult/`.** `Source/` holds an unpublished
document and `TestResult/` holds exported runs containing participant data; both are
gitignored, and the repository is public. This has gone wrong once already — the document
was pushed publicly, and the only reliable fix was deleting and recreating the repository,
because a force-push leaves the old blob fetchable by its SHA. Do not add these paths back
to git, and do not remove them from `.gitignore`.

**Participant fields are metadata only.** Age, gender, weight and height reach the PDF header
and CSV columns and nothing else. The original broken version of this app passed age and
gender into the risk function — that was a bug, and the model takes exactly four variables
(θ, γ, ω, α). They must never reach `risk_index`.

**Three calculation choices are deliberate, not accidents** — the knee-valgus definition,
smoothing before differentiation, and using |ω|. Each is justified by measurements from the
sample footage and explained in `README.md` under "รายละเอียดการคำนวณที่ควรทราบ". Read that
section before changing `biomech.py` or `risk.py`, and preserve the property that
`SMOOTHING_WINDOW = 1` turns the filtering off entirely.

**`cv2.putText` is ASCII-only.** On-frame overlays use `KNEE / VALGUS / VEL / ACC`; Greek
letters and Thai render as boxes. Thai is fine everywhere else (Streamlit, PDF via Chrome).

**Comments and UI strings are Thai; identifiers are British-ish English** (`analyse`,
`stabiliser`, `summarise`). Comments explain *why*, not *what*.

**Tuning constants live in a `ตั้งค่าหลังบ้าน` block at the top of `app.py`**, deliberately not
exposed in the UI so every test runs under identical conditions. `LIVE_LAYOUT_RATIO` and
`RESULT_VIDEO_WIDTH` control layout sizing; note that the on-screen "ขนาดวิดีโอ" control sets
the encoded file resolution, which is a different thing from player size.

**Streamlit:** a `developing-with-streamlit` skill is installed under `.claude/skills/` —
invoke it for UI work. `use_container_width` is deprecated; use `width="stretch"`.

## Verifying UI changes

`streamlit.testing.v1.AppTest` runs the app headlessly and catches exceptions and widget
state — this has caught real bugs (a string column formatted as a float, a widget getting
its default from two places). It cannot render pixels.

For anything about layout or what is actually visible, drive a real browser: launch Chrome
with `--remote-debugging-port` and a dedicated `--user-data-dir` (without its own profile it
hands off to a running Chrome and exits), then use the Chrome DevTools Protocol to click,
upload via `DOM.setFileInputFiles`, and screenshot. Chrome's plain `--screenshot` flag only
ever captures the loading skeleton for this app, because `--virtual-time-budget` stalls the
websocket Streamlit needs. Comparing `document.documentElement.scrollHeight` against
`window.innerHeight` is how the "fits on one screen" requirement is checked.

Note that Streamlit leaves stale elements from the previous run on screen while a long loop
is executing; check `data-stale` before treating leftover content as a layout bug.

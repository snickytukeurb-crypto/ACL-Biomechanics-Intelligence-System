# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit tool that scores ACL (anterior cruciate ligament) injury risk from an
uploaded clip or the viewer's own browser camera, using MediaPipe pose landmarks. The scoring model is fixed and externally
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
- **PDF export shells out to headless Google Chrome.** Without it the page offers the
  report as HTML for the viewer to print instead (see Deployment). **ffmpeg is never
  missing**: `find_ffmpeg()` prefers a system one (`/opt/homebrew/bin/ffmpeg`) and falls
  back to the binary `imageio-ffmpeg` ships.
- Sample footage lives **outside the repo** at `../Video/*.MP4`. `C0006`/`C0007`/`C0009`
  contain real landings; `C0004`/`C0010`/`C0011` have no detectable person.

## Deployment (Streamlit Community Cloud)

Hosted at `acl-bis.streamlit.app`, tracking `main`. `packages.txt` and
`.streamlit/config.toml` exist for that copy, and the deploy needs **Python 3.12 picked in
Advanced settings** — the default is newer and MediaPipe has no wheel for it.

**`packages.txt` is the fragile part.** It is fed straight to apt-get, so it must stay a
bare list (no comments, no blank lines), and a single line apt cannot solve fails the whole
install step — the app then never boots at all, rather than losing one feature. It holds
exactly the libraries the imports need:

| line | needed by |
|---|---|
| `libgl1`, `libglib2.0-0t64` | `import cv2` — `libGL.so.1`, `libgthread-2.0.so.0` |
| `libegl1`, `libgles2` | `PoseLandmarker.create_from_options` — `libEGL.so.1`, `libGLESv2.so.2` |

Every one of those was found by reading the deploy log *after* a failure, never by guessing:
the traceback names the exact `.so`, and
`curl -s https://packages.debian.org/trixie/amd64/<pkg>/filelist` confirms which package
ships it. Follow that loop rather than adding hopeful lines. The image mixes Debian trixie
with a bullseye-security repo, which is why `chromium` and `ffmpeg` cannot be solved at all,
and why glib must be asked for by its trixie name — plain `libglib2.0-0` pulls bullseye's,
which `Breaks` the `libglib2.0-0t64` everything else needs. Given a choice, solve it with a
Python package instead: `imageio-ffmpeg` ships its own ffmpeg and needs no apt line.

Consequences, all handled in code rather than by hand:

- **No Chromium means the hosted copy cannot render PDFs.** The export row swaps the
  "สร้างรายงาน PDF" button for an HTML download whenever `report.find_chrome()` is None, and
  the viewer prints it from their own browser. `build_html` carries a `.printbar` button that
  `@media print` hides, so one HTML serves both routes and the button never reaches paper.
- **The `.task` model file is not in git.** `pose.ensure_model()` downloads it on first use,
  called from `app.py` behind `st.cache_resource`.
- `report.SERVER_FLAGS` adds `--no-sandbox --disable-dev-shm-usage` to Chromium on Linux
  only, so macOS behaviour is untouched.
- The camera mode does **not** differ by host: it always captures through the *viewer's*
  browser via `streamlit-webrtc` (`WebRtcMode.SENDRECV`, `LiveProcessor` in `app.py`), never
  a server-side device. It needs HTTPS (or `localhost`) and, on strict-NAT networks, a TURN
  server beyond the public STUN one configured — out of scope, see `README.md`.

The free tier has roughly 1 GB of memory, which this pipeline used to exceed. Uploads are
streamed to a temp file instead of held in `st.session_state`, frames are downscaled to
`PROCESS_MAX_WIDTH` before MediaPipe, and the per-frame table renders through
`column_config` rather than a pandas `Styler` — together those took peak RSS on
`C0006.MP4` from 1.11 GB to 424 MB. Keep new work inside that budget.

`video.open_writer` uses `FfmpegWriter` on every platform now, piping raw BGR frames into
`libx264` with a fixed CRF (`VIDEO_CRF` in `app.py`, currently 28) and
`-movflags +faststart`. `cv2.VideoWriter` (`avc1`) is a last-resort fallback only, used
solely when `find_ffmpeg()` finds nothing — which in practice never happens, because
`imageio-ffmpeg` ships its own binary. The switch away from `cv2.VideoWriter` as the
macOS-primary path was deliberate: it cannot control compression at all, so files came out
several times larger than necessary. `find_ffmpeg()` still prefers a system ffmpeg over the
bundled one when both exist, so local runs get Homebrew's ffmpeg — which matters because
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

Design points that are easy to break:

- **`Session` is mutated in place and stored in `st.session_state` *before* `analyse()`
  runs.** Pressing "หยุด" makes Streamlit raise a rerun exception inside the frame loop, so
  anything after the `analyse()` call never executes. Moving that assignment back below the
  call silently discards every upload result. The browser-camera flow has the same trap in a
  different shape: `ctx.video_processor` (from `webrtc_streamer`) is only non-`None` while
  the stream is playing, so `app.py` stashes it in `st.session_state.live_processor` on every
  rerun and only reads `.session`/`.critical_image`/`.video_path` off it once the value goes
  back to `None` (i.e. just after STOP). Skipping that stash silently discards the result the
  same way.
- **`Session.update` accumulates elapsed time from `dt`**, rather than deriving it from the
  frame index, so live sessions (variable frame rate — camera or processing load) get a
  truthful time axis.
- **`LiveProcessor` (the `streamlit-webrtc` video processor) runs on its own worker thread**,
  separate from the Streamlit script thread — `__init__`, `recv()` and `on_ended()` must
  never call `st.*`. Anything the page needs to show live is drawn onto the frame itself
  (same `draw_risk_banner`/`draw_metric_strip` calls the upload path uses, via the shared
  `process_frame` helper); only the final summary is handed back to the main script, and only
  after the stream has stopped.

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
the encoded file resolution, which is a different thing from player size. `LIVE_FPS` is the
frame rate asked of the browser camera and the nominal `Session` fps — real time always comes
from measured `dt`. `VIDEO_CRF` sets output compression, `PROCESS_MAX_WIDTH` the downscale
before MediaPipe (0 disables it).

**`PROCESS_MAX_WIDTH` is the one tuning constant that can move the numbers**, so it was
validated rather than assumed: running `../Video/C0006.MP4` at full 1920px versus 1280px
gives the same critical frame (74) and R 50.16 vs 50.25. Re-run that comparison — same
critical frame, |ΔR| ≤ 1.0 — before changing it, driving `pose`/`Session` directly instead of
going through the page.

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

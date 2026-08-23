# roam-count

Real-time **unique people counter** for a **moving (hand-carried) camera**.
No image or video is stored in production -- only numbers + timestamps.

See [`people-counting-dev-plan.md`](people-counting-dev-plan.md) for the full plan.

## Status

- ✅ **Phase 0 — Foundation**: venv, module structure, config-driven camera source.
- ✅ **Phase 1 — Detection**: YOLO11n person detection, live occupancy count + FPS.
- ✅ **Phase 2 — Tracking + unique count**: BoT-SORT + ReID + camera-motion
  compensation, cumulative unique count, JSONL event log (numbers only).
- 🧩 **Phase 3 — Golden set + tuning**: harness ready (`evaluate.py`,
  `golden_set.json`, `tuning_log.md`). Waiting on real hand-carried clips +
  eyeball counts to actually tune to error ≤ 15%.
- ✅ **Phase 4 — Ship PC version**: big overlay, hotkeys (quit/pause/reset),
  camera auto-reconnect, graceful shutdown, session lifecycle log. (1-hour soak +
  unplug/replug tests are yours to run on real hardware.)
- 🧩 **Phase 5 — Mobile**: groundwork on PC — `counter.py` ported to Kotlin
  (`mobile/`), ONNX export done. TFLite export needs Linux/WSL (see `mobile/README.md`).
  Android app + real-device validation not built here.

## Setup

```bash
python -m venv .venv

---1. open PowerShell open computer every things
.venv\Scripts\activate            # Windows (PowerShell/cmd)
# source .venv/bin/activate       # macOS / Linux

# GPU (NVIDIA) -- install torch from the CUDA index first:
---2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
---3. pip install -r requirements.txt

# CPU-only -- just:
# pip install -r requirements.txt
```

First run auto-downloads `yolo11n.pt` (needs internet).

## Run

```bash
---4. python main.py                    # live TRACKING + unique count (Phase 2, default)
python main.py --source clip.mp4  # run against a video file instead
python main.py --detect-only      # Phase 1: detection only, no tracking/count
python main.py --imgsz 480        # override input size (higher FPS, lower accuracy)
python main.py --bench --sweep    # headless FPS benchmark at imgsz 640 / 480 / 320
python main.py --bench-raw        # Phase 0: raw camera read FPS (no model)
python main.py --no-log           # do not write the JSONL event log
```

Live-mode hotkeys: **q** or **ESC** to quit.

The event log (`logs/events.jsonl`, gitignored) records **numbers + timestamps
only** — one line per `new_track` / `lost` event, never any image (plan §7):

```json
{
  "ts": "2026-08-23T17:16:32Z",
  "event": "new_track",
  "track_id": 4,
  "unique_total": 4
}
```

Run the pure counting-logic tests any time:

```bash
python test_counter.py
```

## Files

| File              | Role                                                                        |
| ----------------- | --------------------------------------------------------------------------- |
| `config.json`     | all thresholds live here -- never hard-coded (portability guardrail)        |
| `camera.py`       | camera source abstraction (webcam / RTSP / file)                            |
| `detector.py`     | YOLO11n person detector (class 0 only) + shared inference params            |
| `tracker.py`      | BoT-SORT tracking (ReID + GMC); generates `botsort_custom.yaml` from config |
| `counter.py`      | **pure** counting logic (no OpenCV/torch) -- ports to Kotlin 1:1            |
| `test_counter.py` | unit tests for the pure counter (run without a camera/model)                |
| `main.py`         | entry point: live tracking / detection / benchmark modes                    |
| `evaluate.py`     | Phase 3: run golden clips, report count error %, sweep one param at a time   |
| `golden_set.json` | ground-truth unique counts per clip (numbers only; committed as baseline)    |
| `tuning_log.md`   | record of every tuning round (evidence)                                     |

## Config quick reference (`config.json`)

| Key               | Meaning                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `camera.source`   | `0` = webcam, `"rtsp://..."` = phone stream, `"clip.mp4"` = file |
| `detector.imgsz`  | inference size; lower = faster, less accurate (640 → 480 → 320)  |
| `detector.conf`   | confidence threshold for keeping a detection                     |
| `detector.device` | `"auto"` (GPU if available, else CPU), or `"cuda:0"` / `"cpu"`   |
| `detector.half`   | FP16 on GPU (auto-disabled on CPU)                               |
| `detector.iou` | NMS IoU threshold (overlap above which boxes are merged) |
| `detector.classes` | COCO class ids to keep — `[0]` = person only |
| `tracker.with_reid` | appearance matching so people re-entering aren't re-counted |
| `tracker.gmc_method` | camera-motion compensation: `sparseOptFlow` / `orb` / `ecc` / `none` |
| `tracker.track_buffer` | frames a lost track survives; hand-carried → keep low (~30) |
| `tracker.match_thresh` | association strictness (motion/IoU) |
| `tracker.new_track_thresh` | min confidence to start a new track (higher = fewer ghosts) |
| `counter.min_track_age_frames` | frames a track must survive before it's counted |
| `counter.max_history` | cap on remembered track ids (bounds memory over long runs) |
| `output.show_window` | show the live window (`false` = headless) |
| `output.log_path` | JSONL event log (numbers only, never images) |
| `output.save_frames` | must stay `false` — privacy guarantee (no frames to disk) |

## Hotkeys (live tracking)

| Key | Action |
|---|---|
| `q` / `ESC` | quit (writes a clean `session_end` to the log) |
| `p` / `SPACE` | pause / resume |
| `r` | reset the unique count (start a new walk without restarting) |

If a live camera is unplugged, the app shows **CAMERA LOST** and auto-reconnects
with backoff — it won't crash. `Ctrl+C` also shuts down cleanly.

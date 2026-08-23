# Phase 5 — Port to Mobile (on-device)

Status: **groundwork done on PC; the Android app itself is not built here.** This
folder holds what could be produced + verified without an Android device, plus the
decisions and exact steps for the rest.

## What's done (verifiable now)

| Item | Status |
|---|---|
| `UniqueCounter.kt` — 1:1 port of `counter.py` (pure logic) | ✅ written |
| `UniqueCounterTest.kt` — same 11 cases as `test_counter.py` | ✅ written (⚠️ NOT EXECUTED — no Kotlin toolchain here; identical cases pass 11/11 in Python) |
| ONNX export of `yolo11n` | ✅ `yolo11n.onnx` (10.2 MB) |
| Export sanity (`.pt` vs `.onnx`, same image) | ✅ **person count identical 4=4**; mean conf 0.811 → 0.746; FPS 51 (CUDA) vs 30 (onnxruntime-CPU) |

## ⛔ The export blocker (important)

`ultralytics` 8.4 renamed `tflite` → **LiteRT**, and **LiteRT/TFLite export only
runs on Linux x86 / macOS** — it hard-fails on Windows:

```
AssertionError: LiteRT export only supported on Linux x86 and macOS
```

So the `.tflite int8` model **cannot be produced on this Windows machine.** Options,
easiest first:

1. **WSL2 (Ubuntu) on this same PC** — install ultralytics there and run:
   ```bash
   yolo export model=yolo11n.pt format=tflite int8=True data=coco8.yaml imgsz=640
   ```
   (Use a *representative* calibration set instead of `coco8` for real accuracy.)
2. **Google Colab** (free Linux) — same command, then download the `.tflite`.
3. **Ultralytics Platform** (cloud export) — link printed by the failed export.
4. **NCNN** (also mobile-friendly, the plan's alt) may build via WSL too:
   `yolo export model=yolo11n.pt format=ncnn`.

ONNX (done here) is a valid alternative deployment via **ONNX Runtime Mobile** if you
prefer not to use TFLite.

## Measure quantization impact BEFORE trusting it (plan Phase 5)

Do this on PC (Linux/WSL) once you have the `.tflite`, so you can separate
"accuracy lost to quantization" from "accuracy lost to a different tracker". The
harness already supports it — point `--model` at each export and run the golden set:

```bash
python evaluate.py --model yolo11n.pt              --split tune   # baseline
python evaluate.py --model yolo11n_saved_model/yolo11n_int8.tflite --split tune
```
Compare the `mean_err` lines. (Needs real golden clips in `golden_set.json`.)

⚠️ Windows caveat: running the **ONNX** model through the tracker needs
`detector.device: "cpu"` — `onnxruntime-gpu` throws a tensor-binding error here.
The `.tflite` A/B is meant to run on Linux/WSL anyway, so this only affects local
ONNX spot-checks. Also observed: on a short proxy clip the ONNX model counted 3 vs
the `.pt`'s 4 — a real reminder that export shifts results, so always re-measure.

## Tracker decision: A (ByteTrack) vs B (BoT-SORT + ReID)

🛑 Decide **after** seeing numbers, not now. What we know from PC so far:

- BoT-SORT **+ ReID + GMC** ran **23 FPS** on an RTX 4050 (desktop GPU). A phone is
  much weaker, and **there is no ready-made BoT-SORT for Android** — you'd port it.
- The camera is **hand-carried (moving)**. ReID + GMC are exactly what fights the
  double-counting that camera motion causes (plan §1.5, §2.2).

| | A — ByteTrack in Kotlin | B — BoT-SORT + ReID in Kotlin |
|---|---|---|
| Effort | Lower (motion-only, well-documented) | High (must implement ReID + GMC + a 2nd model) |
| Accuracy on moving camera | Worse — re-counts people who leave/return | Better — appearance matching prevents that |
| Extra inference | none | +1 ReID pass per frame → lower FPS |
| Recommendation | **start here to hit the ≥15 FPS DoD**, measure error | only if A's error is unacceptable AND FPS allows |

**Decision rule:** build A first → run golden clips → if count error is within
target and FPS ≥ 15, ship A. If A's error is too high *because of re-entries*, then
invest in B. Record the numbers in `../tuning_log.md`.

## Android app skeleton (not built here)

```
CameraX (ImageAnalysis, YUV->RGB)
      -> TFLite Interpreter (GPU/NNAPI delegate)   // or ONNX Runtime Mobile
      -> parse boxes, filter class 0 (person)
      -> tracker (A or B, ported to Kotlin)
      -> UniqueCounter.kt   <-- already ported, drop-in
      -> overlay (unique total big + occupancy + FPS)  // mirror main.py HUD
```
Keep the same rules as PC: **no frame written to disk**, ReID embeddings in RAM only
(plan §7). Thermal throttling will drop FPS after minutes — test a sustained run.

## Validate against the same golden set (regression)

Feed the **golden clips as video files** (not the live camera) into the Android
build and compare its unique counts to the PC numbers from `golden_set.json`. That
turns "is mobile good enough?" into a concrete error-% table (plan Phase 5 DoD),
instead of a guess.

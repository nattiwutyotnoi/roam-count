# roam-count — Android scaffold

> **STATUS: uncompiled scaffold (EXAMPLE ONLY).** These files were written without an
> Android toolchain, so **nothing here has been built or run.** They give a correct
> *structure* and the pure logic; the device-specific parts are marked `TODO` and must
> be verified in Android Studio on a real phone.

## What's here

```
mobile/
├── UniqueCounter.kt          ✅ pure logic, 1:1 port of counter.py (verified via Python tests)
├── UniqueCounterTest.kt      ✅ same 11 cases (run as JVM unit test)
└── android/
    └── app/
        ├── build.gradle.kts
        └── src/main/
            ├── AndroidManifest.xml
            └── java/com/roamcount/
                ├── Detection.kt         ✅ data classes
                ├── Nms.kt               ✅ IoU + NMS (pure, testable)
                ├── ByteTrackLite.kt     ⚠️ Option-A tracker (IoU only, no Kalman/ReID)
                ├── PersonDetector.kt    ⚠️ TFLite wrapper — VERIFY output layout + quant
                ├── CountingAnalyzer.kt  ⚠️ CameraX glue — VERIFY YUV->Bitmap
                ├── OverlayView.kt       ✅ boxes + HUD
                └── MainActivity.kt      ⚠️ CameraX wiring skeleton
```

**Move `mobile/UniqueCounter.kt` and `mobile/UniqueCounterTest.kt` into
`app/src/main/java/com/roamcount/` and `app/src/test/java/com/roamcount/`** — they are
already in package `com.roamcount`. That's your one piece of logic that is proven equal
to the PC version.

## The data-flow (identical shape to the PC pipeline)

```
CameraX ImageAnalysis ─▶ PersonDetector (TFLite) ─▶ ByteTrackLite ─▶ UniqueCounter ─▶ OverlayView
      frame                boxes (class 0)            track ids        unique total       HUD
```

## Before it will work — verification checklist

1. **Export the model on Linux/WSL** (TFLite export can't run on Windows — see
   `../README.md`), put `yolo11n.tflite` in `app/src/main/assets/`.
2. **Open the .tflite in [netron.app](https://netron.app)** and confirm in `PersonDetector.kt`:
   - output shape / layout (code assumes `(1, 8400, 84)` = `[cx,cy,w,h,80 class scores]`),
   - whether box coords are pixels `[0,640]` or normalized `[0,1]`,
   - int8 vs float I/O (add dequant/quant for int8).
3. **YUV→Bitmap** in `CountingAnalyzer.kt`: simplest is
   `ImageAnalysis.setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)` + `ImageProxy.toBitmap()`.
4. **Delegate**: try `GpuDelegate`; fall back to NNAPI / CPU threads. Measure FPS.
5. **Validate accuracy**: feed the **golden clips as video files** and compare unique
   counts to the PC numbers in `../../golden_set.json` (plan Phase 5 DoD).

## Tracker choice (plan section 4.5)

`ByteTrackLite` (Option A) ships here to hit the **≥15 FPS** DoD first. It has **no ReID
and no motion prediction**, so it will over-count when people leave and re-enter and when
the camera moves fast. If the golden-set error is unacceptable *for that reason*, invest
in **BoT-SORT + ReID (Option B)**. Decide from the measured numbers, not upfront.

## Privacy (same guarantees as PC — plan section 7)

- No frame or video is ever written to disk. No storage permission is requested.
- If you add Option B, keep ReID embeddings in RAM only; clear them when a track ages out.

## Also needed (standard Android boilerplate, not scaffolded)

`settings.gradle.kts`, root `build.gradle.kts`, `gradle/wrapper`, a layout XML with a
`PreviewView` + `OverlayView` + a Reset button, and `res/values/themes.xml`. Android
Studio's "New Project" generates these — paste these files on top.

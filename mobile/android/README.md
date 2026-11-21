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
                ├── YoloDecoder.kt       ✅ pure tensor/box logic (NCHW input, [1,84,8400] output, normalized boxes, int8) — JVM-tested
                ├── PersonDetector.kt    ⚠️ TFLite wrapper; reads layouts/types from the model — NOT compiled, verify on device
                ├── CountingAnalyzer.kt  ⚠️ CameraX glue — VERIFY YUV->Bitmap
                ├── OverlayView.kt       ✅ boxes + HUD
                └── MainActivity.kt      ⚠️ CameraX wiring skeleton
```

`UniqueCounter.kt` / `UniqueCounterTest.kt` now live in `app/src/main/java/com/roamcount/` and
`app/src/test/java/com/roamcount/` (moved from `mobile/`). That's the one piece of logic proven
equal to the PC version (11/11 under kotlinc).

## Run the pure-logic tests (JVM, no Android needed)

```bash
cd app/src
kotlinc main/java/com/roamcount/Detection.kt main/java/com/roamcount/YoloDecoder.kt \
        test/java/com/roamcount/YoloDecoderTest.kt -include-runtime -d t.jar
java -cp t.jar com.roamcount.YoloDecoderTest      # 14 passed
```
These prove the decode math on synthetic tensors (layouts, normalized vs pixel boxes,
letterbox undo, int8). They do not prove the real `.tflite` — that is the on-device check above.

## The data-flow (identical shape to the PC pipeline)

```
CameraX ImageAnalysis ─▶ PersonDetector (TFLite) ─▶ ByteTrackLite ─▶ UniqueCounter ─▶ OverlayView
      frame                boxes (class 0)            track ids        unique total       HUD
```

## Before it will work — verification checklist

1. **Export the model on Linux/WSL** (TFLite export can't run on Windows — see
   `../README.md`), put `yolo11n.tflite` in `app/src/main/assets/`.
2. **Check the model I/O.** `PersonDetector` now reads the input/output shapes and fp32/int8
   types from the loaded model, handles both layouts, and throws on anything else (no more
   silent garbage). For the ultralytics 8.4 export expect: input NCHW `[1,3,640,640]`,
   output attribute-major `[1,84,8400]`, boxes normalized `[0,1]`. Log `detector.describe()`
   once at startup (or open the .tflite in [netron.app](https://netron.app)) and confirm.
   The one thing a shape cannot reveal is the box scale: `boxesNormalized` (default `true`).
   Check on bus.jpg: 4 persons, boxes within ~3 px of the `.pt` model.
   fp32 (~11 MB) and int8 (~3 MB) are both supported via the tensor's quantization params.
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

## Build setup (now included, still never built here)

`settings.gradle.kts`, root `build.gradle.kts`, `gradle.properties`, `gradle/wrapper/gradle-wrapper.properties`
(no `gradle-wrapper.jar`; Android Studio downloads Gradle 8.7 itself), `res/layout/activity_main.xml`
(PreviewView + OverlayView + Reset button), `res/values/themes.xml`, `proguard-rules.pro`.

**Open `mobile/android` (not `mobile`) as the project root.** Pinned: AGP 8.5.2, Kotlin 1.9.24,
Gradle 8.7, compileSdk 34, JDK 17 (Android Studio's bundled JBR is fine). If Studio offers an AGP
upgrade or asks to install SDK Platform 34, accept. Nothing here was compiled against the Android SDK
(Google's Maven/SDK hosts are unreachable from the dev sandbox), so expect to fix a few compile errors.

The model is not in the repo (`*.tflite` is gitignored): copy `yolo11n.tflite` into `app/src/main/assets/`
before running; the build itself does not need it.

package com.roamcount

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import android.widget.Button
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

// EXAMPLE ONLY / NOT COMPILED.
//
// Wires CameraX Preview + ImageAnalysis(CountingAnalyzer) and shows the overlay.
// Loads config-equivalent thresholds; keeps the SAME rules as PC:
//   * no frame written to disk (privacy)   * ReID/embeddings (Option B) in RAM only
//   * a Reset button maps to counter.reset() + tracker.reset() (the PC 'r' hotkey)

class MainActivity : ComponentActivity() {

    private companion object {
        const val TAG = "RoamCount"
        const val EMPTY_RESET_SECONDS = 2L // frame empty this long -> unique total back to 0 (0 = off)
    }

    private lateinit var overlay: OverlayView
    private lateinit var previewView: PreviewView

    // Built once (loading the model is slow) and reused every time the camera is rebound.
    private var detector: PersonDetector? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var useFrontCamera = false

    // Inference must NOT run on the main thread. A single thread also serializes analyze()
    // and reset, so counter/tracker never see concurrent access.
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val counter = UniqueCounter(minTrackAgeFrames = 5, maxHistory = 10_000)
    private val tracker = ByteTrackLite(trackBuffer = 30)

    // Unique total returns to 0 once the frame has been empty this long. A grace period is
    // needed so a few missed detections / someone stepping out briefly don't wipe the count.
    // Set to 0 to turn the auto-reset off (then only the Reset button clears the total).
    private val emptyReset = EmptyFrameReset(graceNanos = EMPTY_RESET_SECONDS * 1_000_000_000L)

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        findViewById<Button>(R.id.resetButton).setOnClickListener { onResetClicked() }
        findViewById<Button>(R.id.switchCameraButton).setOnClickListener { onSwitchCameraClicked() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun buildDetector(): PersonDetector {
        val model = loadAsset("yolo11n.tflite") // put the exported model in app/src/main/assets/
        // The GPU delegate is only *applied* when the Interpreter is constructed, so the fallback must
        // wrap the whole PersonDetector(...) call, not addDelegate(). (Seen on the emulator: the GPU
        // delegate rejected this graph -- "Batch size mismatch" -- and Interpreter() threw.)
        val detector = try {
            val gpu = Interpreter.Options().apply { addDelegate(GpuDelegate()) }
            PersonDetector(model, gpu, inputSize = 640, confThresh = 0.35f, iouThresh = 0.5f).also {
                Log.i(TAG, "delegate: GPU")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "GPU delegate unusable (${e.message?.lineSequence()?.firstOrNull()}); falling back to CPU")
            val cpu = Interpreter.Options().apply { setNumThreads(4) }
            PersonDetector(model, cpu, inputSize = 640, confThresh = 0.35f, iouThresh = 0.5f).also {
                Log.i(TAG, "delegate: CPU (4 threads)")
            }
        }
        Log.i(TAG, "model: ${detector.describe()}") // verify NCHW / [1,84,8400] / fp32|int8 on the real device
        selfTest(detector)
        return detector
    }

    /**
     * Optional on-device check: if assets/selftest.jpg exists (e.g. ultralytics' bus.jpg), run the
     * detector once and log the result. Expect 4 persons for bus.jpg (PC .pt model: 4, boxes within
     * ~6 px). Never writes anything to disk; skipped silently when the asset is absent.
     */
    private fun selfTest(detector: PersonDetector) {
        val bmp = try {
            assets.open("selftest.jpg").use { BitmapFactory.decodeStream(it) }
        } catch (_: java.io.IOException) {
            return
        } ?: return
        val t0 = System.nanoTime()
        val dets = detector.detect(bmp)
        val ms = (System.nanoTime() - t0) / 1e6
        Log.i(TAG, "selftest: ${dets.size} persons in %.0f ms (first call, includes warm-up)".format(ms))
        dets.sortedBy { it.x1 }.forEach {
            Log.i(TAG, "selftest box x1=%.1f y1=%.1f x2=%.1f y2=%.1f conf=%.2f".format(it.x1, it.y1, it.x2, it.y2, it.conf))
        }
        // Non-person objects (red boxes on screen): bus.jpg should show a bus, never a person.
        Log.i(TAG, "selftest non-person: " + detector.lastNonPersons.joinToString { "${CocoLabels.name(it.cls)} %.2f".format(it.conf) }.ifEmpty { "none" })
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            if (detector == null) detector = buildDetector()
            // Offer the switch only when the phone has both cameras.
            findViewById<Button>(R.id.switchCameraButton).visibility =
                if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) &&
                    provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
                ) View.VISIBLE else View.GONE
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    /** (Re)bind preview + analysis to the selected camera. Safe to call repeatedly. */
    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val det = detector ?: return
        val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        overlay.setMirrored(useFrontCamera)
        Log.i(TAG, "camera: ${if (useFrontCamera) "FRONT (mirrored overlay)" else "BACK"}")
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // ImageProxy.toBitmap()
            .build()
            .also {
                it.setAnalyzer(
                    analysisExecutor,
                    CountingAnalyzer(det, tracker, counter, emptyReset) { r ->
                        overlay.submit(r)
                    },
                )
            }
        provider.unbindAll()
        provider.bindToLifecycle(this, selector, preview, analysis)
    }

    // Flip button: back <-> front. The unique total keeps counting (it is not reset); only the live
    // tracks are dropped -- boxes from the old camera must not match people on the new one, and
    // ids stay unique so UniqueCounter never sees an old id reissued.
    private fun onSwitchCameraClicked() {
        useFrontCamera = !useFrontCamera
        analysisExecutor.execute { tracker.dropTracks() } // same thread as analyze() -> no race
        bindCamera()
    }

    // Reset button handler -> start a fresh walk (mirrors PC 'r' hotkey).
    fun onResetClicked() {
        analysisExecutor.execute { // same thread as analyze() -> no race
            counter.reset()
            tracker.reset()
            emptyReset.reset()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private fun loadAsset(name: String): ByteBuffer {
        assets.openFd(name).use { fd ->
            fd.createInputStream().channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}

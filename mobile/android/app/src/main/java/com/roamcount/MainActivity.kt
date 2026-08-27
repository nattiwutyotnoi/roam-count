package com.roamcount

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

// EXAMPLE ONLY / NOT COMPILED.
//
// Wires CameraX Preview + ImageAnalysis(CountingAnalyzer) and shows the overlay.
// Loads config-equivalent thresholds; keeps the SAME rules as PC:
//   * no frame written to disk (privacy)   * ReID/embeddings (Option B) in RAM only
//   * a Reset button maps to counter.reset() + tracker.reset() (the PC 'r' hotkey)

class MainActivity : ComponentActivity() {

    private lateinit var overlay: OverlayView
    private val counter = UniqueCounter(minTrackAgeFrames = 5, maxHistory = 10_000)
    private val tracker = ByteTrackLite(trackBuffer = 30)

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // TODO inflate a layout with a PreviewView + OverlayView (+ a Reset button).
        // setContentView(R.layout.activity_main); overlay = findViewById(R.id.overlay); ...
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun buildDetector(): PersonDetector {
        val model = loadAsset("yolo11n.tflite") // put the exported model in app/src/main/assets/
        val options = Interpreter.Options().apply {
            // Try GPU delegate; fall back to NNAPI or CPU threads if unsupported.
            try { addDelegate(GpuDelegate()) } catch (_: Throwable) { setNumThreads(4) }
        }
        return PersonDetector(model, options, inputSize = 640, confThresh = 0.35f, iouThresh = 0.5f)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val previewView = PreviewView(this) // TODO add to layout
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val detector = buildDetector()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // simplifies toBitmap()
                .build()
                .also {
                    it.setAnalyzer(
                        ContextCompat.getMainExecutor(this),
                        CountingAnalyzer(detector, tracker, counter) { r ->
                            overlay.submit(r)
                        },
                    )
                }
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis,
            )
        }, ContextCompat.getMainExecutor(this))
    }

    // Reset button handler -> start a fresh walk (mirrors PC 'r' hotkey).
    fun onResetClicked() {
        counter.reset()
        tracker.reset()
    }

    private fun loadAsset(name: String): ByteBuffer {
        assets.openFd(name).use { fd ->
            fd.createInputStream().channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}

package com.roamcount

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

// EXAMPLE ONLY / NOT COMPILED.

/**
 * CameraX ImageAnalysis pipeline: frame -> detector -> tracker -> UniqueCounter -> UI.
 * Runs on CameraX's analysis executor (a background thread), so keep it allocation-light.
 * Privacy: frames stay in memory only -- never written to disk (plan section 7).
 */
class CountingAnalyzer(
    private val detector: PersonDetector,
    private val tracker: ByteTrackLite,
    private val counter: UniqueCounter,
    private val onResult: (Result) -> Unit,
) : ImageAnalysis.Analyzer {

    data class Result(
        val persons: List<TrackedPerson>,
        val uniqueTotal: Int,
        val occupancy: Int,
        val fps: Float,
        val frameW: Int,
        val frameH: Int,
    )

    private var lastNanos = System.nanoTime()

    override fun analyze(image: ImageProxy) {
        try {
            val bmp = image.toRgbBitmap() // TODO: YUV_420_888 -> RGB (see README, use RenderScript/YuvImage/toBitmap())
            val dets = detector.detect(bmp)
            val persons = tracker.update(dets)
            counter.update(persons.map { it.trackId })

            val now = System.nanoTime()
            val fps = 1e9f / (now - lastNanos).coerceAtLeast(1)
            lastNanos = now

            onResult(Result(persons, counter.uniqueTotal, persons.size, fps, bmp.width, bmp.height))
        } finally {
            image.close() // MUST close every frame or the camera pipeline stalls
        }
    }
}

/** TODO implement YUV_420_888 -> Bitmap. On CameraX you can enable
 *  OUTPUT_IMAGE_FORMAT_RGBA_8888 on ImageAnalysis and use ImageProxy.toBitmap()
 *  (androidx.camera 1.3+), which is the simplest correct path. */
private fun ImageProxy.toRgbBitmap(): Bitmap =
    throw NotImplementedError("Convert ImageProxy (YUV/RGBA) to Bitmap -- see README")

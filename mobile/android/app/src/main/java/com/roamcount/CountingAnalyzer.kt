package com.roamcount

import android.graphics.Bitmap
import android.graphics.Matrix
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
    private val emptyReset: EmptyFrameReset = EmptyFrameReset(0), // 0 = never auto-reset
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
            val bmp = image.toRgbBitmap()
            val dets = detector.detect(bmp)
            val persons = tracker.update(dets)
            counter.update(persons.map { it.trackId })

            val now = System.nanoTime()
            // Nobody in frame for the grace period -> the walk is over: start again from 0.
            if (emptyReset.update(persons.size, now)) {
                counter.reset()
                tracker.reset()
            }
            val fps = 1e9f / (now - lastNanos).coerceAtLeast(1)
            lastNanos = now

            onResult(Result(persons, counter.uniqueTotal, persons.size, fps, bmp.width, bmp.height))
        } finally {
            image.close() // MUST close every frame or the camera pipeline stalls
        }
    }
}

/**
 * ImageAnalysis is configured with OUTPUT_IMAGE_FORMAT_RGBA_8888, so toBitmap() (camera-core 1.3+)
 * is a straight copy. The bitmap is in sensor orientation -> rotate upright for the detector.
 * Held in memory only; never written to disk.
 */
private fun ImageProxy.toRgbBitmap(): Bitmap {
    val bmp = toBitmap()
    val rot = imageInfo.rotationDegrees
    if (rot == 0) return bmp
    val m = Matrix().apply { postRotate(rot.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
}

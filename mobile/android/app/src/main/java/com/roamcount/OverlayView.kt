package com.roamcount

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

// EXAMPLE ONLY / NOT COMPILED.

/** Draws boxes + the unique-total HUD over the camera preview (mirrors main.py's overlay). */
class OverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    @Volatile private var persons: List<TrackedPerson> = emptyList()
    @Volatile private var nonPersons: List<Detection> = emptyList()
    @Volatile private var uniqueTotal = 0
    @Volatile private var occupancy = 0
    @Volatile private var fps = 0f
    @Volatile private var scaleX = 1f
    @Volatile private var scaleY = 1f
    /** The front camera's preview is mirrored by PreviewView, but analysis frames are not: flip x. */
    @Volatile private var mirrorX = false

    private val boxPaint = Paint().apply {
        color = Color.rgb(0, 220, 0); style = Paint.Style.STROKE; strokeWidth = 4f
    }
    // Third colour: a PERSON the detector is unsure about (conf < ConfidenceBand.LOW_CONF_THRESHOLD).
    private val lowBoxPaint = Paint().apply {
        color = Color.rgb(255, 165, 0); style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val lowTextPaint = Paint().apply {
        color = Color.rgb(255, 190, 60); textSize = 42f; isAntiAlias = true
    }
    private val otherBoxPaint = Paint().apply {
        color = Color.rgb(230, 0, 0); style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val otherTextPaint = Paint().apply {
        color = Color.rgb(255, 80, 80); textSize = 36f; isAntiAlias = true
    }
    private val bigPaint = Paint().apply {
        color = Color.rgb(0, 220, 0); textSize = 96f; isAntiAlias = true
    }
    private val hudPaint = Paint().apply {
        color = Color.WHITE; textSize = 42f; isAntiAlias = true
    }

    fun setMirrored(mirrored: Boolean) {
        mirrorX = mirrored
        postInvalidate()
    }

    // Frame x -> view x, flipped across the view when mirrored (left edge of a box = its mirrored right edge).
    private fun left(x1: Float, x2: Float) = if (mirrorX) width - x2 * scaleX else x1 * scaleX
    private fun right(x1: Float, x2: Float) = if (mirrorX) width - x1 * scaleX else x2 * scaleX

    /** Called from the analyzer's onResult (post to main thread via postInvalidate). */
    fun submit(r: CountingAnalyzer.Result) {
        persons = r.persons
        nonPersons = r.nonPersons
        uniqueTotal = r.uniqueTotal
        occupancy = r.occupancy
        fps = r.fps
        if (r.frameW > 0 && r.frameH > 0) {
            scaleX = width.toFloat() / r.frameW
            scaleY = height.toFloat() / r.frameH
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Not a person -> red (display only). Person tracks -> green, below.
        for (d in nonPersons) {
            canvas.drawRect(left(d.x1, d.x2), d.y1 * scaleY, right(d.x1, d.x2), d.y2 * scaleY, otherBoxPaint)
            canvas.drawText(
                "${CocoLabels.name(d.cls)} ${"%.2f".format(d.conf)}",
                left(d.x1, d.x2), d.y1 * scaleY - 8f, otherTextPaint,
            )
        }
        for (p in persons) {
            val low = ConfidenceBand.isLow(p.conf)
            canvas.drawRect(left(p.x1, p.x2), p.y1 * scaleY, right(p.x1, p.x2), p.y2 * scaleY, if (low) lowBoxPaint else boxPaint)
            val label = if (low) "id ${p.trackId} ${"%.2f".format(p.conf)}" else "id ${p.trackId}"
            canvas.drawText(label, left(p.x1, p.x2), p.y1 * scaleY - 8f, if (low) lowTextPaint else hudPaint)
        }
        canvas.drawText("$uniqueTotal", 24f, 110f, bigPaint)
        canvas.drawText("UNIQUE TOTAL", 24f, 150f, hudPaint)
        canvas.drawText("Occupancy: $occupancy", 24f, 200f, hudPaint)
        canvas.drawText("FPS: ${"%.1f".format(fps)}", 24f, 246f, hudPaint)
    }
}

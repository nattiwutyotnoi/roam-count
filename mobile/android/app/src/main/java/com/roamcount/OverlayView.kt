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

    private val boxPaint = Paint().apply {
        color = Color.rgb(0, 220, 0); style = Paint.Style.STROKE; strokeWidth = 4f
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
            canvas.drawRect(d.x1 * scaleX, d.y1 * scaleY, d.x2 * scaleX, d.y2 * scaleY, otherBoxPaint)
            canvas.drawText(
                "${CocoLabels.name(d.cls)} ${"%.2f".format(d.conf)}",
                d.x1 * scaleX, d.y1 * scaleY - 8f, otherTextPaint,
            )
        }
        for (p in persons) {
            canvas.drawRect(p.x1 * scaleX, p.y1 * scaleY, p.x2 * scaleX, p.y2 * scaleY, boxPaint)
            canvas.drawText("id ${p.trackId}", p.x1 * scaleX, p.y1 * scaleY - 8f, hudPaint)
        }
        canvas.drawText("$uniqueTotal", 24f, 110f, bigPaint)
        canvas.drawText("UNIQUE TOTAL", 24f, 150f, hudPaint)
        canvas.drawText("Occupancy: $occupancy", 24f, 200f, hudPaint)
        canvas.drawText("FPS: ${"%.1f".format(fps)}", 24f, 246f, hudPaint)
    }
}

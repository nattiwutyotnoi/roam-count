package com.roamcount

import java.nio.ByteBuffer

/**
 * Pure (no Android / TFLite deps) tensor <-> box logic for the YOLO11n person detector.
 * Kept separate from PersonDetector so it can be unit-tested on the JVM -- see
 * YoloDecoderTest.kt. Every function here exists to prevent one of the silent failures
 * seen when porting (a wrong layout/scale never throws; it just returns garbage boxes):
 *
 *   1) INPUT  is NCHW  [1,3,640,640] for the ultralytics 8.4 export  (not NHWC)
 *   2) OUTPUT is attribute-major [1,84,8400]                         (not [1,8400,84])
 *   3) BOXES  are normalized [0,1]                                   (not input pixels)
 *
 * Layouts are detected from the tensor shapes at load time (PersonDetector) instead of
 * being assumed, so a re-export with a different layout still decodes correctly.
 */
object YoloDecoder {

    enum class OutputLayout {
        /** [1, attrs, anchors] e.g. [1,84,8400] -- value of attr a for anchor i is raw[a*anchors + i]. */
        ATTR_MAJOR,

        /** [1, anchors, attrs] e.g. [1,8400,84] -- value of attr a for anchor i is raw[i*attrs + a]. */
        ANCHOR_MAJOR,
    }

    enum class TensorType { FLOAT32, INT8, UINT8 }

    /** attrs (4 box + classes, 84 for COCO) is always far smaller than anchors (8400). */
    fun outputLayout(dim1: Int, dim2: Int): OutputLayout =
        if (dim1 < dim2) OutputLayout.ATTR_MAJOR else OutputLayout.ANCHOR_MAJOR

    // --- quantization -------------------------------------------------------------

    fun quantize(v: Float, scale: Float, zeroPoint: Int, type: TensorType): Int {
        val q = Math.round(v / scale) + zeroPoint
        return if (type == TensorType.INT8) q.coerceIn(-128, 127) else q.coerceIn(0, 255)
    }

    fun dequantize(q: Int, scale: Float, zeroPoint: Int): Float = (q - zeroPoint) * scale

    // --- input --------------------------------------------------------------------

    /**
     * Write ARGB pixels (size x size, already letterboxed) into [out] as RGB scaled to [0,1],
     * in the order the model expects. [nchw]=true -> all R, then all G, then all B;
     * false -> interleaved RGBRGB... (NHWC). INT8/UINT8 models get quantized values.
     */
    fun writeInput(
        px: IntArray,
        size: Int,
        nchw: Boolean,
        type: TensorType,
        scale: Float,
        zeroPoint: Int,
        out: ByteBuffer,
    ) {
        out.clear()
        val n = size * size
        if (nchw) {
            for (ch in 0 until 3) {
                for (i in 0 until n) putChannel(out, channel(px[i], ch), type, scale, zeroPoint)
            }
        } else {
            for (i in 0 until n) {
                for (ch in 0 until 3) putChannel(out, channel(px[i], ch), type, scale, zeroPoint)
            }
        }
        out.rewind()
    }

    private fun channel(argb: Int, ch: Int): Int = when (ch) {
        0 -> (argb shr 16) and 0xFF // R
        1 -> (argb shr 8) and 0xFF  // G
        else -> argb and 0xFF       // B
    }

    private fun putChannel(out: ByteBuffer, v255: Int, type: TensorType, scale: Float, zeroPoint: Int) {
        val v = v255 / 255f
        when (type) {
            TensorType.FLOAT32 -> out.putFloat(v)
            else -> out.put(quantize(v, scale, zeroPoint, type).toByte())
        }
    }

    // --- output -------------------------------------------------------------------

    /** Copy the raw output tensor into [out] as dequantized floats. [buf] must use native byte order. */
    fun readOutput(buf: ByteBuffer, type: TensorType, scale: Float, zeroPoint: Int, out: FloatArray) {
        buf.rewind()
        when (type) {
            TensorType.FLOAT32 -> buf.asFloatBuffer().get(out)
            TensorType.INT8 -> for (i in out.indices) out[i] = dequantize(buf.get().toInt(), scale, zeroPoint)
            TensorType.UINT8 -> for (i in out.indices) out[i] = dequantize(buf.get().toInt() and 0xFF, scale, zeroPoint)
        }
    }

    /**
     * Decode raw YOLO output to person boxes in MODEL-INPUT pixel space (letterboxed 640x640).
     * No NMS here. A box is kept only if the person score >= [confThresh] AND person is the
     * best class for that anchor (same rule as ultralytics: argmax class, then class filter).
     *
     * @param boxScale inputSize when the model emits normalized [0,1] coords, 1f when it emits pixels.
     */
    fun decode(
        raw: FloatArray,
        anchors: Int,
        attrs: Int,
        layout: OutputLayout,
        confThresh: Float,
        personClass: Int,
        boxScale: Float,
    ): List<Detection> {
        require(raw.size == anchors * attrs) { "raw size ${raw.size} != anchors*attrs ${anchors * attrs}" }
        val numCls = attrs - 4
        require(personClass in 0 until numCls) { "personClass $personClass outside 0..${numCls - 1}" }

        fun at(attr: Int, a: Int): Float =
            if (layout == OutputLayout.ATTR_MAJOR) raw[attr * anchors + a] else raw[a * attrs + attr]

        val dets = ArrayList<Detection>()
        for (a in 0 until anchors) {
            val score = at(4 + personClass, a)
            if (score < confThresh) continue // cheap early-out: covers ~all of the 8400 anchors

            var personIsBest = true
            for (c in 0 until numCls) {
                if (c != personClass && at(4 + c, a) > score) {
                    personIsBest = false
                    break
                }
            }
            if (!personIsBest) continue

            val cx = at(0, a) * boxScale
            val cy = at(1, a) * boxScale
            val w = at(2, a) * boxScale
            val h = at(3, a) * boxScale
            dets.add(Detection(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, score, personClass))
        }
        return dets
    }

    /**
     * Candidates whose best class is NOT [personClass] -- for DISPLAY only (red boxes); they are
     * never tracked or counted. Mirror image of [decode]: an anchor belongs to exactly one of
     * the two (ties go to the person), so a person is never reported here.
     *
     * Unlike [decode] there is no cheap early-out (the best of ~79 classes is needed), so scan
     * class by class: with ATTR_MAJOR that keeps every read contiguous.
     */
    fun decodeNonPerson(
        raw: FloatArray,
        anchors: Int,
        attrs: Int,
        layout: OutputLayout,
        confThresh: Float,
        personClass: Int,
        boxScale: Float,
    ): List<Detection> {
        require(raw.size == anchors * attrs) { "raw size ${raw.size} != anchors*attrs ${anchors * attrs}" }
        val numCls = attrs - 4
        require(personClass in 0 until numCls) { "personClass $personClass outside 0..${numCls - 1}" }

        fun at(attr: Int, a: Int): Float =
            if (layout == OutputLayout.ATTR_MAJOR) raw[attr * anchors + a] else raw[a * attrs + attr]

        val bestScore = FloatArray(anchors)
        val bestCls = IntArray(anchors) { -1 }
        for (c in 0 until numCls) {
            if (c == personClass) continue
            for (a in 0 until anchors) {
                val s = at(4 + c, a)
                if (s > bestScore[a]) { bestScore[a] = s; bestCls[a] = c }
            }
        }

        val dets = ArrayList<Detection>()
        for (a in 0 until anchors) {
            val s = bestScore[a]
            if (bestCls[a] < 0 || s < confThresh) continue
            if (at(4 + personClass, a) >= s) continue // person is best (ties -> person)
            val cx = at(0, a) * boxScale
            val cy = at(1, a) * boxScale
            val w = at(2, a) * boxScale
            val h = at(3, a) * boxScale
            dets.add(Detection(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, s, bestCls[a]))
        }
        return dets
    }

    /** Undo the letterbox (model-input pixels -> original-frame pixels) and clip to the frame. */
    fun toFrame(d: Detection, scale: Float, padX: Float, padY: Float, frameW: Int, frameH: Int): Detection =
        Detection(
            ((d.x1 - padX) / scale).coerceIn(0f, frameW.toFloat()),
            ((d.y1 - padY) / scale).coerceIn(0f, frameH.toFloat()),
            ((d.x2 - padX) / scale).coerceIn(0f, frameW.toFloat()),
            ((d.y2 - padY) / scale).coerceIn(0f, frameH.toFloat()),
            d.conf,
            d.cls,
        )
}

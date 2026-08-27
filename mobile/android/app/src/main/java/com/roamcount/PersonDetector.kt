package com.roamcount

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

// EXAMPLE ONLY / NOT COMPILED.
//
// YOLO11n person detector via TFLite. Only class 0 (person) is kept. Thresholds are
// passed in (mirror config.json on PC), never hard-coded elsewhere.
//
// ⚠️ TWO THINGS YOU MUST VERIFY against YOUR exported .tflite (open it in netron.app):
//
//   1) OUTPUT LAYOUT. ONNX gives (1, 84, 8400). The TFLite converter (onnx2tf) usually
//      TRANSPOSES to (1, 8400, 84) -> this code assumes that: each of 8400 rows is
//      [cx, cy, w, h, clsScore0 .. clsScore79] (YOLO11 has no separate objectness).
//      If your shape differs, fix OUT_ANCHORS / OUT_ATTRS and the indexing.
//
//   2) COORDINATE SCALE + QUANTIZATION.
//      - Some exports give box coords already normalized to [0,1]; others in input
//        pixels [0,640]. This code assumes INPUT PIXELS. If normalized, multiply
//        cx,cy,w,h by inputSize first. Verify on a known image.
//      - An INT8 model has quantized input & output (scale, zeroPoint from
//        interpreter.getInputTensor(0).quantizationParams()). Dequantize the output
//        and quantize the input. This skeleton shows the FLOAT path only.

class PersonDetector(
    modelBuffer: ByteBuffer,
    options: Interpreter.Options,
    private val inputSize: Int = 640,
    private val confThresh: Float = 0.35f,
    private val iouThresh: Float = 0.5f,
    private val personClass: Int = 0,
) {
    private val interpreter = Interpreter(modelBuffer, options)

    // TODO verify against interpreter.getOutputTensor(0).shape()
    private val outAnchors = 8400
    private val outAttrs = 84 // 4 bbox + 80 classes

    fun detect(frame: Bitmap): List<Detection> {
        val lb = letterbox(frame, inputSize)
        val input = toFloatInput(lb.bitmap) // [1,640,640,3], /255f  (float path)

        val output = Array(1) { Array(outAnchors) { FloatArray(outAttrs) } }
        interpreter.run(input, output)

        val dets = ArrayList<Detection>()
        for (row in output[0]) {
            var bestCls = -1
            var bestScore = 0f
            for (c in 0 until (outAttrs - 4)) {
                val s = row[4 + c]
                if (s > bestScore) {
                    bestScore = s
                    bestCls = c
                }
            }
            if (bestCls != personClass || bestScore < confThresh) continue

            // assumes input-pixel coords; if normalized, multiply by inputSize here
            val cx = row[0]
            val cy = row[1]
            val w = row[2]
            val h = row[3]
            // undo letterbox -> original-frame pixels
            val x1 = (cx - w / 2f - lb.padX) / lb.scale
            val y1 = (cy - h / 2f - lb.padY) / lb.scale
            val x2 = (cx + w / 2f - lb.padX) / lb.scale
            val y2 = (cy + h / 2f - lb.padY) / lb.scale
            dets.add(Detection(x1, y1, x2, y2, bestScore, bestCls))
        }
        return Nms.apply(dets, iouThresh)
    }

    fun close() = interpreter.close()

    // --- helpers (EXAMPLE ONLY) ------------------------------------------------
    private class Letterboxed(val bitmap: Bitmap, val scale: Float, val padX: Float, val padY: Float)

    /** Resize keeping aspect ratio, center-pad to a square inputSize (like YOLO's letterbox). */
    private fun letterbox(src: Bitmap, size: Int): Letterboxed {
        val scale = minOf(size.toFloat() / src.width, size.toFloat() / src.height)
        val newW = Math.round(src.width * scale)
        val newH = Math.round(src.height * scale)
        val padX = (size - newW) / 2f
        val padY = (size - newH) / 2f
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(114, 114, 114)) // YOLO's grey pad
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        canvas.drawBitmap(resized, padX, padY, null)
        return Letterboxed(out, scale, padX, padY)
    }

    /** Bitmap -> NHWC float32 buffer normalized to [0,1]. (int8: quantize instead.) */
    private fun toFloatInput(bmp: Bitmap): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
        val px = IntArray(inputSize * inputSize)
        bmp.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        for (p in px) {
            buf.putFloat(((p shr 16) and 0xFF) / 255f) // R
            buf.putFloat(((p shr 8) and 0xFF) / 255f)  // G
            buf.putFloat((p and 0xFF) / 255f)          // B
        }
        buf.rewind()
        return buf
    }
}

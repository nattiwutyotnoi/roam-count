package com.roamcount

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

// NOT COMPILED against the Android SDK (no toolchain in the dev environment) -- the pure
// tensor/box logic lives in YoloDecoder.kt and IS unit-tested on the JVM (YoloDecoderTest.kt).
//
// YOLO11n person detector via TFLite (ultralytics 8.4 export). Only class 0 (person) is kept.
// Thresholds are passed in (mirror config.json on PC), never hard-coded elsewhere.
//
// Layout facts for the ultralytics 8.4 export (what the previous version of this file got
// wrong -- all three fail SILENTLY, no exception, just garbage boxes):
//   * input  NCHW  [1,3,640,640]
//   * output       [1,84,8400]  = attribute-major: rows cx,cy,w,h,cls0..cls79 ; columns = anchors
//   * boxes        normalized [0,1]   (so multiplied by inputSize before the letterbox is undone)
//
// Rather than trust those facts, the constructor READS the tensor shapes/types from the
// loaded model and fails loudly if they are not what the decode code can handle. Input
// layout (NCHW/NHWC), output layout and fp32/int8/uint8 are all detected from the model.
// Box scale is the one thing a shape cannot reveal -> `boxesNormalized` (default true);
// verify on a known image (bus.jpg, expect 4 persons, boxes within ~3 px of the .pt model).

class PersonDetector(
    modelBuffer: ByteBuffer,
    options: Interpreter.Options,
    private val inputSize: Int = 640,
    private val confThresh: Float = 0.35f,
    private val iouThresh: Float = 0.5f,
    private val personClass: Int = 0,
    private val boxesNormalized: Boolean = true,
    /** Also report non-person objects (drawn red, never tracked/counted). false = skip the extra scan. */
    private val detectNonPerson: Boolean = true,
    private val nonPersonConfThresh: Float = 0.5f,
) {
    private val interpreter = Interpreter(modelBuffer, options)
    private val nativeOrder = ByteOrder.nativeOrder()

    // --- input tensor ---------------------------------------------------------
    private val inTensor = interpreter.getInputTensor(0)
    private val inType = typeOf(inTensor.dataType())
    private val inScale = inTensor.quantizationParams().scale
    private val inZero = inTensor.quantizationParams().zeroPoint
    private val inputIsNchw = resolveInputLayout(inTensor.shape(), inputSize)

    // --- output tensor --------------------------------------------------------
    private val outTensor = interpreter.getOutputTensor(0)
    private val outType = typeOf(outTensor.dataType())
    private val outScale = outTensor.quantizationParams().scale
    private val outZero = outTensor.quantizationParams().zeroPoint
    private val outLayout: YoloDecoder.OutputLayout
    private val outAnchors: Int
    private val outAttrs: Int

    // --- reusable buffers (no per-frame allocation on the hot path) ------------
    private val inputBuf: ByteBuffer
    private val outputBuf: ByteBuffer
    private val raw: FloatArray
    private val pixels = IntArray(inputSize * inputSize)
    private val lbBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val lbCanvas = Canvas(lbBitmap)
    private val lbPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val os = outTensor.shape()
        check(os.size == 3 && os[0] == 1) { "Unexpected output shape ${os.toList()}; expected [1,84,8400] or [1,8400,84]" }
        outLayout = YoloDecoder.outputLayout(os[1], os[2])
        outAttrs = minOf(os[1], os[2])
        outAnchors = maxOf(os[1], os[2])
        check(outAttrs >= 5) { "Output has only $outAttrs attributes; expected 4 box + class scores" }
        check(personClass in 0 until outAttrs - 4) { "personClass $personClass out of range for ${outAttrs - 4} classes" }

        inputBuf = ByteBuffer.allocateDirect(inTensor.numBytes()).order(nativeOrder)
        check(inputBuf.capacity() == inputSize * inputSize * 3 * bytesPer(inType)) {
            "Input tensor is ${inputBuf.capacity()} bytes; ${inputSize}x$inputSize RGB ($inType) needs " +
                "${inputSize * inputSize * 3 * bytesPer(inType)}. Wrong inputSize?"
        }
        outputBuf = ByteBuffer.allocateDirect(outTensor.numBytes()).order(nativeOrder)
        raw = FloatArray(outAnchors * outAttrs)
    }

    /** One-line summary of what was detected from the model -- log it once at startup. */
    fun describe(): String =
        "in=${inTensor.shape().toList()} ${if (inputIsNchw) "NCHW" else "NHWC"} $inType | " +
            "out=${outTensor.shape().toList()} $outLayout $outType | boxesNormalized=$boxesNormalized"

    /**
     * Non-person objects found by the last [detect] call (frame pixels), for the red boxes.
     * Written and read on the analysis thread only. Not tracked, not counted.
     */
    var lastNonPersons: List<Detection> = emptyList()
        private set

    fun detect(frame: Bitmap): List<Detection> {
        val lb = letterbox(frame)
        lbBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        YoloDecoder.writeInput(pixels, inputSize, inputIsNchw, inType, inScale, inZero, inputBuf)

        outputBuf.rewind()
        interpreter.run(inputBuf, outputBuf)
        YoloDecoder.readOutput(outputBuf, outType, outScale, outZero, raw)

        val boxScale = if (boxesNormalized) inputSize.toFloat() else 1f
        val candidates = YoloDecoder.decode(raw, outAnchors, outAttrs, outLayout, confThresh, personClass, boxScale)
        // NMS in model-input space (like ultralytics), then map the survivors to frame pixels.
        val persons = Nms.apply(candidates, iouThresh)

        lastNonPersons = if (!detectNonPerson) emptyList() else {
            val others = YoloDecoder.decodeNonPerson(
                raw, outAnchors, outAttrs, outLayout, nonPersonConfThresh, personClass, boxScale,
            )
            // Same object labelled twice (e.g. a person also scoring as "chair"): the person wins.
            Nms.dropOverlapping(Nms.apply(others, iouThresh), persons, SAME_OBJECT_IOU).map {
                YoloDecoder.toFrame(it, lb.scale, lb.padX, lb.padY, frame.width, frame.height)
            }
        }

        return persons.map {
            YoloDecoder.toFrame(it, lb.scale, lb.padX, lb.padY, frame.width, frame.height)
        }
    }

    fun close() = interpreter.close()

    // --- helpers -----------------------------------------------------------------
    private class Letterbox(val scale: Float, val padX: Float, val padY: Float)

    /** Resize keeping aspect ratio and center-pad to a square (YOLO's grey 114 letterbox) into lbBitmap. */
    private fun letterbox(src: Bitmap): Letterbox {
        val scale = minOf(inputSize.toFloat() / src.width, inputSize.toFloat() / src.height)
        val newW = src.width * scale
        val newH = src.height * scale
        val padX = (inputSize - newW) / 2f
        val padY = (inputSize - newH) / 2f
        lbCanvas.drawColor(Color.rgb(114, 114, 114))
        lbCanvas.drawBitmap(src, null, RectF(padX, padY, padX + newW, padY + newH), lbPaint)
        return Letterbox(scale, padX, padY)
    }

    private companion object {
        const val SAME_OBJECT_IOU = 0.7f

        fun typeOf(t: DataType): YoloDecoder.TensorType = when (t) {
            DataType.FLOAT32 -> YoloDecoder.TensorType.FLOAT32
            DataType.INT8 -> YoloDecoder.TensorType.INT8
            DataType.UINT8 -> YoloDecoder.TensorType.UINT8
            else -> error("Unsupported tensor type $t (expected FLOAT32 / INT8 / UINT8)")
        }

        fun bytesPer(t: YoloDecoder.TensorType) = if (t == YoloDecoder.TensorType.FLOAT32) 4 else 1

        /** true = NCHW [1,3,H,W], false = NHWC [1,H,W,3]; anything else is an error, never a guess. */
        fun resolveInputLayout(shape: IntArray, size: Int): Boolean = when {
            shape.size == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] == size && shape[3] == size -> true
            shape.size == 4 && shape[0] == 1 && shape[3] == 3 && shape[1] == size && shape[2] == size -> false
            else -> error("Unexpected input shape ${shape.toList()}; expected [1,3,$size,$size] or [1,$size,$size,3]")
        }
    }
}

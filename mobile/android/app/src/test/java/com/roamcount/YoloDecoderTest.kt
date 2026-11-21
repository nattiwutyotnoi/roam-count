package com.roamcount

import com.roamcount.YoloDecoder.OutputLayout
import com.roamcount.YoloDecoder.TensorType
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * JVM tests for YoloDecoder -- one test per silent failure found while porting
 * (input NCHW, output [1,84,8400], normalized boxes) plus letterbox / quantization.
 *
 * Standalone, no JUnit needed (own entry point so it does not clash with UniqueCounterTest's main()):
 *   kotlinc Detection.kt YoloDecoder.kt YoloDecoderTest.kt -include-runtime -d t.jar
 *   java -cp t.jar com.roamcount.YoloDecoderTest
 *
 * These prove the DECODE MATH on synthetic tensors. They do NOT prove the real .tflite emits
 * these layouts -- confirm that with PersonDetector.describe() and bus.jpg on a device.
 */
object YoloDecoderTest {

    private const val ANCHORS = 8400
    private const val ATTRS = 84

    private fun check(cond: Boolean, msg: String) {
        if (!cond) throw AssertionError(msg)
    }

    private fun near(a: Float, b: Float, tol: Float = 1e-3f) = Math.abs(a - b) <= tol

    /** Fill one anchor: box (cx,cy,w,h in whatever unit the test uses) + class scores. */
    private class Tensor(val layout: OutputLayout) {
        val raw = FloatArray(ANCHORS * ATTRS)
        fun set(attr: Int, anchor: Int, v: Float) {
            if (layout == OutputLayout.ATTR_MAJOR) raw[attr * ANCHORS + anchor] = v
            else raw[anchor * ATTRS + attr] = v
        }
        fun anchor(a: Int, cx: Float, cy: Float, w: Float, h: Float, cls: Int, score: Float) {
            set(0, a, cx); set(1, a, cy); set(2, a, w); set(3, a, h); set(4 + cls, a, score)
        }
    }

    private fun decode(t: Tensor, boxScale: Float, conf: Float = 0.35f) =
        YoloDecoder.decode(t.raw, ANCHORS, ATTRS, t.layout, conf, 0, boxScale)

    // 1) layout detection ------------------------------------------------------------------
    private fun testLayoutDetectedFromShape() {
        check(YoloDecoder.outputLayout(84, 8400) == OutputLayout.ATTR_MAJOR, "[1,84,8400] -> attr-major")
        check(YoloDecoder.outputLayout(8400, 84) == OutputLayout.ANCHOR_MAJOR, "[1,8400,84] -> anchor-major")
    }

    // 2) output [1,84,8400] + normalized boxes -- the exact failing case ---------------------
    private fun testAttrMajorNormalizedBox() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(100, cx = 0.5f, cy = 0.5f, w = 0.25f, h = 0.5f, cls = 0, score = 0.9f)
        val d = decode(t, boxScale = 640f)
        check(d.size == 1, "one box, got ${d.size}")
        // cx=320 cy=320 w=160 h=320 -> (240,160)-(400,480)
        check(near(d[0].x1, 240f) && near(d[0].y1, 160f) && near(d[0].x2, 400f) && near(d[0].y2, 480f), "box = $d")
        check(near(d[0].conf, 0.9f), "conf")
    }

    private fun testAnchorMajorGivesSameBox() {
        val t = Tensor(OutputLayout.ANCHOR_MAJOR)
        t.anchor(100, 0.5f, 0.5f, 0.25f, 0.5f, 0, 0.9f)
        val d = decode(t, 640f)
        check(d.size == 1 && near(d[0].x1, 240f) && near(d[0].y2, 480f), "anchor-major box = $d")
    }

    // 3) pixel-coordinate models still work when boxesNormalized=false ----------------------
    private fun testPixelCoordsWhenBoxScaleIsOne() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(7, 320f, 320f, 160f, 320f, 0, 0.8f)
        val d = decode(t, boxScale = 1f)
        check(d.size == 1 && near(d[0].x1, 240f) && near(d[0].y1, 160f), "pixel box = $d")
    }

    // 4) filtering ------------------------------------------------------------------------
    private fun testBelowConfDropped() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(5, 0.5f, 0.5f, 0.2f, 0.2f, 0, 0.34f)
        check(decode(t, 640f).isEmpty(), "0.34 < 0.35 must be dropped")
    }

    private fun testOtherClassBestIsDropped() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(5, 0.5f, 0.5f, 0.2f, 0.2f, 0, 0.60f)
        t.set(4 + 2, 5, 0.95f) // class 2 (car) beats person on the same anchor
        check(decode(t, 640f).isEmpty(), "person not best class -> dropped")
    }

    private fun testOnlyPersonClassKept() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(1, 0.3f, 0.3f, 0.1f, 0.2f, 0, 0.7f)  // person
        t.anchor(2, 0.6f, 0.6f, 0.1f, 0.2f, 16, 0.9f) // dog
        val d = decode(t, 640f)
        check(d.size == 1 && d[0].cls == 0, "only the person survives: $d")
    }

    // 5) letterbox undo --------------------------------------------------------------------
    private fun testLetterboxUndo() {
        // 1280x720 frame -> scale 0.5, content 640x360, padX=0, padY=140
        val box = Detection(240f, 160f, 400f, 480f, 0.9f, 0)
        val f = YoloDecoder.toFrame(box, scale = 0.5f, padX = 0f, padY = 140f, frameW = 1280, frameH = 720)
        check(near(f.x1, 480f) && near(f.y1, 40f) && near(f.x2, 800f) && near(f.y2, 680f), "frame box = $f")
    }

    private fun testBoxClippedToFrame() {
        val box = Detection(-20f, 100f, 700f, 700f, 0.9f, 0)
        val f = YoloDecoder.toFrame(box, 0.5f, 0f, 140f, 1280, 720)
        check(f.x1 == 0f && f.x2 == 1280f && f.y2 == 720f, "clipped: $f")
    }

    // 6) input tensor order: NCHW vs NHWC --------------------------------------------------
    private val px = intArrayOf(
        0xFF_FF_00_00.toInt(), // pixel 0: R=255 G=0   B=0
        0xFF_00_FF_00.toInt(), // pixel 1: R=0   G=255 B=0
        0xFF_00_00_FF.toInt(), // pixel 2: R=0   G=0   B=255
        0xFF_FF_FF_FF.toInt(), // pixel 3: white
    )

    private fun floats(nchw: Boolean): FloatArray {
        val buf = ByteBuffer.allocateDirect(4 * 4 * 3).order(ByteOrder.nativeOrder())
        YoloDecoder.writeInput(px, 2, nchw, TensorType.FLOAT32, 0f, 0, buf)
        return FloatArray(12).also { buf.asFloatBuffer().get(it) }
    }

    private fun testInputNchwOrder() {
        // R plane (4), then G plane (4), then B plane (4)
        val expect = floatArrayOf(1f, 0f, 0f, 1f, /*G*/ 0f, 1f, 0f, 1f, /*B*/ 0f, 0f, 1f, 1f)
        check(floats(true).contentEquals(expect), "NCHW = ${floats(true).toList()}")
    }

    private fun testInputNhwcOrder() {
        val expect = floatArrayOf(1f, 0f, 0f, /*p1*/ 0f, 1f, 0f, /*p2*/ 0f, 0f, 1f, /*p3*/ 1f, 1f, 1f)
        check(floats(false).contentEquals(expect), "NHWC = ${floats(false).toList()}")
    }

    // 7) int8 quantization -----------------------------------------------------------------
    private fun testQuantizeRoundTripAndClamp() {
        val s = 1f / 255f
        val q = YoloDecoder.quantize(0.5f, s, -128, TensorType.INT8)
        check(near(YoloDecoder.dequantize(q, s, -128), 0.5f, 0.005f), "round trip, q=$q")
        check(YoloDecoder.quantize(0f, s, -128, TensorType.INT8) == -128, "0.0 -> -128")
        check(YoloDecoder.quantize(1f, s, -128, TensorType.INT8) == 127, "1.0 -> 127")
        check(YoloDecoder.quantize(5f, s, -128, TensorType.INT8) == 127, "clamp high")
        check(YoloDecoder.quantize(1f, s, 0, TensorType.UINT8) == 255, "uint8 1.0 -> 255")
    }

    private fun testInt8InputAndOutputPath() {
        // 1 int8 input pixel round-trips through writeInput; output int8 dequantizes via readOutput.
        val s = 1f / 255f
        val inBuf = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
        YoloDecoder.writeInput(px, 2, true, TensorType.INT8, s, -128, inBuf)
        check(inBuf.get(0).toInt() == 127 && inBuf.get(1).toInt() == -128, "int8 NCHW first bytes")

        val outBuf = ByteBuffer.allocateDirect(2).order(ByteOrder.nativeOrder())
        outBuf.put(0.toByte()).put((-128).toByte())
        val out = FloatArray(2)
        YoloDecoder.readOutput(outBuf, TensorType.INT8, s, -128, out)
        check(near(out[0], 128f * s, 1e-4f) && near(out[1], 0f, 1e-4f), "dequant = ${out.toList()}")
    }

    // 8) guard: wrong-sized raw array must throw, not silently decode ---------------------
    private fun testSizeMismatchThrows() {
        var threw = false
        try {
            YoloDecoder.decode(FloatArray(100), ANCHORS, ATTRS, OutputLayout.ATTR_MAJOR, 0.35f, 0, 640f)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        check(threw, "size mismatch must throw")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val tests = listOf(
            ::testLayoutDetectedFromShape, ::testAttrMajorNormalizedBox, ::testAnchorMajorGivesSameBox,
            ::testPixelCoordsWhenBoxScaleIsOne, ::testBelowConfDropped, ::testOtherClassBestIsDropped,
            ::testOnlyPersonClassKept, ::testLetterboxUndo, ::testBoxClippedToFrame,
            ::testInputNchwOrder, ::testInputNhwcOrder, ::testQuantizeRoundTripAndClamp,
            ::testInt8InputAndOutputPath, ::testSizeMismatchThrows,
        )
        for (t in tests) { t(); println("PASS ${t.name}") }
        println("\n${tests.size} passed")
    }
}

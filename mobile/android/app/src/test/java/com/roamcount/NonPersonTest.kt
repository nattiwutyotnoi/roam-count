package com.roamcount

import com.roamcount.YoloDecoder.OutputLayout

/**
 * JVM tests for the red-box (non-person) path. The person path is covered by YoloDecoderTest;
 * the first two tests here prove it is untouched by the new code.
 *
 *   kotlinc main/java/com/roamcount/Detection.kt main/java/com/roamcount/YoloDecoder.kt \
 *     main/java/com/roamcount/Nms.kt main/java/com/roamcount/CocoLabels.kt \
 *     test/java/com/roamcount/NonPersonTest.kt -include-runtime -d t.jar
 *   java -cp t.jar com.roamcount.NonPersonTest
 */
object NonPersonTest {
    private const val ANCHORS = 8400
    private const val ATTRS = 84
    private const val PERSON = 0
    private const val CHAIR = 56
    private const val TV = 62

    private fun check(cond: Boolean, msg: String) { if (!cond) throw AssertionError(msg) }
    private fun near(a: Float, b: Float, tol: Float = 1e-3f) = Math.abs(a - b) <= tol

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

    private fun persons(t: Tensor) = YoloDecoder.decode(t.raw, ANCHORS, ATTRS, t.layout, 0.35f, PERSON, 640f)
    private fun others(t: Tensor, conf: Float = 0.5f) =
        YoloDecoder.decodeNonPerson(t.raw, ANCHORS, ATTRS, t.layout, conf, PERSON, 640f)

    private fun scene(layout: OutputLayout): Tensor {
        val t = Tensor(layout)
        t.anchor(10, 0.25f, 0.5f, 0.2f, 0.4f, PERSON, 0.9f)   // a person
        t.anchor(20, 0.75f, 0.5f, 0.2f, 0.2f, CHAIR, 0.8f)    // a chair
        t.anchor(30, 0.5f, 0.2f, 0.1f, 0.1f, TV, 0.3f)        // a weak tv (below 0.5)
        return t
    }

    private fun testPersonPathIgnoresOtherObjects() {
        val d = persons(scene(OutputLayout.ATTR_MAJOR))
        check(d.size == 1 && d[0].cls == PERSON && near(d[0].conf, 0.9f), "only the person: $d")
    }

    private fun testPersonAndOtherAreDisjoint() {
        val t = scene(OutputLayout.ATTR_MAJOR)
        val p = persons(t); val o = others(t)
        check(o.none { it.cls == PERSON }, "person leaked into non-person list: $o")
        check(p.size + o.size == 2, "person + chair, tv is below threshold: ${p.size} + ${o.size}")
    }

    private fun testChairReportedWithClassAndBox() {
        val o = others(scene(OutputLayout.ATTR_MAJOR))
        check(o.size == 1 && o[0].cls == CHAIR && near(o[0].conf, 0.8f), "chair: $o")
        // cx=480 cy=320 w=128 h=128 -> (416,256)-(544,384)
        check(near(o[0].x1, 416f) && near(o[0].y1, 256f) && near(o[0].x2, 544f) && near(o[0].y2, 384f), "box: ${o[0]}")
    }

    private fun testAnchorMajorSameResult() {
        val o = others(scene(OutputLayout.ANCHOR_MAJOR))
        check(o.size == 1 && o[0].cls == CHAIR && near(o[0].x1, 416f), "anchor-major: $o")
    }

    private fun testThresholdApplies() {
        val t = scene(OutputLayout.ATTR_MAJOR)
        check(others(t, conf = 0.25f).map { it.cls }.toSet() == setOf(CHAIR, TV), "lower threshold also shows tv")
        check(others(t, conf = 0.9f).isEmpty(), "higher threshold hides chair")
    }

    private fun testPersonWinsWhenItScoresHigher() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(5, 0.5f, 0.5f, 0.3f, 0.3f, PERSON, 0.9f)
        t.set(4 + CHAIR, 5, 0.6f)   // same anchor also scores as chair, lower than person
        check(others(t).isEmpty(), "person is best -> not red")
        check(persons(t).size == 1, "still a person")
    }

    private fun testOtherWinsWhenItScoresHigher() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(5, 0.5f, 0.5f, 0.3f, 0.3f, CHAIR, 0.8f)
        t.set(4 + PERSON, 5, 0.5f)  // person is also above its own threshold but loses to chair
        check(persons(t).isEmpty(), "chair is best -> not a person")
        check(others(t).size == 1, "red box")
    }

    private fun testTieGoesToPerson() {
        val t = Tensor(OutputLayout.ATTR_MAJOR)
        t.anchor(5, 0.5f, 0.5f, 0.3f, 0.3f, PERSON, 0.7f)
        t.set(4 + CHAIR, 5, 0.7f)
        check(persons(t).size == 1 && others(t).isEmpty(), "tie -> person, never both")
    }

    private fun testEmptyTensor() {
        check(others(Tensor(OutputLayout.ATTR_MAJOR)).isEmpty(), "all zeros -> nothing")
    }

    private fun testSizeMismatchThrows() {
        var threw = false
        try { YoloDecoder.decodeNonPerson(FloatArray(10), ANCHORS, ATTRS, OutputLayout.ATTR_MAJOR, 0.5f, 0, 1f) }
        catch (e: IllegalArgumentException) { threw = true }
        check(threw, "size mismatch must throw")
    }

    private fun testDropOverlapping() {
        val person = Detection(100f, 100f, 200f, 300f, 0.9f, 0)
        val dup = Detection(105f, 102f, 198f, 298f, 0.6f, CHAIR)     // same object, IoU ~0.9
        val far = Detection(400f, 100f, 500f, 200f, 0.7f, CHAIR)
        val r = Nms.dropOverlapping(listOf(dup, far), listOf(person), 0.7f)
        check(r == listOf(far), "duplicate dropped, distant kept: $r")
        check(Nms.dropOverlapping(listOf(far), emptyList(), 0.7f) == listOf(far), "nothing to compare against")
    }

    private fun testCocoLabels() {
        check(CocoLabels.names.size == 80, "80 classes, got ${CocoLabels.names.size}")
        check(CocoLabels.name(0) == "person" && CocoLabels.name(CHAIR) == "chair" && CocoLabels.name(TV) == "tv", "known ids")
        check(CocoLabels.name(79) == "toothbrush" && CocoLabels.name(99) == "cls 99", "last + fallback")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val tests = listOf(
            ::testPersonPathIgnoresOtherObjects, ::testPersonAndOtherAreDisjoint, ::testChairReportedWithClassAndBox,
            ::testAnchorMajorSameResult, ::testThresholdApplies, ::testPersonWinsWhenItScoresHigher,
            ::testOtherWinsWhenItScoresHigher, ::testTieGoesToPerson, ::testEmptyTensor,
            ::testSizeMismatchThrows, ::testDropOverlapping, ::testCocoLabels,
        )
        for (t in tests) { t(); println("PASS ${t.name}") }
        println("\n${tests.size} passed")
    }
}

package com.roamcount

/**
 *   kotlinc main/java/com/roamcount/ConfidenceBand.kt test/java/com/roamcount/ConfidenceBandTest.kt \
 *     -include-runtime -d t.jar && java -cp t.jar com.roamcount.ConfidenceBandTest
 */
object ConfidenceBandTest {
    private fun check(cond: Boolean, msg: String) { if (!cond) throw AssertionError(msg) }

    private fun testBoundary() {
        check(ConfidenceBand.isLow(0.499f), "0.499 is low")
        check(!ConfidenceBand.isLow(0.5f), "exactly 0.5 is NOT low")
        check(!ConfidenceBand.isLow(0.9f), "0.9 is normal")
    }

    private fun testDetectorFloorIsLow() {
        check(ConfidenceBand.isLow(0.35f), "the weakest person the detector reports (0.35) is low")
    }

    private fun testCustomThreshold() {
        check(ConfidenceBand.isLow(0.6f, threshold = 0.7f) && !ConfidenceBand.isLow(0.6f, threshold = 0.6f), "custom threshold")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val tests = listOf(::testBoundary, ::testDetectorFloorIsLow, ::testCustomThreshold)
        for (t in tests) { t(); println("PASS ${t.name}") }
        println("\n${tests.size} passed")
    }
}

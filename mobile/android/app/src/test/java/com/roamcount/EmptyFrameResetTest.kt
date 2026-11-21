package com.roamcount

/**
 * Standalone tests for EmptyFrameReset (plain main(), no JUnit), from app/src:
 *   kotlinc main/java/com/roamcount/Detection.kt main/java/com/roamcount/UniqueCounter.kt \
 *     main/java/com/roamcount/EmptyFrameReset.kt test/java/com/roamcount/EmptyFrameResetTest.kt \
 *     -include-runtime -d t.jar && java -cp t.jar com.roamcount.EmptyFrameResetTest
 */
object EmptyFrameResetTest {
    private const val SEC = 1_000_000_000L
    private var passed = 0

    private fun check(cond: Boolean, msg: String) {
        if (!cond) throw AssertionError(msg)
    }

    private fun test(name: String, body: () -> Unit) {
        body()
        passed++
        println("ok   $name")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        test("does not fire before the grace period") {
            val r = EmptyFrameReset(2 * SEC)
            check(!r.update(0, 0), "t=0")
            check(!r.update(0, 1 * SEC), "t=1s")
            check(!r.update(0, 2 * SEC - 1), "t just under 2s")
        }
        test("fires once at the grace period, not again while still empty") {
            val r = EmptyFrameReset(2 * SEC)
            r.update(0, 0)
            check(r.update(0, 2 * SEC), "fires at 2s")
            check(!r.update(0, 3 * SEC), "not again at 3s")
            check(!r.update(0, 60 * SEC), "not again at 60s")
        }
        test("a person in frame restarts the empty timer") {
            val r = EmptyFrameReset(2 * SEC)
            r.update(0, 0)
            check(!r.update(0, 1 * SEC), "1s empty")
            check(!r.update(1, 1 * SEC + 100), "person back")
            check(!r.update(0, 2 * SEC), "timer restarted, 2s is only 0s of emptiness")
            check(!r.update(0, 3 * SEC), "1s into new empty stretch")
            check(r.update(0, 4 * SEC), "2s into new empty stretch")
        }
        test("fires again after people return and leave again") {
            val r = EmptyFrameReset(1 * SEC)
            r.update(0, 0)
            check(r.update(0, 1 * SEC), "first stretch")
            r.update(2, 2 * SEC)
            r.update(0, 3 * SEC)
            check(r.update(0, 4 * SEC), "second stretch")
        }
        test("single dropped frame does not fire") {
            val r = EmptyFrameReset(2 * SEC)
            var t = 0L
            repeat(30) { r.update(1, t); t += SEC / 15 }
            check(!r.update(0, t), "one empty frame"); t += SEC / 15
            check(!r.update(1, t), "person again")
        }
        test("grace <= 0 disables it") {
            val r = EmptyFrameReset(0)
            check(!r.update(0, 0) && !r.update(0, 100 * SEC), "disabled")
        }
        test("reset() forgets the current empty stretch") {
            val r = EmptyFrameReset(2 * SEC)
            r.update(0, 0)
            r.reset()
            check(!r.update(0, 2 * SEC), "timer restarted by reset()")
            check(r.update(0, 4 * SEC), "fires 2s after reset")
        }
        test("integration: UniqueCounter returns to 0 after the empty stretch, then counts anew") {
            val counter = UniqueCounter(minTrackAgeFrames = 3)
            val guard = EmptyFrameReset(2 * SEC)
            var t = 0L
            fun frame(ids: List<Int>) {
                counter.update(ids)
                if (guard.update(ids.size, t)) counter.reset()
                t += SEC / 10
            }
            repeat(5) { frame(listOf(1, 2)) }
            check(counter.uniqueTotal == 2, "two people counted")
            repeat(19) { frame(emptyList()) }   // 1.9 s empty
            check(counter.uniqueTotal == 2, "grace not yet over")
            repeat(2) { frame(emptyList()) }
            check(counter.uniqueTotal == 0, "empty >= 2s -> 0")
            repeat(5) { frame(listOf(7)) }
            check(counter.uniqueTotal == 1, "new person counted from 0")
        }
        println("$passed passed")
    }
}

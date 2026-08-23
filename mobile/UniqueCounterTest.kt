package com.roamcount

/**
 * Port of test_counter.py -- same cases, same expected numbers, so the Kotlin
 * counter is provably equivalent to the Python one (portability guardrail).
 *
 * Standalone (no JUnit needed):   kotlinc UniqueCounter.kt UniqueCounterTest.kt -include-runtime -d t.jar && java -jar t.jar
 * Or drop both files into app/src/test and wrap each fun in @Test.
 *
 * NOT EXECUTED here (no Kotlin toolchain in this environment). The identical
 * cases pass 11/11 in Python (`python test_counter.py`).
 */

private fun check(cond: Boolean, msg: String) {
    if (!cond) throw AssertionError(msg)
}

private fun testOccupancyIsLength() {
    check(countOccupancy(emptyList<Int>()) == 0, "empty -> 0")
    check(countOccupancy(listOf(1, 2, 3)) == 3, "three -> 3")
}

private fun testMinAgeGating() {
    val c = UniqueCounter(minTrackAgeFrames = 3)
    check(c.update(listOf(1)).isEmpty() && c.uniqueTotal == 0, "frame1")
    check(c.update(listOf(1)).isEmpty() && c.uniqueTotal == 0, "frame2")
    val ev = c.update(listOf(1))
    check(c.uniqueTotal == 1, "counted at frame3")
    check(ev == listOf(CountEvent("new_track", 1, 1)), "new_track event")
}

private fun testNoDoubleCountWhenStaying() {
    val c = UniqueCounter(minTrackAgeFrames = 2)
    repeat(50) { c.update(listOf(1)) }
    check(c.uniqueTotal == 1, "standing still -> stays 1")
}

private fun testFivePeopleCountsFive() {
    val c = UniqueCounter(minTrackAgeFrames = 3)
    repeat(5) { c.update(listOf(1, 2, 3, 4, 5)) }
    check(c.uniqueTotal == 5, "five people -> 5")
}

private fun testGhostBelowMinAgeNotCounted() {
    val c = UniqueCounter(minTrackAgeFrames = 5)
    c.update(listOf(9))
    val ev = c.update(emptyList())
    check(c.uniqueTotal == 0, "ghost not counted")
    check(ev.isEmpty(), "no events for uncounted ghost")
}

private fun testLostEventOnlyForCounted() {
    val c = UniqueCounter(minTrackAgeFrames = 2)
    c.update(listOf(1)); c.update(listOf(1))
    val ev = c.update(emptyList())
    check(ev == listOf(CountEvent("lost", 1, 1)), "lost event for counted track")
}

private fun testReappearingSameIdNotRecounted() {
    val c = UniqueCounter(minTrackAgeFrames = 2)
    c.update(listOf(1)); c.update(listOf(1))
    c.update(emptyList())
    c.update(listOf(1)); c.update(listOf(1))
    check(c.uniqueTotal == 1, "same id back -> not recounted")
}

private fun testBoundedMemoryMonotonic() {
    val c = UniqueCounter(minTrackAgeFrames = 1, maxHistory = 10)
    for (tid in 1..100) c.update(listOf(tid))
    check(c.uniqueTotal == 100, "count stays correct")
}

private fun testBoundedMemoryNeverRecountsActiveOverCap() {
    val c = UniqueCounter(minTrackAgeFrames = 1, maxHistory = 5)
    val ids = (0 until 20).toList()
    repeat(10) { c.update(ids) }
    check(c.uniqueTotal == 20, "20 always-present -> counted once each")
}

private fun testResetClearsState() {
    val c = UniqueCounter(minTrackAgeFrames = 1)
    c.update(listOf(1, 2, 3))
    check(c.uniqueTotal == 3, "before reset")
    c.reset()
    check(c.uniqueTotal == 0, "after reset")
    c.update(listOf(9))
    check(c.uniqueTotal == 1, "new walk counts from zero")
}

private fun testInvalidArgsRejected() {
    for (bad in listOf(0, -1)) {
        var threw = false
        try { UniqueCounter(minTrackAgeFrames = bad) } catch (e: IllegalArgumentException) { threw = true }
        check(threw, "expected IllegalArgumentException for $bad")
    }
}

fun main() {
    val tests = listOf(
        ::testOccupancyIsLength, ::testMinAgeGating, ::testNoDoubleCountWhenStaying,
        ::testFivePeopleCountsFive, ::testGhostBelowMinAgeNotCounted,
        ::testLostEventOnlyForCounted, ::testReappearingSameIdNotRecounted,
        ::testBoundedMemoryMonotonic, ::testBoundedMemoryNeverRecountsActiveOverCap,
        ::testResetClearsState, ::testInvalidArgsRejected,
    )
    for (t in tests) { t(); println("PASS ${t.name}") }
    println("\n${tests.size} passed")
}

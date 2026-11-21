package com.roamcount

/**
 * Tracks vs ids when the camera is switched (ByteTrackLite.dropTracks).
 *
 *   kotlinc main/java/com/roamcount/Detection.kt main/java/com/roamcount/Nms.kt \
 *     main/java/com/roamcount/ByteTrackLite.kt main/java/com/roamcount/UniqueCounter.kt \
 *     test/java/com/roamcount/ByteTrackLiteTest.kt -include-runtime -d t.jar
 *   java -cp t.jar com.roamcount.ByteTrackLiteTest
 */
object ByteTrackLiteTest {
    private fun check(cond: Boolean, msg: String) { if (!cond) throw AssertionError(msg) }

    private fun person(x: Float = 100f) = Detection(x, 100f, x + 100f, 300f, 0.9f, 0)

    private fun testIdsAreStableWhileTracking() {
        val t = ByteTrackLite()
        val a = t.update(listOf(person())).single().trackId
        val b = t.update(listOf(person(102f))).single().trackId
        check(a == b, "same person keeps the id: $a vs $b")
    }

    private fun testDropTracksKeepsIdsUnique() {
        val t = ByteTrackLite()
        val first = t.update(listOf(person())).single().trackId
        t.dropTracks()
        val second = t.update(listOf(person())).single().trackId
        check(second != first, "after dropTracks the same spot is a NEW track: $first -> $second")
        check(second > first, "ids keep increasing (never reissued): $first -> $second")
    }

    private fun testResetRestartsIds() {
        val t = ByteTrackLite()
        val first = t.update(listOf(person())).single().trackId
        t.reset()
        check(t.update(listOf(person())).single().trackId == first, "reset() restarts the id counter")
    }

    /** The reason dropTracks exists: reset() alone would reissue an id UniqueCounter already counted. */
    private fun testSwitchDoesNotLoseCountsButResetWould() {
        fun run(afterFirstPerson: (ByteTrackLite) -> Unit): Int {
            val tracker = ByteTrackLite(); val counter = UniqueCounter(minTrackAgeFrames = 3)
            repeat(5) { counter.update(tracker.update(listOf(person())).map { it.trackId }) }   // person A, counted
            afterFirstPerson(tracker)
            repeat(5) { counter.update(tracker.update(listOf(person(400f))).map { it.trackId }) } // person B elsewhere
            return counter.uniqueTotal
        }
        check(run { it.dropTracks() } == 2, "dropTracks: A and B both counted")
        check(run { it.reset() } == 1, "reset() without resetting the counter reissues id 1 -> B is lost (the bug)")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val tests = listOf(
            ::testIdsAreStableWhileTracking, ::testDropTracksKeepsIdsUnique, ::testResetRestartsIds,
            ::testSwitchDoesNotLoseCountsButResetWould,
        )
        for (t in tests) { t(); println("PASS ${t.name}") }
        println("\n${tests.size} passed")
    }
}

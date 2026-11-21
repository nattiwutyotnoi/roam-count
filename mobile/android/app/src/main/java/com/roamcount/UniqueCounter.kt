package com.roamcount

/**
 * Cumulative count of unique people from a stream of per-frame track ids.
 *
 * PURE logic -- a 1:1 port of counter.py (portability guardrail, plan section 2.3).
 * No Android / camera / model dependencies, so it can be unit-tested on the JVM
 * exactly like the Python version, and behaves identically frame-for-frame.
 *
 * Rules (plan section 4, Phase 2):
 *  - A track is counted only after it survives `minTrackAgeFrames` consecutive
 *    frames (guards against ghost tracks / one-frame false positives).
 *  - Each track id is counted at most once; `uniqueTotal` never decreases.
 *  - Memory is bounded (`maxHistory`): only ids that are NOT currently active are
 *    evicted. ultralytics track ids increase monotonically and a gone id is never
 *    reissued, so evicting an inactive id can never cause a re-count.
 */

data class CountEvent(val event: String, val trackId: Int, val uniqueTotal: Int)

class UniqueCounter(
    private val minTrackAgeFrames: Int = 5,
    private val maxHistory: Int = 10_000,
) {
    init {
        require(minTrackAgeFrames >= 1) { "minTrackAgeFrames must be >= 1" }
        require(maxHistory >= 1) { "maxHistory must be >= 1" }
    }

    private val age = HashMap<Int, Int>()                  // active id -> consecutive frames seen
    private val counted = LinkedHashMap<Int, Boolean>()    // ids already counted (insertion order)
    private var prevActive: Set<Int> = emptySet()

    var uniqueTotal: Int = 0
        private set

    /** Clear all state for a fresh walk -- uniqueTotal goes back to 0 ('r' hotkey). */
    fun reset() {
        age.clear()
        counted.clear()
        prevActive = emptySet()
        uniqueTotal = 0
    }

    /** Advance one frame with the ids present this frame. Returns lifecycle events. */
    fun update(activeIds: Iterable<Int>): List<CountEvent> {
        val active = activeIds.toHashSet()
        val events = ArrayList<CountEvent>()

        for (tid in active) {
            val a = (age[tid] ?: 0) + 1
            age[tid] = a
            if (!counted.containsKey(tid) && a >= minTrackAgeFrames) {
                counted[tid] = true
                uniqueTotal += 1
                events.add(event("new_track", tid))
            }
        }

        for (tid in prevActive - active) {
            val wasCounted = counted.containsKey(tid)
            age.remove(tid)                 // a re-associated id simply re-ages from 1
            if (wasCounted) events.add(event("lost", tid))
        }

        prevActive = active
        boundCounted()
        return events
    }

    private fun event(name: String, trackId: Int) =
        CountEvent(name, trackId, uniqueTotal)

    private fun boundCounted() {
        if (counted.size <= maxHistory) return
        val it = counted.keys.iterator()    // insertion order -> oldest first
        while (counted.size > maxHistory && it.hasNext()) {
            val tid = it.next()
            if (tid !in prevActive) it.remove()
        }
    }

    /** Tracks present in the most recent frame (confirmed or not). */
    val activeCount: Int get() = prevActive.size
}

/** Persons visible right now = number of detections/tracks in this frame. */
fun countOccupancy(detections: Collection<*>): Int = detections.size

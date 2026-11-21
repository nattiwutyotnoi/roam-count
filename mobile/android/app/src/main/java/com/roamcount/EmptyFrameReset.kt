package com.roamcount

/**
 * Decides when the walk is "over" because nobody has been in frame for a while.
 *
 * PURE logic (no Android deps) so it can be unit-tested on the JVM like UniqueCounter.
 * It deliberately lives OUTSIDE UniqueCounter: that class is a 1:1 port of counter.py and
 * guarantees `uniqueTotal` never decreases. This policy sits on top of it -- the caller
 * (CountingAnalyzer) resets counter + tracker when [update] returns true.
 *
 * A grace period is required: one or two frames with no detection (motion blur, a person
 * briefly leaving the frame) must NOT wipe the count. Fires once per empty stretch.
 *
 * @param graceNanos continuous time with zero persons before firing; <= 0 disables it.
 */
class EmptyFrameReset(private val graceNanos: Long) {
    private var emptySince = NOT_EMPTY
    private var fired = false

    /** Feed one frame. Returns true exactly once, when the frame has been empty for >= graceNanos. */
    fun update(personCount: Int, nowNanos: Long): Boolean {
        if (graceNanos <= 0L) return false
        if (personCount > 0) {
            emptySince = NOT_EMPTY
            fired = false
            return false
        }
        if (emptySince == NOT_EMPTY) emptySince = nowNanos
        if (!fired && nowNanos - emptySince >= graceNanos) {
            fired = true
            return true
        }
        return false
    }

    /** Forget the current empty stretch (call after a manual Reset). */
    fun reset() {
        emptySince = NOT_EMPTY
        fired = false
    }

    private companion object {
        const val NOT_EMPTY = Long.MIN_VALUE
    }
}

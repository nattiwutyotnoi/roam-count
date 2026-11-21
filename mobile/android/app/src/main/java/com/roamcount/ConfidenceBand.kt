package com.roamcount

/**
 * Which colour a PERSON box gets on screen. Display only -- the tracker and UniqueCounter
 * are not affected: a low-confidence person is still tracked and still counted.
 *
 * Pure (no Android deps) so the boundary can be unit-tested on the JVM.
 */
object ConfidenceBand {
    /** Persons below this confidence are drawn in the third (orange) colour. Detector floor is 0.35. */
    const val LOW_CONF_THRESHOLD = 0.5f

    /** Strictly below the threshold is low; exactly at the threshold is normal (green). */
    fun isLow(conf: Float, threshold: Float = LOW_CONF_THRESHOLD): Boolean = conf < threshold
}

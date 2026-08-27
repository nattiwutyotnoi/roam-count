package com.roamcount

// EXAMPLE ONLY / NOT COMPILED. Pure geometry -- unit-testable on the JVM (no Android deps).

/** IoU + greedy Non-Max Suppression. */
object Nms {

    fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = maxOf(0f, a.x2 - a.x1) * maxOf(0f, a.y2 - a.y1)
        val areaB = maxOf(0f, b.x2 - b.x1) * maxOf(0f, b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Keep the highest-score box, drop any remaining box overlapping it by > iouThresh. */
    fun apply(dets: List<Detection>, iouThresh: Float): List<Detection> {
        val pool = dets.sortedByDescending { it.conf }.toMutableList()
        val keep = ArrayList<Detection>()
        while (pool.isNotEmpty()) {
            val best = pool.removeAt(0)
            keep.add(best)
            pool.removeAll { iou(best, it) > iouThresh }
        }
        return keep
    }
}

package com.roamcount

// EXAMPLE ONLY / NOT COMPILED.

/**
 * ByteTrackLite -- Option A tracker: IoU/motion association only. NO Kalman, NO ReID.
 *
 * A pragmatic starting point to hit the >=15 FPS DoD. Known limits (by design):
 *   - re-counts a person who leaves the frame and returns (no appearance memory)
 *   - weaker when the camera moves fast (no motion prediction / GMC)
 * If golden-set error is too high *because of re-entries*, port BoT-SORT + ReID
 * (Option B) -- see ../../README.md. Decide with numbers, not now (plan section 4.5).
 *
 * Emits stable-ish track ids; feed `update(...).map { it.trackId }` into UniqueCounter.
 */
class ByteTrackLite(
    private val iouMatchThresh: Float = 0.3f,
    private val trackBuffer: Int = 30,   // frames a lost track survives before removal
    private val highConf: Float = 0.35f, // ByteTrack first-stage score gate
) {
    private class Track(var box: Detection, val id: Int, var lostFrames: Int = 0)

    private val tracks = ArrayList<Track>()
    private var nextId = 1

    /** One frame of detections -> the persons tracked in this frame. */
    fun update(detections: List<Detection>): List<TrackedPerson> {
        val dets = detections.filter { it.conf >= highConf }
        val matched = BooleanArray(dets.size)

        // Greedy IoU association: each existing track grabs its best-overlap detection.
        for (t in tracks) {
            var bestIdx = -1
            var bestIou = iouMatchThresh
            for (i in dets.indices) {
                if (matched[i]) continue
                val v = Nms.iou(t.box, dets[i])
                if (v > bestIou) {
                    bestIou = v
                    bestIdx = i
                }
            }
            if (bestIdx >= 0) {
                t.box = dets[bestIdx]
                t.lostFrames = 0
                matched[bestIdx] = true
            } else {
                t.lostFrames += 1
            }
        }

        // Unmatched detections -> new tracks.
        for (i in dets.indices) {
            if (!matched[i]) tracks.add(Track(dets[i], nextId++))
        }

        // Bound memory: forget tracks lost longer than the buffer.
        tracks.removeAll { it.lostFrames > trackBuffer }

        // Report only tracks actually seen this frame.
        return tracks.filter { it.lostFrames == 0 }.map {
            TrackedPerson(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.conf, it.id)
        }
    }

    fun reset() {
        tracks.clear()
        nextId = 1
    }
}

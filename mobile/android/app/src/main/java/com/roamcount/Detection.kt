package com.roamcount

// EXAMPLE ONLY / NOT COMPILED (no Android toolchain in the dev environment).

/** A detected person in pixel coords of the ORIGINAL frame (letterbox already undone). */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val cls: Int,
)

/** A tracked person = a detection plus a stable track id (this id feeds UniqueCounter). */
data class TrackedPerson(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val trackId: Int,
)

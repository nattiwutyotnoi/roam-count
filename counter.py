"""Counting logic -- PURE module (no OpenCV, no torch, no I/O).

Kept dependency-free on purpose: this is the one piece that must convert to
Kotlin 1:1 for the mobile port (portability guardrail, plan section 2.3).

  * count_occupancy  -- persons visible in the current frame (Phase 1)
  * UniqueCounter    -- cumulative count of unique people across frames (Phase 2)
"""
from __future__ import annotations

from collections.abc import Iterable, Sized


def count_occupancy(detections: Sized) -> int:
    """Persons visible right now = number of detections/tracks in this frame."""
    return len(detections)


class UniqueCounter:
    """Cumulative count of unique people, from a stream of per-frame track ids.

    Rules (see plan section 4, Phase 2):
      * A track is counted only after it survives `min_track_age_frames`
        consecutive frames -- guards against ghost tracks / one-frame false
        positives inflating the total.
      * Each track id is counted at most once; `unique_total` never decreases.
      * Memory is bounded (`max_history`) so a long run cannot leak (risk R7).
        Eviction is safe: ultralytics track ids are monotonically increasing and
        a truly-gone id is never reissued, so an evicted id cannot be re-counted.

    Emits lifecycle events (no image data) for the JSONL log:
      {"event": "new_track", "track_id": t, "unique_total": n}  -- when counted
      {"event": "lost",      "track_id": t, "unique_total": n}  -- counted track gone
    """

    def __init__(self, min_track_age_frames: int = 5, max_history: int = 10000):
        if min_track_age_frames < 1:
            raise ValueError("min_track_age_frames must be >= 1")
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.min_age = int(min_track_age_frames)
        self.max_history = int(max_history)
        self._age: dict[int, int] = {}       # active id -> consecutive frames seen
        self._counted: dict[int, None] = {}  # ids already counted (insertion-ordered)
        self._prev_active: set[int] = set()
        self.unique_total = 0                 # monotonic

    def reset(self) -> None:
        """Clear all state for a fresh walk -- unique_total goes back to 0.

        Used by the 'r' hotkey (Phase 4) to start counting a new route without
        restarting the program (plan Open Question #3: reset per walk)."""
        self._age.clear()
        self._counted.clear()
        self._prev_active = set()
        self.unique_total = 0

    def update(self, active_ids: Iterable[int]) -> list[dict]:
        """Advance one frame with the ids present this frame. Returns events."""
        active = {int(i) for i in active_ids}
        events: list[dict] = []

        for tid in active:
            self._age[tid] = self._age.get(tid, 0) + 1
            if tid not in self._counted and self._age[tid] >= self.min_age:
                self._counted[tid] = None
                self.unique_total += 1
                events.append(self._event("new_track", tid))

        for tid in self._prev_active - active:
            was_counted = tid in self._counted
            self._age.pop(tid, None)  # a re-associated id will simply re-age from 1
            if was_counted:
                events.append(self._event("lost", tid))

        self._prev_active = active
        self._bound_counted()
        return events

    def _event(self, name: str, track_id: int) -> dict:
        return {"event": name, "track_id": track_id, "unique_total": self.unique_total}

    def _bound_counted(self) -> None:
        # Bound memory (risk R7) by evicting only ids that are NOT currently
        # active: ultralytics track ids increase monotonically and a gone id is
        # never reissued, so evicting an inactive id can never cause a re-count.
        # Active ids are always kept -- never double-count someone still on screen,
        # even if that briefly pushes the map past max_history.
        if len(self._counted) <= self.max_history:
            return
        for tid in list(self._counted):  # insertion order -> oldest first
            if len(self._counted) <= self.max_history:
                break
            if tid not in self._prev_active:
                del self._counted[tid]

    @property
    def active_count(self) -> int:
        """Tracks present in the most recent frame (confirmed or not)."""
        return len(self._prev_active)

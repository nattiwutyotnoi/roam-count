"""Unit tests for counter.py (pure logic -- fully deterministic, no camera/model).

Run:  python test_counter.py           (self-contained, no pytest needed)
  or: python -m pytest test_counter.py  (if pytest is installed)
"""
from counter import UniqueCounter, count_occupancy


def test_occupancy_is_length():
    assert count_occupancy([]) == 0
    assert count_occupancy([1, 2, 3]) == 3


def test_min_age_gating():
    c = UniqueCounter(min_track_age_frames=3)
    assert c.update([1]) == [] and c.unique_total == 0        # frame 1
    assert c.update([1]) == [] and c.unique_total == 0        # frame 2
    ev = c.update([1])                                        # frame 3 -> counted
    assert c.unique_total == 1
    assert ev == [{"event": "new_track", "track_id": 1, "unique_total": 1}]


def test_no_double_count_when_staying():
    c = UniqueCounter(min_track_age_frames=2)
    for _ in range(50):
        c.update([1])
    assert c.unique_total == 1  # stand still -> total does not keep rising


def test_five_people_walking_counts_five():
    c = UniqueCounter(min_track_age_frames=3)
    for _ in range(5):                 # all five present long enough
        c.update([1, 2, 3, 4, 5])
    assert c.unique_total == 5


def test_ghost_track_below_min_age_not_counted():
    c = UniqueCounter(min_track_age_frames=5)
    c.update([9])            # appears for a single frame ...
    ev = c.update([])        # ... then gone before confirmation
    assert c.unique_total == 0
    assert ev == []          # no new_track, and no "lost" for an uncounted ghost


def test_lost_event_only_for_counted_track():
    c = UniqueCounter(min_track_age_frames=2)
    c.update([1])
    c.update([1])            # counted here
    ev = c.update([])        # now gone -> lost event
    assert ev == [{"event": "lost", "track_id": 1, "unique_total": 1}]


def test_reappearing_same_id_not_recounted():
    c = UniqueCounter(min_track_age_frames=2)
    c.update([1]); c.update([1])        # counted (total=1)
    c.update([])                        # lost
    c.update([1]); c.update([1])        # same id back
    assert c.unique_total == 1          # not double counted


def test_bounded_memory_keeps_total_monotonic():
    c = UniqueCounter(min_track_age_frames=1, max_history=10)
    for tid in range(1, 101):           # 100 distinct one-frame-confirmed ids
        c.update([tid])
    assert c.unique_total == 100         # count stays correct ...
    assert len(c._counted) <= 10         # ... while memory stays bounded


def test_bounded_memory_never_recounts_active_over_cap():
    # More people continuously on screen than max_history: must NOT double count.
    c = UniqueCounter(min_track_age_frames=1, max_history=5)
    ids = list(range(20))
    for _ in range(10):
        c.update(ids)
    assert c.unique_total == 20          # each counted exactly once across frames


def test_reset_clears_state():
    c = UniqueCounter(min_track_age_frames=1)
    c.update([1, 2, 3])
    assert c.unique_total == 3
    c.reset()
    assert c.unique_total == 0
    c.update([9])            # a brand-new walk counts from zero again
    assert c.unique_total == 1


def test_invalid_args_rejected():
    for bad in (0, -1):
        try:
            UniqueCounter(min_track_age_frames=bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()

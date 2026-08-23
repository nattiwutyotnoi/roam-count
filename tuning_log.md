# Tuning Log — Phase 3

Evidence of every tuning round (plan section 4, Phase 3). **Rule: change ONE
parameter at a time** — otherwise you can't tell which change moved the number.

- Tune only against `split: tune` clips. Keep `split: holdout` untouched until the
  very end, then run it once to check you didn't overfit (risk R5).
- Target: **mean count error ≤ 15%** on the golden set (Success Metric).
- Results below are appended automatically by `evaluate.py --report tuning_log.md`.

## How to run

```bash
# baseline (current config.json) on the tune set
python evaluate.py --split tune --report tuning_log.md

# sweep ONE parameter at a time
python evaluate.py --sweep tracker.track_buffer=20,30,45,60 --report tuning_log.md
python evaluate.py --sweep tracker.gmc_method=sparseOptFlow,orb --report tuning_log.md
python evaluate.py --sweep detector.conf=0.30,0.35,0.40 --report tuning_log.md
python evaluate.py --sweep tracker.new_track_thresh=0.5,0.6,0.7 --report tuning_log.md
python evaluate.py --sweep counter.min_track_age_frames=3,5,8 --report tuning_log.md

# after picking values, edit config.json, then confirm on the holdout set
python evaluate.py --split holdout --report tuning_log.md
```

## Parameters worth sweeping (plan section 4)

| Parameter | Effect |
|---|---|
| `tracker.track_buffer` | frames a lost track survives. Hand-carried → start low (~30): people who leave frame rarely return, and long buffers cause ID reuse / double counting. |
| `tracker.gmc_method` | camera-motion compensation. `sparseOptFlow` (default) vs `orb` — test both when walking is fast / ground is rough. |
| `tracker.match_thresh` | association strictness (motion/IoU). |
| `tracker.new_track_thresh` | min conf to start a new track — higher = fewer ghost tracks. |
| `tracker.appearance_thresh` / `proximity_thresh` | ReID matching strictness. |
| `counter.min_track_age_frames` | frames before a track is counted — guards against ghost tracks inflating the total. |

## Rounds

<!-- evaluate.py appends timestamped result blocks below this line -->

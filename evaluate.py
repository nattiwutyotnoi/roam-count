"""Phase 3 evaluation harness -- run golden clips through the pipeline and report
count error vs. human-counted ground truth.

Deterministic: seeds are fixed and each clip runs on a fresh tracker, so the same
clip + config yields the same number every run (Phase 3 DoD).

Usage:
  python evaluate.py                                  eval all clips in golden_set.json
  python evaluate.py --split tune                     only the tune set
  python evaluate.py --split holdout                  only the untouched holdout set
  python evaluate.py --sweep tracker.track_buffer=20,30,45,60   tune ONE param
  python evaluate.py --sweep detector.conf=0.3,0.35,0.4 --report tuning_log.md
  python evaluate.py --set my_golden.json --config config.json

Golden clips live in the (gitignored) clips_dir from golden_set.json -- never
committed (privacy, plan section 7). golden_set.json itself holds only numbers.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from camera import CameraSource
from counter import UniqueCounter
from tracker import Tracker


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_unique_in_video(config: dict, video_path: str) -> tuple[int, int, float]:
    """Run one clip through a FRESH tracker + counter. Returns
    (predicted_unique, frames, fps). Fresh state -> no cross-clip contamination."""
    det_cfg = dict(config["detector"])
    tracker = Tracker(det_cfg, dict(config.get("tracker", {})))
    cnt_cfg = config.get("counter", {})
    counter = UniqueCounter(
        min_track_age_frames=cnt_cfg.get("min_track_age_frames", 5),
        max_history=cnt_cfg.get("max_history", 10000),
    )
    frames = 0
    t0 = time.perf_counter()
    with CameraSource(video_path) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break
            persons = tracker.track(frame)
            counter.update(p.track_id for p in persons)
            frames += 1
    dt = time.perf_counter() - t0
    return counter.unique_total, frames, (frames / dt if dt else 0.0)


def evaluate(config: dict, golden: dict, clips: list[dict], seed: int) -> list[dict]:
    clips_dir = Path(golden.get("clips_dir", "golden"))
    rows: list[dict] = []
    for clip in clips:
        path = clips_dir / clip["file"]
        row = {
            "file": clip["file"],
            "split": clip.get("split", "tune"),
            "scenario": clip.get("scenario", ""),
            "ground_truth": clip["unique_count"],
        }
        if not path.exists():
            rows.append({**row, "predicted": None, "error_pct": None,
                         "fps": None, "status": "MISSING"})
            continue
        set_seeds(seed)  # determinism per clip
        pred, _frames, fps = count_unique_in_video(config, str(path))
        gt = clip["unique_count"]
        err = abs(pred - gt) / gt * 100.0 if gt else float("inf")
        rows.append({**row, "predicted": pred, "error_pct": err,
                     "fps": fps, "status": "OK"})
    return rows


def aggregate(rows: list[dict], split: str | None = None) -> dict | None:
    ok = [r for r in rows if r["status"] == "OK"
          and (split is None or r["split"] == split)]
    if not ok:
        return None
    errs = [r["error_pct"] for r in ok]
    return {
        "clips": len(ok),
        "mean_error_pct": sum(errs) / len(errs),
        "max_error_pct": max(errs),
        "mean_fps": sum(r["fps"] for r in ok) / len(ok),
    }


def format_table(rows: list[dict]) -> str:
    head = f"{'file':<28} {'split':<8} {'GT':>4} {'pred':>5} {'err%':>7} {'fps':>6}  status"
    lines = [head, "-" * len(head)]
    for r in rows:
        pred = "-" if r["predicted"] is None else str(r["predicted"])
        err = "-" if r["error_pct"] is None else f"{r['error_pct']:.1f}"
        fps = "-" if r["fps"] is None else f"{r['fps']:.1f}"
        lines.append(f"{r['file']:<28} {r['split']:<8} {r['ground_truth']:>4} "
                     f"{pred:>5} {err:>7} {fps:>6}  {r['status']}")
    return "\n".join(lines)


def print_report(rows: list[dict]) -> None:
    print(format_table(rows))
    missing = [r for r in rows if r["status"] == "MISSING"]
    if missing:
        print(f"\n[warn] {len(missing)} clip(s) missing on disk -> excluded from totals")
    for split in ("tune", "holdout"):
        agg = aggregate(rows, split)
        if agg:
            gate = "PASS" if agg["mean_error_pct"] <= 15 else "FAIL"
            print(f"[{split:<7}] clips={agg['clips']} "
                  f"mean_err={agg['mean_error_pct']:.1f}% "
                  f"max_err={agg['max_error_pct']:.1f}% "
                  f"mean_fps={agg['mean_fps']:.1f}  gate(<=15%): {gate}")


# ---- one-parameter sweep (tuning) ------------------------------------------
def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def run_sweep(base_config: dict, golden: dict, clips: list[dict],
              spec: str, seed: int) -> list[dict]:
    """spec = "dotted.key=v1,v2,v3" -- vary ONE parameter, all else fixed."""
    key, _, raw = spec.partition("=")
    values = [_coerce(v.strip()) for v in raw.split(",") if v.strip()]
    results = []
    for val in values:
        cfg = copy.deepcopy(base_config)
        set_by_path(cfg, key.strip(), val)
        rows = evaluate(cfg, golden, clips, seed)
        agg = aggregate(rows) or {"mean_error_pct": float("nan"),
                                  "max_error_pct": float("nan"), "mean_fps": float("nan")}
        results.append({"param": key.strip(), "value": val, **agg})
        print(f"  {key.strip()}={val!r:<10} mean_err={agg['mean_error_pct']:.1f}% "
              f"max_err={agg['max_error_pct']:.1f}% mean_fps={agg['mean_fps']:.1f}")
    return results


def append_report(path: str, title: str, body: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n### {ts} — {title}\n\n```\n{body}\n```\n")
    print(f"\n[report] appended to {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 golden-set evaluation")
    p.add_argument("--config", default="config.json")
    p.add_argument("--model", default=None,
                   help="override detector.model (e.g. yolo11n.onnx) to A/B a "
                        ".pt vs an exported model on the same clips")
    p.add_argument("--set", dest="golden_path", default="golden_set.json")
    p.add_argument("--split", choices=["all", "tune", "holdout"], default="all")
    p.add_argument("--sweep", default=None,
                   help='vary ONE param, e.g. "tracker.track_buffer=20,30,45"')
    p.add_argument("--report", default=None, help="append results table to this file")
    p.add_argument("--seed", type=int, default=0)
    return p


def _select_clips(golden: dict, split: str) -> list[dict]:
    clips = golden.get("clips", [])
    if split == "all":
        return clips
    return [c for c in clips if c.get("split", "tune") == split]


def main() -> int:
    args = build_parser().parse_args()
    config = load_json(args.config)
    if args.model:
        config["detector"]["model"] = args.model
    golden = load_json(args.golden_path)

    # For a sweep, default to the tune set (never tune against holdout).
    split = args.split
    if args.sweep and split == "all":
        split = "tune"
    clips = _select_clips(golden, split)

    if not clips:
        print(f"No clips for split={split!r} in {args.golden_path}.")
        print("Add entries under \"clips\" and drop the files in the clips_dir "
              f"({golden.get('clips_dir', 'golden')}/). See the _example in that file.")
        return 0

    if args.sweep:
        print(f"Sweep on {split} set ({len(clips)} clips): {args.sweep}")
        results = run_sweep(config, golden, clips, args.sweep, args.seed)
        if args.report:
            body = "\n".join(
                f"{r['param']}={r['value']!r}  mean_err={r['mean_error_pct']:.1f}%  "
                f"max_err={r['max_error_pct']:.1f}%  mean_fps={r['mean_fps']:.1f}"
                for r in results)
            append_report(args.report, f"sweep {args.sweep} (split={split})", body)
        return 0

    rows = evaluate(config, golden, clips, args.seed)
    print_report(rows)
    if args.report:
        append_report(args.report, f"eval (split={split})", format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

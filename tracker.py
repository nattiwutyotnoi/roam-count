"""Tracker (Phase 2) -- BoT-SORT with ReID + camera-motion compensation.

Wraps ultralytics `model.track(persist=True, ...)`. The tracker YAML is generated
from config.json at startup so config.json stays the single source of truth for
every threshold (portability guardrail, plan section 2.3): tuning in Phase 3
happens in config.json and propagates to the generated YAML.

For a hand-carried (moving) camera the plan requires:
  * with_reid: true          -- appearance matching so people leaving/re-entering
                                aren't re-counted as new
  * gmc_method: sparseOptFlow -- compensate for camera motion
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ultralytics
import yaml
from ultralytics import YOLO

from detector import infer_params

# config.json tracker keys -> botsort.yaml keys (only these are overlaid).
_TRACKER_KEY_MAP = {
    "with_reid": "with_reid",
    "gmc_method": "gmc_method",
    "track_buffer": "track_buffer",
    "match_thresh": "match_thresh",
    "new_track_thresh": "new_track_thresh",
    "track_high_thresh": "track_high_thresh",
    "track_low_thresh": "track_low_thresh",
    "proximity_thresh": "proximity_thresh",
    "appearance_thresh": "appearance_thresh",
}


@dataclass(frozen=True)
class TrackedPerson:
    """One tracked person for the current frame, with its stable track id."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: int


def _shipped_botsort_defaults() -> dict:
    p = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def build_tracker_yaml(tracker_cfg: dict, out_path: str | None = None) -> str:
    """Write botsort_custom.yaml = shipped defaults + config.json overrides.

    Returns the path to the generated YAML (to pass to model.track()).
    """
    cfg = _shipped_botsort_defaults()
    for src_key, yaml_key in _TRACKER_KEY_MAP.items():
        if src_key in tracker_cfg and tracker_cfg[src_key] is not None:
            cfg[yaml_key] = tracker_cfg[src_key]
    # "auto" lets BoT-SORT reuse the detector's features for ReID -- no extra
    # model download, keeps FPS reasonable on the moving-camera path.
    if cfg.get("with_reid"):
        cfg.setdefault("model", "auto")

    out = out_path or tracker_cfg.get("config_file", "botsort_custom.yaml")
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


class Tracker:
    def __init__(self, det_cfg: dict, tracker_cfg: dict):
        self.p = infer_params(det_cfg)
        self.model = YOLO(self.p.model_name)
        self.tracker_yaml = build_tracker_yaml(tracker_cfg)

    @property
    def device(self) -> str:
        return self.p.device

    def track(self, frame) -> list[TrackedPerson]:
        """Track persons in one BGR frame. Only detections that the tracker has
        assigned a stable id to are returned (untracked boxes are skipped)."""
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            imgsz=self.p.imgsz,
            conf=self.p.conf,
            iou=self.p.iou,
            classes=self.p.classes,
            device=self.p.device,
            quantize=self.p.quantize,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        return [
            TrackedPerson(float(x1), float(y1), float(x2), float(y2), float(c), int(t))
            for (x1, y1, x2, y2), c, t in zip(xyxy, confs, ids)
        ]

    @property
    def summary(self) -> dict:
        return {
            "model": self.p.model_name,
            "device": self.p.device,
            "quantize": self.p.quantize,
            "imgsz": self.p.imgsz,
            "conf": self.p.conf,
            "tracker_yaml": self.tracker_yaml,
        }

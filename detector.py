"""YOLO person detector (Phase 1) + shared inference-parameter resolution.

Only class 0 (person) is kept. Every threshold comes from config -- nothing is
hard-coded -- so the values tuned on PC carry over to mobile unchanged
(portability guardrail, plan section 2.3). Model is fixed to YOLO11n.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """One detected person, in pixel coordinates of the input frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


def resolve_device(requested: str) -> str:
    """Map config's device ("auto"/"cuda:0"/"cpu") to a concrete torch device.

    "auto" -> cuda:0 when a GPU is available, else cpu. Falling back to cpu keeps
    the program runnable on machines without CUDA instead of crashing.
    """
    if requested and requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class InferParams:
    """Inference args shared by the detector (predict) and tracker (track)."""
    device: str
    quantize: str | None
    imgsz: int
    conf: float
    iou: float
    classes: list[int]
    model_name: str


def infer_params(cfg: dict) -> InferParams:
    """Resolve the common inference parameters from the detector config block.

    Single source of truth so Detector and Tracker cannot drift apart.
    ultralytics 8.4 replaced the `half` predict arg with `quantize`; config's
    `half: true` maps to quantize="fp16" (GPU only -- silently off on CPU).
    """
    device = resolve_device(cfg.get("device", "auto"))
    want_half = bool(cfg.get("half", False)) and device.startswith("cuda")
    return InferParams(
        device=device,
        quantize="fp16" if want_half else None,
        imgsz=int(cfg.get("imgsz", 640)),
        conf=float(cfg.get("conf", 0.35)),
        iou=float(cfg.get("iou", 0.5)),
        classes=list(cfg.get("classes", [0])),
        model_name=cfg.get("model", "yolo11n.pt"),
    )


class Detector:
    def __init__(self, cfg: dict):
        self.p = infer_params(cfg)
        self.model = YOLO(self.p.model_name)

    # convenience passthroughs used by main's banner / HUD
    @property
    def device(self) -> str:
        return self.p.device

    @property
    def imgsz(self) -> int:
        return self.p.imgsz

    def detect(self, frame, imgsz: int | None = None) -> list[Detection]:
        """Run detection on one BGR frame and return the list of persons."""
        result = self.model.predict(
            frame,
            imgsz=imgsz or self.p.imgsz,
            conf=self.p.conf,
            iou=self.p.iou,
            classes=self.p.classes,
            device=self.p.device,
            quantize=self.p.quantize,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(float(x1), float(y1), float(x2), float(y2), float(c), int(k))
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss)
        ]

    @property
    def summary(self) -> dict:
        return {
            "model": self.p.model_name,
            "device": self.p.device,
            "quantize": self.p.quantize,
            "imgsz": self.p.imgsz,
            "conf": self.p.conf,
            "iou": self.p.iou,
            "classes": self.p.classes,
        }

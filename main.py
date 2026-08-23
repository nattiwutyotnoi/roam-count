"""roam-count -- entry point.

Phase 2 (default live mode): person tracking with cumulative unique count.
Phase 1 modes kept for detection-only work and FPS benchmarking.

Usage:
  python main.py                       live tracking + unique count (webcam/file/RTSP)
  python main.py --source clip.mp4     override the camera source
  python main.py --detect-only         Phase 1: detection only (no tracking/count)
  python main.py --imgsz 480           override detector input size
  python main.py --bench [--sweep]     headless inference-FPS benchmark
  python main.py --bench-raw           Phase 0: raw camera read FPS (no model)
  python main.py --no-log              do not write the JSONL event log

Hotkeys (live tracking): q/ESC = quit, p or SPACE = pause, r = reset count.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from camera import CameraSource
from counter import UniqueCounter, count_occupancy
from detector import Detector
from tracker import Tracker

# ---- overlay styling -------------------------------------------------------
_BOX_COLOR = (0, 200, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- event log (numbers + timestamp only -- never an image, plan section 7) --
class EventLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, event: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---- drawing ---------------------------------------------------------------
def draw_boxes(frame, items, with_id: bool) -> None:
    for it in items:
        p1 = (int(it.x1), int(it.y1))
        p2 = (int(it.x2), int(it.y2))
        cv2.rectangle(frame, p1, p2, _BOX_COLOR, _BOX_THICKNESS)
        label = f"id {it.track_id} {it.conf:.2f}" if with_id else f"{it.conf:.2f}"
        cv2.putText(frame, label, (p1[0], max(0, p1[1] - 6)),
                    _FONT, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, lines: list[str]) -> None:
    y = 30
    for text in lines:
        # black outline then white fill -> readable over any background
        cv2.putText(frame, text, (12, y), _FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), _FONT, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        y += 34


def _text(frame, s, org, scale, color, weight=1) -> None:
    """Text with a black outline so it stays readable over any background."""
    cv2.putText(frame, s, org, _FONT, scale, (0, 0, 0), weight + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, _FONT, scale, color, weight, cv2.LINE_AA)


def draw_track_overlay(frame, unique_total, occupancy, fps, device, paused) -> None:
    """Phase 4 HUD: big unique-total number + smaller status lines + hotkey hint."""
    green, white = (0, 220, 0), (255, 255, 255)
    _text(frame, str(unique_total), (16, 92), 3.0, green, weight=6)
    _text(frame, "UNIQUE TOTAL", (20, 120), 0.6, white, weight=1)
    for i, line in enumerate((f"Occupancy: {occupancy}",
                              f"FPS: {fps:4.1f}",
                              f"Device: {device}")):
        _text(frame, line, (20, 152 + i * 28), 0.7, white, weight=1)
    if paused:
        _text(frame, "PAUSED", (20, 244), 0.9, (0, 200, 255), weight=2)
    _text(frame, "q/ESC quit   p/SPACE pause   r reset",
          (12, frame.shape[0] - 14), 0.55, white, weight=1)


def _handle_key(key, counter, logger, paused):
    """Process one hotkey. Returns (should_quit, paused)."""
    if key in (ord("q"), 27):
        return True, paused
    if key in (ord("p"), ord(" ")):
        return False, not paused
    if key == ord("r"):
        before = counter.unique_total
        counter.reset()
        if logger:
            logger.write({"event": "reset", "unique_total": 0})
        print(f"[reset] unique_total {before} -> 0")
    return False, paused


def _paused_step(last_frame, counter, tracker, fps, show_window, window_name, logger):
    """Render the paused view and poll keys. Returns (should_quit, paused)."""
    if not (show_window and last_frame is not None):
        return True, False  # headless: pausing is meaningless -> stop
    view = last_frame.copy()
    draw_track_overlay(view, counter.unique_total, 0, fps, tracker.device, True)
    cv2.imshow(window_name, view)
    return _handle_key(cv2.waitKey(50) & 0xFF, counter, logger, True)


def _reconnect_loop(cam, show_window, window_name, logger, backoff_max=5.0) -> bool:
    """Auto-reconnect a dropped live camera with capped backoff. Returns True on
    success, False if the user quit during reconnection (windowed mode only)."""
    delay = 0.5
    while True:
        if show_window:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            draw_hud(blank, ["CAMERA LOST", "reconnecting...", "q/ESC to quit"])
            cv2.imshow(window_name, blank)
            if (cv2.waitKey(int(delay * 1000)) & 0xFF) in (ord("q"), 27):
                return False
        else:
            time.sleep(delay)
        if cam.reconnect():
            print("[camera] reconnected")
            if logger:
                logger.write({"event": "camera_reconnect"})
            return True
        delay = min(delay * 2, backoff_max)


def _camera_from(config: dict, args) -> CameraSource:
    cam_cfg = config["camera"]
    source = args.source if args.source is not None else cam_cfg["source"]
    return CameraSource(source, cam_cfg.get("width"), cam_cfg.get("height"),
                        cam_cfg.get("fps_request"))


def _rolling_fps(frame_times: deque) -> float:
    return len(frame_times) / sum(frame_times) if frame_times else 0.0


# ---- Phase 2: tracking + unique count (default) ----------------------------
def run_track(config: dict, args) -> int:
    det_cfg = dict(config["detector"])
    if args.imgsz is not None:
        det_cfg["imgsz"] = args.imgsz

    tracker = Tracker(det_cfg, dict(config.get("tracker", {})))
    cnt_cfg = config.get("counter", {})
    counter = UniqueCounter(
        min_track_age_frames=cnt_cfg.get("min_track_age_frames", 5),
        max_history=cnt_cfg.get("max_history", 10000),
    )
    print("[tracker]", tracker.summary)

    out_cfg = config.get("output", {})
    show_window = out_cfg.get("show_window", True) and not args.no_window
    window_name = out_cfg.get("window_name", "roam-count")
    logger = None
    if not args.no_log and out_cfg.get("log_path"):
        logger = EventLogger(out_cfg["log_path"])
        logger.write({"event": "session_start", "unique_total": 0})

    frame_times: deque[float] = deque(maxlen=30)
    frames = 0
    fps = 0.0
    paused = False
    last_frame = None
    try:
        with _camera_from(config, args) as cam:
            print("[camera]", cam.info)
            print("[hotkeys] q/ESC quit | p or SPACE pause | r reset")
            while True:
                if paused:
                    quit_, paused = _paused_step(last_frame, counter, tracker, fps,
                                                 show_window, window_name, logger)
                    if quit_:
                        break
                    continue

                frame = cam.read()
                if frame is None:
                    if cam.is_file:
                        print("[camera] end of stream -> stopping")
                        break
                    print("[camera] lost -> reconnecting")
                    if logger:
                        logger.write({"event": "camera_lost"})
                    if not _reconnect_loop(cam, show_window, window_name, logger):
                        break
                    continue

                t0 = time.perf_counter()
                persons = tracker.track(frame)
                events = counter.update(p.track_id for p in persons)
                frame_times.append(time.perf_counter() - t0)
                fps = _rolling_fps(frame_times)
                if logger:
                    for e in events:
                        logger.write(e)

                frames += 1
                if show_window:
                    draw_boxes(frame, persons, with_id=True)
                    draw_track_overlay(frame, counter.unique_total,
                                       count_occupancy(persons), fps, tracker.device, False)
                    last_frame = frame
                    cv2.imshow(window_name, frame)
                    quit_, paused = _handle_key(cv2.waitKey(1) & 0xFF, counter, logger, paused)
                    if quit_:
                        break
                elif args.duration and frames / max(fps, 1e-6) >= args.duration:
                    break
    except KeyboardInterrupt:
        print("\n[shutdown] interrupted -> closing cleanly")
    finally:
        if logger:
            logger.write({"event": "session_end", "unique_total": counter.unique_total})
            logger.close()
        if show_window:
            cv2.destroyAllWindows()

    print(f"[track] frames={frames} unique_total={counter.unique_total} fps~{fps:.1f}")
    return 0


# ---- Phase 1: detection only -----------------------------------------------
def run_detect_only(config: dict, args) -> int:
    det_cfg = dict(config["detector"])
    if args.imgsz is not None:
        det_cfg["imgsz"] = args.imgsz
    detector = Detector(det_cfg)
    print("[detector]", detector.summary)

    out_cfg = config.get("output", {})
    show_window = out_cfg.get("show_window", True) and not args.no_window
    window_name = out_cfg.get("window_name", "roam-count")

    frame_times: deque[float] = deque(maxlen=30)
    frames = 0
    fps = 0.0
    with _camera_from(config, args) as cam:
        print("[camera]", cam.info)
        while True:
            frame = cam.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            frame_times.append(time.perf_counter() - t0)
            fps = _rolling_fps(frame_times)
            frames += 1
            if show_window:
                draw_boxes(frame, detections, with_id=False)
                draw_hud(frame, [
                    f"Occupancy: {count_occupancy(detections)}",
                    f"FPS: {fps:4.1f}",
                    f"Device: {detector.device}",
                ])
                cv2.imshow(window_name, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif args.duration and frames / max(fps, 1e-6) >= args.duration:
                break
    if show_window:
        cv2.destroyAllWindows()
    print(f"[detect] frames={frames} fps~{fps:.1f}")
    return 0


# ---- benchmark helpers -----------------------------------------------------
def _get_bench_frame(config: dict):
    """A representative frame for benchmarking. Prefer an image that actually
    contains people so the person-count is also sanity-checked."""
    try:
        from ultralytics.utils import ASSETS
        sample = Path(ASSETS) / "bus.jpg"
        img = cv2.imread(str(sample))
        if img is not None:
            return img, f"sample:{sample.name}"
    except Exception:
        pass
    try:
        with _camera_from(config, argparse.Namespace(source=None)) as cam:
            frame = cam.read()
            if frame is not None:
                return frame, "camera"
    except Exception:
        pass
    h = config["camera"].get("height") or 720
    w = config["camera"].get("width") or 1280
    return np.zeros((h, w, 3), dtype=np.uint8), "synthetic-black"


def run_bench(config: dict, args) -> int:
    det_cfg = dict(config["detector"])
    detector = Detector(det_cfg)
    print("[detector]", detector.summary)

    frame, source = _get_bench_frame(config)
    print(f"[bench] frame source={source} shape={frame.shape} iters={args.iters}")

    sizes = [640, 480, 320] if args.sweep else [args.imgsz or det_cfg["imgsz"]]
    print(f"{'imgsz':>6} | {'persons':>7} | {'FPS':>7} | {'ms/frame':>9}")
    print("-" * 40)
    for imgsz in sizes:
        for _ in range(5):  # warmup (model load, CUDA init, weight download)
            detector.detect(frame, imgsz=imgsz)
        persons = len(detector.detect(frame, imgsz=imgsz))
        t0 = time.perf_counter()
        for _ in range(args.iters):
            detector.detect(frame, imgsz=imgsz)
        elapsed = time.perf_counter() - t0
        print(f"{imgsz:>6} | {persons:>7} | {args.iters / elapsed:>7.1f} | "
              f"{1000.0 * elapsed / args.iters:>9.2f}")
    return 0


def run_bench_raw(config: dict, args) -> int:
    """Phase 0 DoD: raw camera read FPS, no model."""
    duration = args.duration or 10.0
    frames = 0
    with _camera_from(config, args) as cam:
        print("[camera]", cam.info)
        t_end = time.perf_counter() + duration
        while time.perf_counter() < t_end:
            if cam.read() is None:
                print("[camera] read failed / end of stream")
                break
            frames += 1
    print(f"[bench-raw] {frames} frames in ~{duration:.1f}s -> "
          f"{frames / duration:.1f} FPS (read only)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="roam-count -- people counter (moving camera)")
    p.add_argument("--config", default="config.json", help="path to config.json")
    p.add_argument("--source", default=None,
                   help="override camera source (index, file path, or rtsp url)")
    p.add_argument("--imgsz", type=int, default=None, help="override detector input size")
    p.add_argument("--detect-only", action="store_true",
                   help="Phase 1: detection only, no tracking / unique count")
    p.add_argument("--bench", action="store_true", help="headless inference-FPS benchmark")
    p.add_argument("--sweep", action="store_true", help="with --bench: test imgsz 640/480/320")
    p.add_argument("--iters", type=int, default=100, help="benchmark iterations (default 100)")
    p.add_argument("--bench-raw", action="store_true", help="Phase 0: raw camera read FPS")
    p.add_argument("--no-window", action="store_true", help="live mode without a display window")
    p.add_argument("--no-log", action="store_true", help="do not write the JSONL event log")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds to run (bench-raw / headless live)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.bench_raw:
        return run_bench_raw(config, args)
    if args.bench:
        return run_bench(config, args)
    if args.detect_only:
        return run_detect_only(config, args)
    return run_track(config, args)


if __name__ == "__main__":
    raise SystemExit(main())

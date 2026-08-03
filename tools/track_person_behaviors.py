import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


Box = Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    box: Box
    last_frame: int
    missed: int = 0
    observations: List[dict] = field(default_factory=list)


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def center_distance_score(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    norm_w = max(aw, bw)
    norm_h = max(ah, bh)
    dx = abs(acx - bcx) / norm_w
    dy = abs(acy - bcy) / norm_h
    distance = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - distance)


class SimpleBehaviorTracker:
    def __init__(self, iou_threshold: float = 0.25, center_threshold: float = 0.35, max_missed: int = 30):
        self.iou_threshold = iou_threshold
        self.center_threshold = center_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.finished_tracks: Dict[int, Track] = {}

    def _match_score(self, track_box: Box, det_box: Box) -> float:
        overlap = iou(track_box, det_box)
        center_score = center_distance_score(track_box, det_box)
        if overlap >= self.iou_threshold:
            return 1.0 + overlap
        if center_score >= self.center_threshold:
            return center_score
        return 0.0

    def update(self, detections: List[dict], frame_index: int, timestamp: float) -> List[dict]:
        matches = []
        used_tracks = set()
        used_dets = set()

        candidates = []
        for tid, track in self.tracks.items():
            for det_index, det in enumerate(detections):
                score = self._match_score(track.box, det["box"])
                if score > 0:
                    candidates.append((score, tid, det_index))

        for score, tid, det_index in sorted(candidates, reverse=True):
            if tid in used_tracks or det_index in used_dets:
                continue
            track = self.tracks[tid]
            det = detections[det_index]
            track.box = det["box"]
            track.last_frame = frame_index
            track.missed = 0
            obs = {
                "person_id": tid,
                "frame": frame_index,
                "time_sec": round(timestamp, 3),
                "behavior": det["class_name"],
                "confidence": round(det["confidence"], 4),
                "box": [round(v, 2) for v in det["box"]],
            }
            track.observations.append(obs)
            det["person_id"] = tid
            matches.append(det)
            used_tracks.add(tid)
            used_dets.add(det_index)

        for det_index, det in enumerate(detections):
            if det_index in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            obs = {
                "person_id": tid,
                "frame": frame_index,
                "time_sec": round(timestamp, 3),
                "behavior": det["class_name"],
                "confidence": round(det["confidence"], 4),
                "box": [round(v, 2) for v in det["box"]],
            }
            self.tracks[tid] = Track(track_id=tid, box=det["box"], last_frame=frame_index, observations=[obs])
            det["person_id"] = tid
            matches.append(det)

        for tid in list(self.tracks):
            if tid in used_tracks:
                continue
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                self.finished_tracks[tid] = self.tracks.pop(tid)

        return matches

    def all_tracks(self) -> Dict[int, Track]:
        merged = dict(self.finished_tracks)
        merged.update(self.tracks)
        return merged


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def extract_detections(result, allowed_classes: Optional[set]) -> List[dict]:
    detections = []
    names = result.names
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = str(names[cls_id])
        if allowed_classes and class_name not in allowed_classes:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "class_name": class_name,
            "confidence": conf,
            "box": (x1, y1, x2, y2),
        })
    return detections


def draw_tracks(frame, tracked_dets: List[dict]):
    for det in tracked_dets:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        person_id = det["person_id"]
        label = f"P{person_id} {det['class_name']} {det['confidence']:.2f}"
        color = ((person_id * 37) % 255, (person_id * 67) % 255, (person_id * 97) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def write_outputs(output_dir: Path, tracks: Dict[int, Track]):
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "person_behavior_timeline.csv"
    summary_path = output_dir / "person_behavior_summary.csv"
    json_path = output_dir / "person_behavior_summary.json"

    all_rows = []
    summary_rows = []
    json_summary = []

    for person_id, track in sorted(tracks.items()):
        if not track.observations:
            continue
        observations = sorted(track.observations, key=lambda item: item["frame"])
        all_rows.extend(observations)
        counts = Counter(obs["behavior"] for obs in observations)
        dominant_behavior, dominant_count = counts.most_common(1)[0]
        avg_conf = sum(float(obs["confidence"]) for obs in observations) / len(observations)
        first_time = observations[0]["time_sec"]
        last_time = observations[-1]["time_sec"]
        summary = {
            "person_id": person_id,
            "dominant_behavior": dominant_behavior,
            "dominant_count": dominant_count,
            "total_detections": len(observations),
            "first_time_sec": first_time,
            "last_time_sec": last_time,
            "avg_confidence": round(avg_conf, 4),
            "behavior_counts": dict(counts),
        }
        summary_rows.append(summary)
        json_summary.append(summary)

    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "frame", "time_sec", "behavior", "confidence", "box"])
        writer.writeheader()
        for row in all_rows:
            row = dict(row)
            row["box"] = json.dumps(row["box"], ensure_ascii=False)
            writer.writerow(row)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "person_id",
                "dominant_behavior",
                "dominant_count",
                "total_detections",
                "first_time_sec",
                "last_time_sec",
                "avg_confidence",
                "behavior_counts",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["behavior_counts"] = json.dumps(row["behavior_counts"], ensure_ascii=False)
            writer.writerow(row)

    json_path.write_text(json.dumps(json_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path, timeline_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Track per-person classroom behaviors in one video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default="object_cheating/models/tile_7.pt", help="YOLO behavior model path.")
    parser.add_argument("--output", default="runs/person_behavior_tracking", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO image size.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--track-iou", type=float, default=0.25, help="Tracker IoU match threshold.")
    parser.add_argument("--center-threshold", type=float, default=0.35, help="Tracker center-distance match threshold.")
    parser.add_argument("--max-missed", type=int, default=30, help="Frames to keep a missing person track.")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional behavior names to keep.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    video_path = resolve_path(project_root, args.video)
    model_path = resolve_path(project_root, args.model)
    output_dir = resolve_path(project_root, args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    model = YOLO(str(model_path))
    tracker = SimpleBehaviorTracker(args.track_iou, args.center_threshold, args.max_missed)
    allowed_classes = set(args.classes) if args.classes else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = output_dir / "person_behavior_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    frame_index = 0
    processed_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % args.frame_stride != 0:
            if writer is not None:
                writer.write(frame)
            continue

        timestamp = frame_index / fps
        predict_kwargs = {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device
        result = model(frame, **predict_kwargs)[0]
        detections = extract_detections(result, allowed_classes)
        tracked = tracker.update(detections, frame_index, timestamp)
        processed_count += 1

        if writer is not None:
            writer.write(draw_tracks(frame, tracked))

        if processed_count % 50 == 0:
            print(f"Processed frames: {processed_count}, active tracks: {len(tracker.tracks)}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video saved to: {out_video_path}")

    summary_path, timeline_path, json_path = write_outputs(output_dir, tracker.all_tracks())
    print(f"Summary saved to: {summary_path}")
    print(f"Timeline saved to: {timeline_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()

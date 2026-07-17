import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "slbd_yolov5"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


class SLBDYoloV5Detector:
    """YOLOv5-style detector wrapper for the SLBD student behavior model."""

    DEFAULT_MODEL_RELATIVE_PATH = "object_cheating/models/SLBD_model.pt"
    DEFAULT_IMAGE_SIZE = 1280

    COLOR_MAP: Dict[str, tuple[int, int, int]] = {
        "hand_raising": (71, 99, 255),
        "reading": (0, 252, 124),
        "writing": (235, 99, 37),
        "using_phone": (0, 191, 255),
        "bowing_head": (211, 0, 148),
        "leaning_over_table": (255, 255, 0),
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        self.model_path = Path(model_path or self.DEFAULT_MODEL_RELATIVE_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SLBD model file not found: {self.model_path}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = attempt_load(str(self.model_path), map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, s=self.stride)
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names

        if self.device.type != "cpu":
            warmup = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            warmup = warmup.half() if self.half else warmup.float()
            with torch.no_grad():
                self.model(warmup)

    @classmethod
    def _color_for_label(cls, label: str) -> tuple[int, int, int]:
        return cls.COLOR_MAP.get(label, (255, 255, 255))

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        selected_targets: List[str] | None = None,
    ):
        start_time = time.time()
        active_targets = list(selected_targets or [])

        image = letterbox(frame, self.image_size, stride=self.stride, auto=False)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(tensor)[0]
        predictions = non_max_suppression(
            prediction,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
        )

        processed_frame = frame.copy()
        total_detections = 0
        highest_conf = 0.0
        highest_class = "N/A"
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
        all_detections: List[Dict[str, object]] = []

        for det in predictions:
            if len(det) == 0:
                continue

            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_name = str(self.names[int(cls)])
                if active_targets and class_name not in active_targets:
                    continue

                x1, y1, x2, y2 = [int(v.item()) if hasattr(v, "item") else int(v) for v in xyxy]
                confidence = float(conf)
                total_detections += 1

                detection = {
                    "class_name": class_name,
                    "conf": confidence,
                    "coords": {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                    },
                }
                all_detections.append(detection)

                if confidence > highest_conf:
                    highest_conf = confidence
                    highest_class = class_name
                    coords = {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                    }

                color = self._color_for_label(class_name)
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    processed_frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        process_time = round(time.time() - start_time, 1)
        print(
            f"SLBD Model 6 predict: detections={total_detections}, "
            f"highest={highest_class}, confidence={round(highest_conf * 100)}%, "
            f"conf_thres={confidence_threshold}, iou_thres={iou_threshold}"
        )
        return (
            processed_frame,
            total_detections,
            process_time,
            highest_class,
            round(highest_conf * 100),
            coords,
            all_detections,
        )

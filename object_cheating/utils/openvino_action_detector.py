from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import openvino as ov
except ImportError:  # pragma: no cover - handled at runtime in the UI
    ov = None


class OpenVINOActionDetector:
    """Open Model Zoo Smart Classroom person/action detector."""

    LABELS = [
        "sitting",
        "writing",
        "raising_hand",
        "standing",
        "turned_around",
        "lie_on_the_desk",
    ]

    MODEL_RELATIVE_PATH = Path(
        "object_cheating/models/open_model_zoo/"
        "person-detection-action-recognition-0006/FP16/"
        "person-detection-action-recognition-0006.xml"
    )

    INPUT_HEIGHT = 400
    INPUT_WIDTH = 680
    DETECTION_CONF_NAME = "ActionNet/out_detection_conf"
    DETECTION_LOC_NAME = "ActionNet/out_detection_loc"
    ACTION_HEAD_NAMES = [
        "ActionNet/action_heads/out_head_1_anchor_1",
        "ActionNet/action_heads/out_head_2_anchor_1",
        "ActionNet/action_heads/out_head_2_anchor_2",
        "ActionNet/action_heads/out_head_2_anchor_3",
        "ActionNet/action_heads/out_head_2_anchor_4",
    ]
    HEADS = [
        {
            "step": 8,
            "anchors": [(26.17863728, 58.670372)],
            "shape": (50, 85),
        },
        {
            "step": 16,
            "anchors": [
                (35.36, 81.829632),
                (45.8114572, 107.651852),
                (63.31491832, 142.595732),
                (93.5070856, 201.107692),
            ],
            "shape": (25, 43),
        },
    ]
    VARIANCES = (0.1, 0.1, 0.2, 0.2)

    def __init__(self, model_path: str | Path | None = None, device: str = "CPU"):
        if ov is None:
            raise RuntimeError(
                "OpenVINO is not installed. Install it with `pip install openvino` "
                "before using Model 4."
            )

        self.model_path = Path(model_path or self.MODEL_RELATIVE_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Open Model Zoo model file not found: {self.model_path}"
            )

        self.core = ov.Core()
        self.model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_name = self.compiled_model.input(0).get_any_name()
        self.output_map = {
            output.get_any_name().split(":")[0]: output
            for output in self.compiled_model.outputs
        }
        self.head_ranges = self._build_head_ranges()

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.7,
        selected_target: str = "All",
        selected_targets: List[str] | None = None,
        action_confidence_threshold: float = 0.75,
    ):
        start_time = time.time()
        detections = self._infer(
            frame,
            confidence_threshold,
            action_confidence_threshold,
        )

        active_targets = self._normalize_targets(selected_target, selected_targets)
        if active_targets:
            detections = [
                detection
                for detection in detections
                if detection["class_name"] in active_targets
            ]

        detections = self._nms(detections, iou_threshold)
        processed_frame = frame.copy()
        highest_class = "N/A"
        highest_conf = 0.0
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}

        for detection in detections:
            det_coords = detection["coords"]
            color = self._color_for_label(detection["class_name"])
            left = det_coords["xmin"]
            top = det_coords["ymin"]
            right = det_coords["xmax"]
            bottom = det_coords["ymax"]
            confidence = float(detection["conf"])

            if confidence > highest_conf:
                highest_conf = confidence
                highest_class = str(detection["class_name"])
                coords = det_coords

            label = f"{detection['class_name']} {confidence * 100:.1f}%"
            cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                processed_frame,
                label,
                (left, max(20, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        process_time = round(time.time() - start_time, 1)
        return (
            processed_frame,
            len(detections),
            process_time,
            highest_class,
            round(highest_conf * 100),
            coords,
            detections,
        )

    @staticmethod
    def _normalize_targets(
        selected_target: str = "All",
        selected_targets: List[str] | None = None,
    ) -> List[str]:
        targets = list(selected_targets or [])
        if not targets and selected_target and selected_target != "All":
            targets = [selected_target]
        if not targets or "All" in targets:
            return []
        return targets

    def _infer(
        self,
        frame: np.ndarray,
        detection_confidence_threshold: float,
        action_confidence_threshold: float,
    ):
        original_height, original_width = frame.shape[:2]
        input_frame = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        input_tensor = np.expand_dims(input_frame.astype(np.float32), axis=0)
        raw_result = self.compiled_model({self.input_name: input_tensor})

        loc = raw_result[self.output_map[self.DETECTION_LOC_NAME]][0]
        det_conf = raw_result[self.output_map[self.DETECTION_CONF_NAME]][0]
        action_heads = [
            raw_result[self.output_map[name]][0] for name in self.ACTION_HEAD_NAMES
        ]

        detections = []
        for prior_index in range(loc.shape[0]):
            detection_conf = float(det_conf[prior_index, 1])
            if detection_conf < detection_confidence_threshold:
                continue

            head_id, head_pos, anchor_id = self._locate_anchor(prior_index)
            action_label, action_conf = self._decode_action(
                action_heads, head_id, head_pos, anchor_id
            )
            if action_conf < action_confidence_threshold:
                continue

            prior_box = self._generate_prior_box(head_id, head_pos, anchor_id)
            box = self._decode_box(
                prior_box,
                loc[prior_index],
                original_width,
                original_height,
            )
            if box is None:
                continue

            class_name = self.LABELS[action_label]
            confidence = min(detection_conf, action_conf)
            detections.append(
                {
                    "class_name": class_name,
                    "conf": confidence,
                    "coords": box,
                }
            )

        return detections

    def _decode_action(self, action_heads, head_id: int, head_pos: int, anchor_id: int):
        head = self.HEADS[head_id]
        _, blob_width = head["shape"]
        row = head_pos // blob_width
        col = head_pos % blob_width
        action_logits = action_heads[self._global_anchor_id(head_id, anchor_id)][row, col]
        scaled_logits = action_logits.astype(np.float64) * 16.0
        scaled_logits -= np.max(scaled_logits)
        probs = np.exp(scaled_logits)
        probs /= np.sum(probs)
        label = int(np.argmax(probs))
        return label, float(probs[label])

    def _decode_box(
        self,
        prior_box: Tuple[float, float, float, float],
        encoded_box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, int] | None:
        prior_xmin, prior_ymin, prior_xmax, prior_ymax = prior_box
        prior_width = prior_xmax - prior_xmin
        prior_height = prior_ymax - prior_ymin
        prior_center_x = 0.5 * (prior_xmin + prior_xmax)
        prior_center_y = 0.5 * (prior_ymin + prior_ymax)

        decoded_center_x = (
            self.VARIANCES[0] * float(encoded_box[0]) * prior_width + prior_center_x
        )
        decoded_center_y = (
            self.VARIANCES[1] * float(encoded_box[1]) * prior_height + prior_center_y
        )
        decoded_width = np.exp(self.VARIANCES[2] * float(encoded_box[2])) * prior_width
        decoded_height = (
            np.exp(self.VARIANCES[3] * float(encoded_box[3])) * prior_height
        )

        left = int((decoded_center_x - 0.5 * decoded_width) * frame_width)
        top = int((decoded_center_y - 0.5 * decoded_height) * frame_height)
        right = int((decoded_center_x + 0.5 * decoded_width) * frame_width)
        bottom = int((decoded_center_y + 0.5 * decoded_height) * frame_height)

        left = max(0, min(frame_width - 1, left))
        top = max(0, min(frame_height - 1, top))
        right = max(0, min(frame_width - 1, right))
        bottom = max(0, min(frame_height - 1, bottom))
        if right <= left or bottom <= top:
            return None
        return {"xmin": left, "ymin": top, "xmax": right, "ymax": bottom}

    def _generate_prior_box(self, head_id: int, head_pos: int, anchor_id: int):
        head = self.HEADS[head_id]
        blob_height, blob_width = head["shape"]
        del blob_height
        row = head_pos // blob_width
        col = head_pos % blob_width
        anchor_width, anchor_height = head["anchors"][anchor_id]

        center_x = (col + 0.5) * head["step"]
        center_y = (row + 0.5) * head["step"]
        return (
            (center_x - 0.5 * anchor_width) / self.INPUT_WIDTH,
            (center_y - 0.5 * anchor_height) / self.INPUT_HEIGHT,
            (center_x + 0.5 * anchor_width) / self.INPUT_WIDTH,
            (center_y + 0.5 * anchor_height) / self.INPUT_HEIGHT,
        )

    def _locate_anchor(self, prior_index: int):
        head_id = 0
        while prior_index >= self.head_ranges[head_id + 1]:
            head_id += 1
        head_pos_with_anchor = prior_index - self.head_ranges[head_id]
        anchor_count = len(self.HEADS[head_id]["anchors"])
        anchor_id = head_pos_with_anchor % anchor_count
        head_pos = head_pos_with_anchor // anchor_count
        return head_id, head_pos, anchor_id

    def _build_head_ranges(self):
        ranges = [0]
        total = 0
        for head in self.HEADS:
            height, width = head["shape"]
            total += height * width * len(head["anchors"])
            ranges.append(total)
        return ranges

    @staticmethod
    def _global_anchor_id(head_id: int, anchor_id: int):
        return anchor_id if head_id == 0 else 1 + anchor_id

    @staticmethod
    def _nms(detections: List[Dict[str, object]], iou_threshold: float):
        if not detections:
            return []

        kept_detections = []
        for class_name in sorted({str(item["class_name"]) for item in detections}):
            class_detections = [
                detection
                for detection in detections
                if detection["class_name"] == class_name
            ]
            boxes = []
            scores = []
            for detection in class_detections:
                coords = detection["coords"]
                left = int(coords["xmin"])
                top = int(coords["ymin"])
                width = int(coords["xmax"]) - left
                height = int(coords["ymax"]) - top
                boxes.append([left, top, width, height])
                scores.append(float(detection["conf"]))

            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)
            if len(indices) == 0:
                continue
            flat_indices = np.array(indices).reshape(-1).tolist()
            kept_detections.extend(class_detections[index] for index in flat_indices)

        return kept_detections

    @staticmethod
    def _color_for_label(label: str):
        color_map = {
            "sitting": (0, 252, 124),
            "writing": (255, 167, 38),
            "raising_hand": (71, 99, 255),
            "standing": (255, 255, 0),
            "turned_around": (238, 130, 238),
            "lie_on_the_desk": (60, 76, 231),
        }
        return color_map.get(label, (255, 255, 255))

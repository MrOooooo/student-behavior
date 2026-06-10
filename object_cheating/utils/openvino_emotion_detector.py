from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

try:
    import openvino as ov
except ImportError:  # pragma: no cover - handled at runtime in the UI
    ov = None


class OpenVINOEmotionDetector:
    """Open Model Zoo face detector + emotion recognition pipeline."""

    LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

    FACE_MODEL_RELATIVE_PATH = Path(
        "object_cheating/models/open_model_zoo/"
        "face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
    )
    EMOTION_MODEL_RELATIVE_PATH = Path(
        "object_cheating/models/open_model_zoo/"
        "emotions-recognition-retail-0003/FP16/"
        "emotions-recognition-retail-0003.xml"
    )

    def __init__(
        self,
        face_model_path: str | Path | None = None,
        emotion_model_path: str | Path | None = None,
        device: str = "CPU",
    ):
        if ov is None:
            raise RuntimeError(
                "OpenVINO is not installed. Install it with `pip install openvino` "
                "before using Model 5."
            )

        self.face_model_path = Path(face_model_path or self.FACE_MODEL_RELATIVE_PATH)
        self.emotion_model_path = Path(
            emotion_model_path or self.EMOTION_MODEL_RELATIVE_PATH
        )
        if not self.face_model_path.exists():
            raise FileNotFoundError(f"Face model file not found: {self.face_model_path}")
        if not self.emotion_model_path.exists():
            raise FileNotFoundError(
                f"Emotion model file not found: {self.emotion_model_path}"
            )

        self.core = ov.Core()
        self.face_model = self.core.read_model(str(self.face_model_path))
        self.emotion_model = self.core.read_model(str(self.emotion_model_path))
        self.compiled_face_model = self.core.compile_model(self.face_model, device)
        self.compiled_emotion_model = self.core.compile_model(self.emotion_model, device)

        self.face_input = self.compiled_face_model.input(0)
        self.face_output = self.compiled_face_model.output(0)
        self.face_input_name = self.face_input.get_any_name()
        self.face_input_height, self.face_input_width = self._input_hw(self.face_input)

        self.emotion_input = self.compiled_emotion_model.input(0)
        self.emotion_output = self.compiled_emotion_model.output(0)
        self.emotion_input_name = self.emotion_input.get_any_name()
        self.emotion_input_height, self.emotion_input_width = self._input_hw(
            self.emotion_input
        )

    def predict(
        self,
        frame: np.ndarray,
        face_confidence_threshold: float = 0.5,
        emotion_confidence_threshold: float = 0.35,
        selected_target: str = "All",
        selected_targets: List[str] | None = None,
    ):
        start_time = time.time()
        active_targets = self._normalize_targets(selected_target, selected_targets)
        detections = self._infer(
            frame,
            face_confidence_threshold,
            emotion_confidence_threshold,
        )

        if active_targets:
            detections = [
                detection
                for detection in detections
                if detection["class_name"] in active_targets
            ]

        processed_frame = frame.copy()
        highest_class = "N/A"
        highest_conf = 0.0
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}

        for detection in detections:
            det_coords = detection["coords"]
            color = self._color_for_label(str(detection["class_name"]))
            left = int(det_coords["xmin"])
            top = int(det_coords["ymin"])
            right = int(det_coords["xmax"])
            bottom = int(det_coords["ymax"])
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

    def _infer(
        self,
        frame: np.ndarray,
        face_confidence_threshold: float,
        emotion_confidence_threshold: float,
    ) -> List[Dict[str, object]]:
        frame_height, frame_width = frame.shape[:2]
        face_input = self._preprocess(frame, self.face_input_width, self.face_input_height)
        face_result = self.compiled_face_model({self.face_input_name: face_input})
        raw_faces = face_result[self.face_output]

        detections: List[Dict[str, object]] = []
        for raw_face in np.squeeze(raw_faces).reshape(-1, 7):
            confidence = float(raw_face[2])
            if confidence < face_confidence_threshold:
                continue

            xmin = int(max(0, min(frame_width - 1, raw_face[3] * frame_width)))
            ymin = int(max(0, min(frame_height - 1, raw_face[4] * frame_height)))
            xmax = int(max(0, min(frame_width - 1, raw_face[5] * frame_width)))
            ymax = int(max(0, min(frame_height - 1, raw_face[6] * frame_height)))
            if xmax <= xmin or ymax <= ymin:
                continue

            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                continue

            emotion_label, emotion_confidence = self._predict_emotion(face_crop)
            if emotion_confidence < emotion_confidence_threshold:
                continue

            detections.append(
                {
                    "class_name": emotion_label,
                    "conf": emotion_confidence,
                    "face_conf": confidence,
                    "coords": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    },
                }
            )

        return detections

    def _predict_emotion(self, face_crop: np.ndarray):
        emotion_input = self._preprocess(
            face_crop,
            self.emotion_input_width,
            self.emotion_input_height,
        )
        emotion_result = self.compiled_emotion_model(
            {self.emotion_input_name: emotion_input}
        )
        probabilities = np.array(emotion_result[self.emotion_output]).reshape(-1)
        class_index = int(np.argmax(probabilities))
        return self.LABELS[class_index], float(probabilities[class_index])

    @staticmethod
    def _preprocess(frame: np.ndarray, width: int, height: int):
        resized = cv2.resize(frame, (width, height))
        return np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32), axis=0)

    @staticmethod
    def _input_hw(input_port):
        shape = list(input_port.shape)
        if len(shape) != 4:
            raise ValueError(f"Unsupported OpenVINO input shape: {shape}")
        return int(shape[2]), int(shape[3])

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

    @staticmethod
    def _color_for_label(label: str):
        color_map = {
            "neutral": (0, 252, 124),
            "happy": (0, 215, 255),
            "sad": (255, 144, 30),
            "surprise": (255, 255, 0),
            "anger": (60, 76, 231),
        }
        return color_map.get(label, (255, 255, 255))

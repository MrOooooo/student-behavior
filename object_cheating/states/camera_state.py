import reflex as rx
from typing import TypedDict, List, ClassVar
import cv2
import base64
import numpy as np
import asyncio
import os
import time
import re
from datetime import datetime
from pathlib import Path
import tempfile
import csv
import json
import math
from collections import Counter
from typing import List, Dict
from object_cheating.utils.eye_tracker import EyeTracker
from object_cheating.utils.openvino_action_detector import OpenVINOActionDetector
from object_cheating.utils.openvino_emotion_detector import OpenVINOEmotionDetector
from object_cheating.utils.slbd_yolov5_detector import SLBDYoloV5Detector
from object_cheating.utils.face_identity import get_registry, extract_face_encoding
from ultralytics import YOLO
from object_cheating.states.threshold_state import ThresholdState

class DetectionResult(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int

class CameraState(ThresholdState):
    # Model state
    active_model: int = 1  # Contoh definisi state variable
    right_panel_collapsed: bool = False
    
    # Stats panel
    detection_count: int = 0
    processing_time: float = 0.0
    
    # Behaviour panel
    highest_confidence_class: str = "N/A"
    highest_confidence: float = 0.0
    
    # Coordinate panel
    highest_conf_xmin: int = 0
    highest_conf_ymin: int = 0
    highest_conf_xmax: int = 0
    highest_conf_ymax: int = 0
    
    # Table panel
    table_data: List[Dict[str, str]] = []
    table_entry_counter: int = 0
    _person_tracking_state: ClassVar[Dict[int, Dict[str, object]]] = {}
    _person_behavior_log: ClassVar[Dict[int, Dict[int, List[Dict[str, object]]]]] = {}
    _person_identity_map: ClassVar[Dict[int, str]] = {}  # session-local person_id → global identity_id
    _person_track_max_missed: ClassVar[int] = 15
    PERSON_MIN_HITS_TO_PERSIST: ClassVar[int] = 3  # tracks need 3+ hits before getting permanent ID
    PERSON_TRACK_NMS_IOU: ClassVar[float] = 0.30  # merge near-duplicate tracks
    PERSON_MATCH_MIN_SCORE: ClassVar[float] = 0.48
    PERSON_CENTER_MIN_SCORE: ClassVar[float] = 0.42
    PERSON_SIZE_MIN_SIMILARITY: ClassVar[float] = 0.35
    PERSON_TRACK_SMOOTHING: ClassVar[float] = 0.65
    PERSON_BODY_BINDING_ENABLED: ClassVar[bool] = True
    PERSON_SHARED_ACROSS_MODELS: ClassVar[bool] = True
    PERSON_BODY_CONF_THRESHOLD: ClassVar[float] = 0.20
    PERSON_BODY_ACTION_CONF_THRESHOLD: ClassVar[float] = 0.0
    PERSON_BODY_NMS_IOU: ClassVar[float] = 0.45
    PERSON_BODY_ASSOC_MIN_SCORE: ClassVar[float] = 0.18
    PERSON_CONTAINMENT_BONUS: ClassVar[float] = 0.35
    PERSON_ANCHOR_IOU_WEIGHT: ClassVar[float] = 0.58
    PERSON_CROSS_MODEL_SIZE_MIN: ClassVar[float] = 0.12
    _cached_body_boxes: ClassVar[List[Dict] | None] = None
    _cached_body_frame_index: ClassVar[int] = -1
    MODEL7_SLICED_ENABLED: ClassVar[bool] = True
    MODEL7_SLICED_IMGSZ: ClassVar[int] = 960
    MODEL7_TILE_GRID: ClassVar[tuple[int, int]] = (3, 3)
    MODEL7_TILE_OVERLAP: ClassVar[float] = 0.15
    MODEL7_TILE_NMS_IOU: ClassVar[float] = 0.5
    MODEL7_FULL_FRAME_FUSION: ClassVar[bool] = True
    
    # Add table color mapping
    table_color_map: Dict[str, str] = {
        "cheating": "tomato",
        "left": "orange",
        "right": "orange",
        "Look Around": "violet",
        "Normal": "grass",
        "normal": "grass",
        "center": "green",
        "Bend Over The Desk": "cyan",
        "Hand Under Table": "indigo",
        "Stand Up": "sky",
        "Wave": "pink",
        "sitting": "grass",
        "writing": "blue",
        "raising_hand": "tomato",
        "standing": "amber",
        "turned_around": "violet",
        "lie_on_the_desk": "crimson",
        "neutral": "grass",
        "happy": "amber",
        "sad": "blue",
        "surprise": "yellow",
        "anger": "tomato",
        "hand_raising": "tomato",
        "reading": "grass",
        "using_phone": "amber",
        "bowing_head": "violet",
        "leaning_over_table": "cyan",
    }
    
    # Constants for frame capture
    FRAME_CAPTURE_INTERVAL = 10  # Capture every 10th frame
    MAX_SAVES_PER_MINUTE = 6  # Maximum 6 saves per minute (1 every 10 seconds)
    
    # Add timestamp tracking for rate limiting
    _last_save_time: float = 0
    @staticmethod
    def _upload_dir() -> str:
        """Return a writable absolute upload directory and ensure it exists."""
        upload_dir = Path(tempfile.gettempdir()) / "student_behavior_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        return str(upload_dir)

    @staticmethod
    def _safe_upload_name(original_name: str) -> str:
        """Make uploaded filenames unique and filesystem-safe."""
        stem, ext = os.path.splitext(original_name)
        ext = ext.lower()
        if ext not in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}:
            ext = ".mp4"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        if not cleaned:
            cleaned = "video"
        return f"{cleaned}_{stamp}{ext}"
    
    @rx.event
    def prev_model(self):
        if self.active_model == 7:
            self.active_model = 5
        elif self.active_model > 1:
            self.active_model -= 1

    @rx.event
    def next_model(self):
        if self.active_model == 5:
            self.active_model = 7
        elif self.active_model < 5:
            self.active_model += 1

    @rx.event
    def toggle_right_panel(self):
        self.right_panel_collapsed = not self.right_panel_collapsed
            
    # Add new state variables for dialog
    show_warning_dialog: bool = False
    target_model: int = 0  # To store the model we want to switch to
    
    @rx.event
    async def try_change_model(self, target: int):
        """Try to change model, show warning if detection is enabled"""
        if self.detection_enabled:
            self.target_model = target
            self.show_warning_dialog = True
        else:
            # If detection is disabled, change model directly
            if target > self.active_model:
                self.next_model()
            else:
                self.prev_model()
                
            self.selected_target = "All"
            self.selected_targets = ["All"]
            # Set default thresholds for new model
            self.set_model_defaults(self.active_model)  # Use the actual model after skipping hidden models
                
    @rx.event
    async def close_warning_dialog(self):
        """Close the warning dialog without changing model"""
        self.show_warning_dialog = False
        self.target_model = 0
        
    # Add new state variables for delete dialog
    show_delete_dialog: bool = False
    
    @rx.event
    async def try_clear_camera(self):
        """Show confirmation dialog before clearing"""
        self.show_delete_dialog = True
    
    @rx.event
    async def confirm_clear(self):
        """Confirm and execute clear operation"""
        self.show_delete_dialog = False
        return CameraState.clear_camera
    
    @rx.event
    async def cancel_clear(self):
        """Cancel clear operation"""
        self.show_delete_dialog = False
            
    detection_enabled: bool = False
    eye_alerts: list[str] = []
    
    # Eye tracking state
    eye_alert_counter: int = 0
    eye_frame_counter: int = 0
    
    _original_frame_bytes: bytes = b""
    
    # Stream state
    camera_active: bool = False
    processing_active: bool = False
    current_frame: str = ""  # Base64 encoded image
    error_message: str = ""
    
    # Tambahkan state untuk upload gambar
    uploaded_image: str = ""  # Untuk menyimpan gambar yang diupload
    
    video_playing: bool = False
    video_path: str = ""
    # Face detection state
    face_detection_active: bool = False
    detection_results: List[DetectionResult] = []
    min_neighbors: int = 5
    scale_factor: float = 1.3
    
    # Performance metrics
    fps: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    face_count: int = 0
    
    # Model YOLO
    _yolo_model = None
    
    selected_target: str = "All"
    selected_targets: List[str] = ["All"]
    cross_model_enabled: bool = False
    model1_cross_targets: List[str] = []
    model2_cross_targets: List[str] = []
    model3_cross_targets: List[str] = []
    model4_cross_targets: List[str] = []
    model5_cross_targets: List[str] = []
    model6_cross_targets: List[str] = []
    model7_cross_targets: List[str] = []
    model8_cross_targets: List[str] = []
    
    # Add new YOLO model for Model 2
    _yolo_model_2 = None
    _yolo_model_6 = None
    _yolo_model_7 = None
    _yolo_model_8 = None
    _openvino_action_model = None
    _openvino_emotion_model = None
    _slbd_model = None
    
    @classmethod
    def get_yolo_model(cls):
        """Get or initialize YOLO model"""
        if cls._yolo_model is None:
            cls._yolo_model = YOLO("object_cheating/models/modelv11.pt")
        return cls._yolo_model
    
    @classmethod
    def get_yolo_model_2(cls):
        """Get or initialize YOLO model 2 for cheating detection"""
        if cls._yolo_model_2 is None:
            cls._yolo_model_2 = YOLO("object_cheating/models/modelv8-2.pt")
        return cls._yolo_model_2

    @classmethod
    def get_yolo_model_6(cls):
        """Get or initialize YOLOv5-style SLBD model for student behavior detection."""
        if cls._slbd_model is None:
            cls._slbd_model = SLBDYoloV5Detector("object_cheating/models/SLBD_model.pt")
        return cls._slbd_model

    @classmethod
    def get_yolo_model_7(cls):
        """Get or initialize trained SCB YOLO student behavior model."""
        if cls._yolo_model_7 is None:
            cls._yolo_model_7 = YOLO("object_cheating/models/tile_7.pt")
        return cls._yolo_model_7

    @classmethod
    def get_yolo_model_8(cls):
        """Get or initialize high-accuracy YOLO student behavior model."""
        if cls._yolo_model_8 is None:
            cls._yolo_model_8 = YOLO("object_cheating/models/high_147.pt")
        return cls._yolo_model_8

    @classmethod
    def get_openvino_action_model(cls):
        """Get or initialize Open Model Zoo person/action detector."""
        if cls._openvino_action_model is None:
            cls._openvino_action_model = OpenVINOActionDetector()
        return cls._openvino_action_model

    @classmethod
    def get_openvino_emotion_model(cls):
        """Get or initialize Open Model Zoo face/emotion detector."""
        if cls._openvino_emotion_model is None:
            cls._openvino_emotion_model = OpenVINOEmotionDetector()
        return cls._openvino_emotion_model

    @classmethod
    def get_class_color(cls, class_name: str) -> tuple:
        """Get color for each class in Model 1"""
        color_map = {
            "Normal": (0, 255, 128),        # Green
            "Bend Over The Desk": (255, 255, 0),    # Aqua
            "Hand Under Table": (255, 105, 65),      # Royal Blue
            "Look Around": (238, 130, 238),         # Violet
            "Stand Up": (250, 230, 230),           # Lavender
            "Wave": (193, 182, 255)                # Light Pink
        }
        return color_map.get(class_name, (0, 255, 128))  # Default to green if class not found

    @classmethod
    def get_model6_class_color(cls, class_name: str) -> tuple:
        color_map = {
            "hand_raising": (71, 99, 255),
            "reading": (0, 252, 124),
            "writing": (235, 99, 37),
            "using_phone": (0, 191, 255),
            "bowing_head": (211, 0, 148),
            "leaning_over_table": (255, 255, 0),
        }
        return color_map.get(class_name, (255, 255, 255))

    @classmethod
    def get_model7_class_color(cls, class_name: str) -> tuple:
        return cls.get_model6_class_color(class_name)

    @classmethod
    def get_model8_class_color(cls, class_name: str) -> tuple:
        color_map = {
            "leaning_over_table": (255, 255, 0),
            "Hand Under Table": (255, 105, 65),
            "Look Around": (238, 130, 238),
            "Normal": (0, 255, 128),
            "standing": (0, 215, 255),
            "Wave": (193, 182, 255),
            "sitting": (124, 252, 0),
            "writing": (235, 99, 37),
            "hand_raising": (71, 99, 255),
            "turned_around": (148, 0, 211),
            "lie_on_the_desk": (60, 20, 220),
            "reading": (0, 252, 124),
            "using_phone": (0, 191, 255),
            "bowing_head": (211, 0, 148),
        }
        return color_map.get(class_name, (255, 255, 255))

    def __init__(self, *args, **kwargs):
        """Initialize state with parent initialization."""
        super().__init__(*args, **kwargs)
        
    @rx.event
    def set_active_model(self, model_num: int):
        if model_num in (1, 2, 3, 4, 5, 7):
            self.active_model = model_num
            self.selected_target = "All"
            self.selected_targets = ["All"]
        else:
            print(f"模型编号无效: {model_num}。必须在 1 到 8 之间。")
        
    
    def get_face_cascade(self) -> cv2.CascadeClassifier:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @rx.event
    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            return CameraState.process_camera_feed
        else:
            self.current_frame = ""
            
    @property
    def original_frame(self) -> np.ndarray:
        """Convert bytes back to numpy array when needed"""
        if not self._original_frame_bytes:
            return None
        nparr = np.frombuffer(self._original_frame_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def set_original_frame(self, frame: np.ndarray):
        """Convert numpy array to bytes for storage"""
        if frame is None:
            self._original_frame_bytes = b""
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            self._original_frame_bytes = buffer.tobytes()
            
    @rx.event
    async def save_current_frame(self):
        """Save the current frame as an image."""
        try:
            if not self.current_frame:
                self.error_message = "No frame to save."
                return

            header, encoded = self.current_frame.split(",", 1)
            image_data = base64.b64decode(encoded)

            save_dir = os.path.join("saved_frames", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_data)
                
            return rx.toast.success(
                f"Frame saved to {filepath}.", position="bottom-right"
            )

        except Exception as e:
            self.error_message = f"Error saving frame: {str(e)}"
            
    @rx.event
    def set_selected_target(self, target: str):
        """Set the selected target class."""
        self.selected_target = target
        self.selected_targets = [target]

    @rx.event
    def toggle_selected_target(self, target: str, checked: bool):
        """Toggle one target class for multi-select filtering."""
        current_targets = list(self.selected_targets or ["All"])

        if target == "All":
            self.selected_targets = ["All"] if checked else []
            self.selected_target = "All" if checked else ""
            return

        if "All" in current_targets:
            current_targets.remove("All")

        if checked and target not in current_targets:
            current_targets.append(target)
        elif not checked and target in current_targets:
            current_targets.remove(target)

        self.selected_targets = current_targets or ["All"]
        self.selected_target = "All" if "All" in self.selected_targets else ",".join(self.selected_targets)

    def _active_selected_targets(self) -> List[str]:
        targets = list(self.selected_targets or [])
        if not targets or "All" in targets:
            return []
        return targets

    def _target_matches(self, class_name: str, targets: List[str] | None = None) -> bool:
        targets = self._active_selected_targets() if targets is None else targets
        return not targets or class_name in targets

    @rx.event
    def toggle_cross_model(self, enabled: bool):
        self.cross_model_enabled = enabled

    @rx.event
    def toggle_cross_model_target(self, model_num: int, target: str, checked: bool):
        current_targets = self._get_cross_targets(model_num)

        if target == "All":
            next_targets = ["All"] if checked else []
        else:
            next_targets = [item for item in current_targets if item != "All"]
            if checked and target not in next_targets:
                next_targets.append(target)
            elif not checked and target in next_targets:
                next_targets.remove(target)

        self._set_cross_targets(model_num, next_targets)

    def _get_cross_targets(self, model_num: int) -> List[str]:
        if model_num == 1:
            return list(self.model1_cross_targets or [])
        if model_num == 2:
            return list(self.model2_cross_targets or [])
        if model_num == 3:
            return list(self.model3_cross_targets or [])
        if model_num == 4:
            return list(self.model4_cross_targets or [])
        if model_num == 5:
            return list(self.model5_cross_targets or [])
        if model_num == 6:
            return list(self.model6_cross_targets or [])
        if model_num == 7:
            return list(self.model7_cross_targets or [])
        if model_num == 8:
            return list(self.model8_cross_targets or [])
        return []

    def _set_cross_targets(self, model_num: int, targets: List[str]):
        if model_num == 1:
            self.model1_cross_targets = targets
        elif model_num == 2:
            self.model2_cross_targets = targets
        elif model_num == 3:
            self.model3_cross_targets = targets
        elif model_num == 4:
            self.model4_cross_targets = targets
        elif model_num == 5:
            self.model5_cross_targets = targets
        elif model_num == 6:
            self.model6_cross_targets = targets
        elif model_num == 7:
            self.model7_cross_targets = targets
        elif model_num == 8:
            self.model8_cross_targets = targets

    def _active_cross_targets(self, model_num: int) -> List[str]:
        targets = self._get_cross_targets(model_num)
        if not targets:
            return []
        if "All" in targets:
            return ["All"]
        return targets

    def _normalize_model_targets(self, targets: List[str]) -> List[str]:
        if not targets or "All" in targets:
            return []
        return targets
            
    @staticmethod
    def _starts_for_axis(length: int, tile: int, overlap: float) -> List[int]:
        if tile >= length:
            return [0]
        stride = max(1, int(tile * (1.0 - overlap)))
        starts = list(range(0, max(1, length - tile + 1), stride))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return sorted(set(starts))

    @classmethod
    def _make_tile_windows(cls, width: int, height: int):
        cols, rows = cls.MODEL7_TILE_GRID
        tile_w = math.ceil(width / cols)
        tile_h = math.ceil(height / rows)
        for y1 in cls._starts_for_axis(height, tile_h, cls.MODEL7_TILE_OVERLAP):
            for x1 in cls._starts_for_axis(width, tile_w, cls.MODEL7_TILE_OVERLAP):
                x2 = min(width, x1 + tile_w)
                y2 = min(height, y1 + tile_h)
                yield x1, y1, x2, y2

    @staticmethod
    def _yolo_class_name(names, class_id: int) -> str:
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _box_iou_for_detection(a: Dict[str, object], b: Dict[str, object]) -> float:
        a_coords = a["coords"]
        b_coords = b["coords"]
        x1 = max(float(a_coords["xmin"]), float(b_coords["xmin"]))
        y1 = max(float(a_coords["ymin"]), float(b_coords["ymin"]))
        x2 = min(float(a_coords["xmax"]), float(b_coords["xmax"]))
        y2 = min(float(a_coords["ymax"]), float(b_coords["ymax"]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a_coords["xmax"]) - float(a_coords["xmin"])) * max(0.0, float(a_coords["ymax"]) - float(a_coords["ymin"]))
        area_b = max(0.0, float(b_coords["xmax"]) - float(b_coords["xmin"])) * max(0.0, float(b_coords["ymax"]) - float(b_coords["ymin"]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _nms_yolo_detections(cls, detections: List[Dict[str, object]], iou_threshold: float) -> List[Dict[str, object]]:
        kept: List[Dict[str, object]] = []
        by_class: Dict[int, List[Dict[str, object]]] = {}
        for det in detections:
            by_class.setdefault(int(det.get("class_id", -1)), []).append(det)

        for class_detections in by_class.values():
            pending = sorted(class_detections, key=lambda item: float(item.get("conf", 0.0)), reverse=True)
            while pending:
                best = pending.pop(0)
                kept.append(best)
                pending = [
                    candidate
                    for candidate in pending
                    if cls._box_iou_for_detection(best, candidate) < iou_threshold
                ]
        return sorted(kept, key=lambda item: float(item.get("conf", 0.0)), reverse=True)

    def _detections_from_yolo_result(self, result, offset_x: int, offset_y: int, active_targets: List[str]) -> List[Dict[str, object]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None)
        detections: List[Dict[str, object]] = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = self._yolo_class_name(names, class_id)
            if not self._target_matches(class_name, active_targets):
                continue
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "conf": conf,
                    "coords": {
                        "xmin": int(round(x1 + offset_x)),
                        "ymin": int(round(y1 + offset_y)),
                        "xmax": int(round(x2 + offset_x)),
                        "ymax": int(round(y2 + offset_y)),
                    },
                }
            )
        return detections

    def _apply_sliced_yolo_prediction(
        self,
        model,
        frame,
        selected_targets=None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        class_color_getter=None,
        model_num: int | None = None,
        frame_index: int | None = None,
    ):
        """Run Model 7 on overlapping tiles so small students in wide shots stay detectable."""
        start_time = time.time()
        active_targets = self._normalize_model_targets(
            self._active_selected_targets() if selected_targets is None else selected_targets
        )
        confidence_threshold = (
            self.confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        iou_threshold = self.iou_threshold if iou_threshold is None else iou_threshold

        height, width = frame.shape[:2]
        tile_windows = list(self._make_tile_windows(width, height))
        raw_detections: List[Dict[str, object]] = []

        # Full-frame fusion keeps large/near students from being cut by tile borders.
        if self.MODEL7_FULL_FRAME_FUSION:
            full_results = model.predict(
                source=frame,
                imgsz=self.MODEL7_SLICED_IMGSZ,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            for result in full_results:
                raw_detections.extend(
                    self._detections_from_yolo_result(result, 0, 0, active_targets)
                )

        for x1, y1, x2, y2 in tile_windows:
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            results = model.predict(
                source=tile,
                imgsz=self.MODEL7_SLICED_IMGSZ,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            for result in results:
                raw_detections.extend(
                    self._detections_from_yolo_result(result, x1, y1, active_targets)
                )

        all_detections = self._nms_yolo_detections(raw_detections, self.MODEL7_TILE_NMS_IOU)
        processed_frame = frame.copy()
        highest_conf = 0.0
        highest_class = "N/A"
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}

        for detection in all_detections:
            class_name = str(detection["class_name"])
            conf = float(detection["conf"])
            det_coords = detection["coords"]
            x1 = int(det_coords["xmin"])
            y1 = int(det_coords["ymin"])
            x2 = int(det_coords["xmax"])
            y2 = int(det_coords["ymax"])

            if conf > highest_conf:
                highest_conf = conf
                highest_class = class_name
                coords = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}

            color = class_color_getter(class_name) if class_color_getter is not None else self.get_model7_class_color(class_name)
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                processed_frame,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        process_time = round((time.time() - start_time), 1)
        if model_num is not None and all_detections:
            all_detections = self._assign_person_ids_to_detections(model_num, all_detections, frame_index, frame=frame)

        print(
            f"SCB Model {self._display_model_num(model_num or 7)} sliced: "
            f"tiles={len(tile_windows)}, full_frame={self.MODEL7_FULL_FRAME_FUSION}, detections={len(all_detections)}, "
            f"conf_thres={confidence_threshold}, iou_thres={iou_threshold}"
        )
        return processed_frame, len(all_detections), process_time, highest_class, round(highest_conf * 100), coords, all_detections

    def _apply_yolo_prediction(
        self,
        model,
        frame,
        is_model_1=True,
        selected_targets=None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        class_color_getter=None,
        model_num: int | None = None,
        frame_index: int | None = None,
    ):
        """Helper method to apply YOLO prediction with current thresholds"""
        start_time = time.time()
        active_targets = self._normalize_model_targets(
            self._active_selected_targets() if selected_targets is None else selected_targets
        )
        confidence_threshold = (
            self.confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        iou_threshold = self.iou_threshold if iou_threshold is None else iou_threshold
        
        if model_num == 7 and self.MODEL7_SLICED_ENABLED and not isinstance(model, SLBDYoloV5Detector):
            return self._apply_sliced_yolo_prediction(
                model,
                frame,
                selected_targets=active_targets,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                class_color_getter=class_color_getter,
                model_num=model_num,
                frame_index=frame_index,
            )
        
        if isinstance(model, SLBDYoloV5Detector):
            result = model.predict(
                frame,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                selected_targets=active_targets,
            )
            if model_num is not None and isinstance(result, tuple) and len(result) >= 7 and result[6]:
                tracked = self._assign_person_ids_to_detections(model_num, result[6], frame_index, frame=frame)
                return result[:6] + (tracked,)
            return result

        # Run prediction
        results = model(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold
        )
        
        processed_frame = frame.copy()
        total_detections = 0
        highest_conf = 0.0
        highest_class = "N/A"
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
        all_detections = []
        # First pass: Count all detections and draw boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = box.cls[0]
                class_name = model.names[int(cls)]
                
                # Filter detections based on selected target
                if not self._target_matches(class_name, active_targets):
                    continue
                
                total_detections += 1
                
                # Simpan setiap deteksi
                detection = {
                    "class_name": class_name,
                    "conf": conf,
                    "coords": {
                        "xmin": int(x1),
                        "ymin": int(y1),
                        "xmax": int(x2),
                        "ymax": int(y2),
                    }
                }
                all_detections.append(detection)
                
                # Track highest confidence detection
                if conf > highest_conf:
                    highest_conf = conf
                    highest_class = class_name
                    coords["xmin"] = int(x1)
                    coords["ymin"] = int(y1)
                    coords["xmax"] = int(x2)
                    coords["ymax"] = int(y2)
                
                # Draw detection regardless of selected target
                label = f"{class_name} {conf:.2f}"
                
                if class_color_getter is not None:
                    color = class_color_getter(class_name)
                elif is_model_1:
                    color = self.get_class_color(class_name)
                else:
                    color = (71, 99, 255) if class_name == "cheating" else (0, 252, 124)
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw bounding box and label
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate runtime
        end_time = time.time()
        process_time = round((end_time - start_time), 1)
        
        if model_num is not None and all_detections:
            all_detections = self._assign_person_ids_to_detections(model_num, all_detections, frame_index, frame=frame)

        print(total_detections)
        print(process_time)
        
        return processed_frame, total_detections, process_time, highest_class, round(highest_conf * 100), coords, all_detections

    def _apply_openvino_prediction(
        self,
        frame,
        selected_targets=None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        action_confidence_threshold: float | None = None,
        model_num: int | None = None,
        frame_index: int | None = None,
    ):
        """Run Open Model Zoo Smart Classroom person/action detection."""
        openvino_model = self.get_openvino_action_model()
        result = openvino_model.predict(
            frame,
            confidence_threshold=(
                self.confidence_threshold if confidence_threshold is None else confidence_threshold
            ),
            iou_threshold=self.iou_threshold if iou_threshold is None else iou_threshold,
            selected_targets=self._normalize_model_targets(
                self._active_selected_targets() if selected_targets is None else selected_targets
            ),
            action_confidence_threshold=(
                self.model4_action_confidence_threshold
                if action_confidence_threshold is None
                else action_confidence_threshold
            ),
        )
        if model_num is not None and isinstance(result, tuple) and len(result) >= 7 and result[6]:
            tracked = self._assign_person_ids_to_detections(model_num, result[6], frame_index, frame=frame)
            return result[:6] + (tracked,)
        return result

    def _apply_openvino_emotion_prediction(
        self,
        frame,
        selected_targets=None,
        face_confidence_threshold: float | None = None,
        emotion_confidence_threshold: float | None = None,
        model_num: int | None = None,
        frame_index: int | None = None,
    ):
        """Run Open Model Zoo face detection + emotion recognition."""
        emotion_model = self.get_openvino_emotion_model()
        result = emotion_model.predict(
            frame,
            face_confidence_threshold=(
                self.confidence_threshold
                if face_confidence_threshold is None
                else face_confidence_threshold
            ),
            emotion_confidence_threshold=(
                self.iou_threshold
                if emotion_confidence_threshold is None
                else emotion_confidence_threshold
            ),
            selected_targets=self._normalize_model_targets(
                self._active_selected_targets() if selected_targets is None else selected_targets
            ),
        )
        if model_num is not None and isinstance(result, tuple) and len(result) >= 7 and result[6]:
            tracked = self._assign_person_ids_to_detections(model_num, result[6], frame_index, frame=frame)
            self._link_face_identities(frame, tracked, frame_index or 0)
            return result[:6] + (tracked,)
        return result

    async def _update_detection_metrics(
        self,
        total_detections: int,
        process_time: float,
        highest_class: str,
        highest_conf: float,
        coords: Dict[str, int],
        fps: float | None = None,
    ):
        async with self:
            self.detection_count = total_detections
            self.processing_time = process_time
            if fps is not None:
                self.fps = fps
            self.highest_confidence_class = highest_class
            self.highest_confidence = highest_conf
            self.highest_conf_xmin = coords["xmin"]
            self.highest_conf_ymin = coords["ymin"]
            self.highest_conf_xmax = coords["xmax"]
            self.highest_conf_ymax = coords["ymax"]

    def _class_color_for_model(self, model_num: int, class_name: str):
        if model_num == 1:
            return self.get_class_color(class_name)
        if model_num == 2:
            return (71, 99, 255) if class_name == "cheating" else (0, 252, 124)
        if model_num == 4:
            return OpenVINOActionDetector._color_for_label(class_name)
        if model_num == 5:
            return OpenVINOEmotionDetector._color_for_label(class_name)
        if model_num == 6:
            return self.get_model6_class_color(class_name)
        if model_num == 7:
            return self.get_model7_class_color(class_name)
        if model_num == 8:
            return self.get_model8_class_color(class_name)
        return (255, 255, 255)

    def _draw_model_detections(self, frame, detections: List[Dict[str, object]], model_num: int):
        processed_frame = frame.copy()
        for detection in detections:
            coords = detection["coords"]
            left = int(coords["xmin"])
            top = int(coords["ymin"])
            right = int(coords["xmax"])
            bottom = int(coords["ymax"])
            class_name = str(detection["class_name"])
            confidence = float(detection.get("conf", 0.0))
            color = self._class_color_for_model(model_num, class_name)
            label = f"M{model_num}:{class_name} {confidence * 100:.1f}%"

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
        return processed_frame

    def _run_cross_model_detection(
        self,
        frame,
        model_cache: Dict[str, object] | None = None,
        eye_tracker: EyeTracker | None = None,
        is_video: bool = False,
        alert_counter: int = 0,
        frame_counter: int = 0,
    ):
        model_cache = model_cache if model_cache is not None else {}
        processed_frame = frame.copy()
        total_detections = 0
        total_process_time = 0.0
        highest_class = "N/A"
        highest_conf = 0.0
        highest_coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
        detections_by_model: Dict[int, List[Dict[str, object]]] = {}
        alerts: List[str] = []

        def update_highest(class_name: str, confidence: float, coords: Dict[str, int]):
            nonlocal highest_class, highest_conf, highest_coords
            if confidence > highest_conf:
                highest_class = class_name
                highest_conf = confidence
                highest_coords = coords

        # Run eye tracking first so later object/action boxes are drawn on top.
        model3_targets = self._active_cross_targets(3)
        if model3_targets:
            eye_targets = self._normalize_model_targets(model3_targets)
            eye_tracker = eye_tracker or model_cache.get("eye_tracker") or EyeTracker()
            model_cache["eye_tracker"] = eye_tracker
            (
                processed_frame,
                eye_alerts,
                model_detections,
                process_time,
                class_name,
                confidence,
                coords,
            ) = eye_tracker.process_eye_detections(
                frame,
                alert_counter,
                frame_counter,
                cnn_threshold=self.model3_confidence_threshold,
                duration_threshold=self.model3_duration_threshold,
                is_video=is_video,
                selected_targets=eye_targets,
            )
            total_detections += model_detections
            total_process_time += process_time
            alerts.extend(eye_alerts)
            if model_detections > 0:
                confidence_ratio = float(confidence) / 100.0
                eye_detections = [
                    {"class_name": class_name, "conf": confidence_ratio, "coords": coords}
                ]
                # Assign person IDs to eye tracking via shared tracker
                eye_detections = self._assign_person_ids_to_detections(
                    3, eye_detections, frame_index=self.frame_count, frame=frame
                )
                detections_by_model[3] = eye_detections
                update_highest(class_name, float(confidence), coords)

        # Run Model 4 first (after eye tracking) to establish body anchors
        # in the shared tracker. All other models then match against these anchors.
        model_order = (4, 1, 2, 5, 7)
        for model_num in model_order:
            model_targets = self._active_cross_targets(model_num)
            if not model_targets:
                continue

            active_targets = self._normalize_model_targets(model_targets)
            if model_num == 1:
                model = model_cache.get("model1") or self.get_yolo_model()
                model_cache["model1"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    True,
                    selected_targets=active_targets,
                    model_num=model_num,
                    confidence_threshold=self.model1_confidence_threshold,
                    iou_threshold=self.model1_iou_threshold,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
            elif model_num == 2:
                model = model_cache.get("model2") or self.get_yolo_model_2()
                model_cache["model2"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    False,
                    selected_targets=active_targets,
                    model_num=model_num,
                    confidence_threshold=self.model2_confidence_threshold,
                    iou_threshold=self.model2_iou_threshold,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
            elif model_num == 4:
                model = model_cache.get("model4") or self.get_openvino_action_model()
                model_cache["model4"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = model.predict(
                    frame,
                    confidence_threshold=self.model4_confidence_threshold,
                    iou_threshold=self.model4_iou_threshold,
                    selected_targets=active_targets,
                    action_confidence_threshold=self.model4_action_confidence_threshold,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
                    self._link_body_identities(frame, model_num)
            elif model_num == 5:
                model = model_cache.get("model5") or self.get_openvino_emotion_model()
                model_cache["model5"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = model.predict(
                    frame,
                    face_confidence_threshold=self.model5_face_confidence_threshold,
                    emotion_confidence_threshold=self.model5_emotion_confidence_threshold,
                    selected_targets=active_targets,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
                    self._link_face_identities(frame, detections, self.frame_count)
            elif model_num == 6:
                model = model_cache.get("model6") or self.get_yolo_model_6()
                model_cache["model6"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    False,
                    selected_targets=active_targets,
                    model_num=model_num,
                    confidence_threshold=self.model6_confidence_threshold,
                    iou_threshold=self.model6_iou_threshold,
                    class_color_getter=self.get_model6_class_color,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
            elif model_num == 7:
                model = model_cache.get("model7") or self.get_yolo_model_7()
                model_cache["model7"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    False,
                    selected_targets=active_targets,
                    model_num=model_num,
                    confidence_threshold=self.model7_confidence_threshold,
                    iou_threshold=self.model7_iou_threshold,
                    class_color_getter=self.get_model7_class_color,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
            elif model_num == 8:
                model = model_cache.get("model8") or self.get_yolo_model_8()
                model_cache["model8"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    False,
                    selected_targets=active_targets,
                    model_num=model_num,
                    confidence_threshold=self.model8_confidence_threshold,
                    iou_threshold=self.model8_iou_threshold,
                    class_color_getter=self.get_model8_class_color,
                )
                if detections:
                    detections = self._assign_person_ids_to_detections(model_num, detections, frame_index=self.frame_count, frame=frame)
            total_detections += model_detections
            total_process_time += process_time
            if model_detections > 0:
                detections_by_model[model_num] = detections
                processed_frame = self._draw_model_detections(
                    processed_frame,
                    detections,
                    model_num,
                )
                update_highest(class_name, float(confidence), coords)

        # After all models: link body positions to persistent identities
        # (fallback for tracks that don't have face-based identity yet)
        if self.PERSON_SHARED_ACROSS_MODELS:
            self._link_body_identities(frame, 4)

        return (
            processed_frame,
            total_detections,
            round(total_process_time, 1),
            highest_class,
            highest_conf,
            highest_coords,
            detections_by_model,
            alerts,
        )
    
    def add_table_entry(self, location_file: str, behaviour: str, coordinate: str):
        """Add a new entry to the table with an incremented number."""
        self.table_entry_counter += 1
        new_entry = {
            "no": str(self.table_entry_counter),
            "person_id": "Person_000",
            "location_file": location_file,
            "behaviour": behaviour,
            "coordinate": coordinate,
        }
        self.table_data.insert(0, new_entry)
        print(f"Added entry to table_data: {new_entry}")

    @staticmethod
    def _display_model_num(model_num: int) -> int:
        # Model 6 and 8 are hidden from the page; original Model 7 is shown as Model 6.
        return 6 if model_num == 7 else model_num

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "unknown"

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
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

    @staticmethod
    def _bbox_center_score(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
        bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
        dx = abs(acx - bcx) / max(aw, bw)
        dy = abs(acy - bcy) / max(ah, bh)
        distance = (dx * dx + dy * dy) ** 0.5
        return max(0.0, 1.0 - distance)

    @staticmethod
    def _bbox_size_similarity(box_a, box_b) -> float:
        aw = max(1.0, float(box_a[2]) - float(box_a[0]))
        ah = max(1.0, float(box_a[3]) - float(box_a[1]))
        bw = max(1.0, float(box_b[2]) - float(box_b[0]))
        bh = max(1.0, float(box_b[3]) - float(box_b[1]))
        width_similarity = min(aw, bw) / max(aw, bw)
        height_similarity = min(ah, bh) / max(ah, bh)
        return min(width_similarity, height_similarity)

    @staticmethod
    def _bbox_containment_score(inner_box, outer_box) -> float:
        """How much is inner_box contained within outer_box?
        Returns 0-1 score. >0.5 means mostly contained."""
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
        inter_x1 = max(ix1, ox1)
        inter_y1 = max(iy1, oy1)
        inter_x2 = min(ix2, ox2)
        inter_y2 = min(iy2, oy2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        return inter_area / inner_area

    @staticmethod
    def _smooth_box(previous_box, current_box, keep_ratio: float):
        return tuple(
            float(previous_box[i]) * keep_ratio + float(current_box[i]) * (1.0 - keep_ratio)
            for i in range(4)
        )

    @classmethod
    def _bbox_track_score(cls, track: dict, det_box) -> float:
        last_box = track.get("box", det_box)
        anchor_box = track.get("anchor_box", last_box)
        body_anchor = track.get("body_anchor_box")
        last_iou = cls._bbox_iou(last_box, det_box)
        last_center = cls._bbox_center_score(last_box, det_box)
        last_size = cls._bbox_size_similarity(last_box, det_box)
        anchor_iou = cls._bbox_iou(anchor_box, det_box)
        anchor_center = cls._bbox_center_score(anchor_box, det_box)
        containment = cls._bbox_containment_score(det_box, last_box)
        anchor_containment = cls._bbox_containment_score(det_box, anchor_box)

        # Body anchor matching: if we have a body anchor for this track,
        # check containment and proximity to the body anchor (highest signal)
        body_score = 0.0
        body_containment = 0.0
        if body_anchor is not None:
            body_containment = cls._bbox_containment_score(det_box, body_anchor)
            body_center = cls._bbox_center_score(body_anchor, det_box)
            body_size = cls._bbox_size_similarity(body_anchor, det_box)
            body_iou = cls._bbox_iou(body_anchor, det_box)
            body_score = max(body_iou, body_containment * 0.65 + body_center * 0.35)

        # Dynamic size threshold: lower for cross-model (face vs body etc.)
        best_containment = max(containment, anchor_containment, body_containment)
        if best_containment > 0.6:
            size_min = cls.PERSON_CROSS_MODEL_SIZE_MIN
        else:
            size_min = cls.PERSON_SIZE_MIN_SIMILARITY

        if last_size < size_min and body_score < 0.3:
            return 0.0

        if last_iou < 0.05 and last_center < 0.30 and body_score < 0.3:
            return 0.0

        # Core score: blend last-frame match, anchor match, and body-anchor match
        last_score = last_iou * 0.40 + last_center * 0.42 + last_size * 0.18
        anchor_score = anchor_iou * 0.45 + anchor_center * 0.40 + anchor_containment * 0.15

        # If body anchor exists and gives a strong signal, use it
        if body_score > 0.5:
            return body_score * 0.55 + last_score * 0.30 + anchor_score * 0.15

        # Containment bonus for detections nested inside track boxes
        containment_bonus = 0.0
        if best_containment > 0.5:
            containment_bonus = cls.PERSON_CONTAINMENT_BONUS * best_containment

        return last_score * 0.65 + anchor_score * 0.35 + containment_bonus

    @staticmethod
    def _detection_box(detection: dict):
        coords = detection.get("coords", {})
        return (
            float(coords.get("xmin", 0)),
            float(coords.get("ymin", 0)),
            float(coords.get("xmax", 0)),
            float(coords.get("ymax", 0)),
        )

    @staticmethod
    def _tracking_sort_key(index_and_detection):
        _, detection = index_and_detection
        x1, y1, x2, y2 = CameraState._detection_box(detection)
        return ((y1 + y2) / 2.0, (x1 + x2) / 2.0)

    @staticmethod
    def _box_area(box) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    @staticmethod
    def _box_intersection_area(box_a, box_b) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _center_inside_box(inner_box, outer_box, margin_ratio: float = 0.12) -> bool:
        cx = (float(inner_box[0]) + float(inner_box[2])) / 2.0
        cy = (float(inner_box[1]) + float(inner_box[3])) / 2.0
        ow = max(1.0, float(outer_box[2]) - float(outer_box[0]))
        oh = max(1.0, float(outer_box[3]) - float(outer_box[1]))
        x1 = float(outer_box[0]) - ow * margin_ratio
        y1 = float(outer_box[1]) - oh * margin_ratio
        x2 = float(outer_box[2]) + ow * margin_ratio
        y2 = float(outer_box[3]) + oh * margin_ratio
        return x1 <= cx <= x2 and y1 <= cy <= y2

    @classmethod
    def _person_body_association_score(cls, behavior_box, body_box) -> float:
        behavior_area = max(1.0, cls._box_area(behavior_box))
        overlap_ratio = cls._box_intersection_area(behavior_box, body_box) / behavior_area
        center_score = cls._bbox_center_score(behavior_box, body_box)
        iou = cls._bbox_iou(behavior_box, body_box)
        containment = cls._bbox_containment_score(behavior_box, body_box)
        center_inside_bonus = 0.30 if cls._center_inside_box(behavior_box, body_box) else 0.0
        # Blend containment and overlap for face/body mismatches
        if containment > 0.70:
            return max(iou, containment * 0.50 + center_score * 0.20 + center_inside_bonus)
        return max(iou, overlap_ratio * 0.45 + center_score * 0.30 + containment * 0.25)

    @classmethod
    def _nms_person_boxes(cls, detections: List[Dict[str, object]], iou_threshold: float) -> List[Dict[str, object]]:
        kept: List[Dict[str, object]] = []
        pending = sorted(detections, key=lambda item: float(item.get("conf", 0.0)), reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            best_box = cls._detection_box(best)
            pending = [
                candidate
                for candidate in pending
                if cls._bbox_iou(best_box, cls._detection_box(candidate)) < iou_threshold
            ]
        return kept

    def _detect_person_body_boxes(self, frame, frame_index: int | None = None) -> List[Dict[str, object]]:
        if frame is None:
            return []

        fi = frame_index if frame_index is not None else self.frame_count
        if (CameraState._cached_body_boxes is not None
                and CameraState._cached_body_frame_index == fi):
            return CameraState._cached_body_boxes

        try:
            detector = self.get_openvino_action_model()
            raw_detections = detector._infer(
                frame,
                self.PERSON_BODY_CONF_THRESHOLD,
                self.PERSON_BODY_ACTION_CONF_THRESHOLD,
            )
            body_detections = []
            for detection in raw_detections:
                coords = detection.get("coords", {})
                body_detections.append(
                    {
                        "class_name": "person",
                        "conf": float(detection.get("conf", 0.0)),
                        "coords": {
                            "xmin": int(coords.get("xmin", 0)),
                            "ymin": int(coords.get("ymin", 0)),
                            "xmax": int(coords.get("xmax", 0)),
                            "ymax": int(coords.get("ymax", 0)),
                        },
                    }
                )
            result = self._nms_person_boxes(body_detections, self.PERSON_BODY_NMS_IOU)
            CameraState._cached_body_boxes = result
            CameraState._cached_body_frame_index = fi
            return result
        except Exception as exc:
            print(f"Person body detector unavailable, fallback: {exc}")
            return []

    def _bind_behaviors_to_person_bodies(self, detections: list, body_detections: List[Dict[str, object]]):
        if not detections or not body_detections:
            return detections

        bound_detections = []
        for detection in detections:
            behavior_box = self._detection_box(detection)
            best_index = None
            best_score = 0.0
            for body_index, body_detection in enumerate(body_detections):
                body_box = self._detection_box(body_detection)
                score = self._person_body_association_score(behavior_box, body_box)
                if score > best_score:
                    best_score = score
                    best_index = body_index

            if best_index is not None and best_score >= self.PERSON_BODY_ASSOC_MIN_SCORE:
                body_detection = body_detections[best_index]
                detection["behavior_coords"] = dict(detection.get("coords", {}))
                detection["coords"] = dict(body_detection.get("coords", {}))
                detection["body_conf"] = float(body_detection.get("conf", 0.0))
                detection["body_binding_score"] = round(float(best_score), 3)
                detection["body_detection_index"] = int(best_index)
            bound_detections.append(detection)
        return bound_detections

    def _person_tracker_key(self, model_num: int):
        return "shared" if self.PERSON_SHARED_ACROSS_MODELS else model_num

    def _ensure_person_tracker(self, model_num: int) -> Dict[str, object]:
        tracker_key = self._person_tracker_key(model_num)
        if tracker_key not in CameraState._person_tracking_state:
            CameraState._person_tracking_state[tracker_key] = {"next_id": 1, "tracks": {}}
        return CameraState._person_tracking_state[tracker_key]

    def _reset_person_tracking(self):
        CameraState._person_tracking_state = {}
        CameraState._person_behavior_log = {}
        CameraState._person_identity_map = {}

    def _add_person_behavior_record(self, model_num: int, person_id: int, detection: dict, frame_index: int, box):
        CameraState._person_behavior_log.setdefault(model_num, {}).setdefault(person_id, []).append({
            "frame": int(frame_index),
            "behavior": str(detection.get("class_name", "unknown")),
            "confidence": float(detection.get("conf", 0.0)),
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
        })

    def _link_face_identities(self, frame, detections: list, frame_index: int = 0):
        """Extract face crops from Model 5 detections and link to persistent identities.

        For each detection with a person_id, extract the face crop from the frame,
        compute a face encoding, and match against the persistent PersonRegistry.
        Stores the mapping in _person_identity_map for use during summary writing.

        Expands face crop by 40% in each direction to ensure enough context for
        face_recognition encoding (needs ≥48px).
        """
        if not detections:
            return

        registry = get_registry()
        frame_h, frame_w = frame.shape[:2]

        for detection in detections:
            person_id = detection.get("person_id")
            if person_id is None:
                continue

            # Allow re-linking if previous attempt got no encoding
            if int(person_id) in CameraState._person_identity_map:
                existing = CameraState._person_identity_map[int(person_id)]
                if registry.has_face(existing):
                    continue  # Already linked with a face encoding, skip

            coords = detection.get("coords", {})
            if not coords:
                continue

            fx1 = int(coords.get("xmin", 0))
            fy1 = int(coords.get("ymin", 0))
            fx2 = int(coords.get("xmax", 0))
            fy2 = int(coords.get("ymax", 0))
            fw, fh = fx2 - fx1, fy2 - fy1

            # Expand face crop by 50% in each direction for better encoding context
            expand_w = int(fw * 0.5)
            expand_h = int(fh * 0.5)
            x1 = max(0, fx1 - expand_w)
            y1 = max(0, fy1 - expand_h)
            x2 = min(frame_w, fx2 + expand_w)
            y2 = min(frame_h, fy2 + expand_h)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Try encoding; skip if face too small or no face found
            encoding = extract_face_encoding(face_crop)

            # Compute body center using original face box center for spatial fallback
            cx = (fx1 + fx2) / 2.0
            cy = (fy1 + fy2) / 2.0

            # Match or create identity
            identity_id, is_new = registry.find_or_create(
                encoding,
                body_center=(cx, cy),
                frame_size=(frame_w, frame_h),
            )

            CameraState._person_identity_map[int(person_id)] = identity_id

            # Save face image periodically
            if is_new or frame_index % 150 == 0:
                try:
                    registry.save_face_image(identity_id, face_crop)
                except Exception:
                    pass

    def _link_body_identities(self, frame, model_num: int):
        """Fallback: link body-level detections to identities by position.

        For person tracks that don't yet have a face-based identity, try to
        match by body center position against the PersonRegistry (same seat).
        """
        tracker = self._ensure_person_tracker(model_num)
        tracks = tracker.get("tracks", {})
        if not tracks:
            return

        registry = get_registry()
        frame_h, frame_w = frame.shape[:2]

        for track_id, track in tracks.items():
            if int(track_id) in CameraState._person_identity_map:
                continue

            # Use body_anchor_box if available, otherwise the tracked box
            box = track.get("body_anchor_box") or track.get("box")
            if box is None:
                continue

            bx1, by1, bx2, by2 = box
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0

            identity_id, is_new = registry.find_or_create(
                None,  # No face encoding
                body_center=(cx, cy),
                frame_size=(frame_w, frame_h),
            )

            CameraState._person_identity_map[int(track_id)] = identity_id

    def _assign_person_ids_to_detections(
        self,
        model_num: int,
        detections: list,
        frame_index: int | None = None,
        frame=None,
    ):
        tracker = self._ensure_person_tracker(model_num)
        tracks = tracker["tracks"]
        next_id = int(tracker["next_id"])
        frame_index = self.frame_count if frame_index is None else frame_index

        # Phase 0: Body binding — cache-aware body detection once per frame
        body_detections = None
        if self.PERSON_BODY_BINDING_ENABLED and frame is not None:
            body_detections = self._detect_person_body_boxes(frame, frame_index)
            if body_detections:
                detections = self._bind_behaviors_to_person_bodies(detections, body_detections)

        # Group detections by body-binding group
        detection_groups: Dict[object, List[int]] = {}
        for det_index, detection in enumerate(detections):
            group_key = detection.get("body_detection_index")
            if group_key is None:
                group_key = ("det", det_index)
            else:
                group_key = ("body", int(group_key))
            detection_groups.setdefault(group_key, []).append(det_index)

        # CRITICAL: For body-bound groups, use the BODY box (not the face/hand box)
        # so that matching against body tracks works correctly.
        group_boxes = {}
        for group_key, det_indices in detection_groups.items():
            if isinstance(group_key, tuple) and group_key[0] == "body" and body_detections:
                body_idx = group_key[1]
                if body_idx < len(body_detections):
                    group_boxes[group_key] = self._detection_box(body_detections[body_idx])
                else:
                    group_boxes[group_key] = self._detection_box(detections[det_indices[0]])
            else:
                group_boxes[group_key] = self._detection_box(detections[det_indices[0]])

        # Phase 1: Containment-first matching — detections fully inside
        # a track's body_anchor_box get highest priority
        candidates = []
        containment_matches = {}
        for track_id, track in tracks.items():
            body_anchor = track.get("body_anchor_box")
            for group_key, det_box in group_boxes.items():
                score = self._bbox_track_score(track, det_box)
                if score >= self.PERSON_MATCH_MIN_SCORE:
                    candidates.append((score, track_id, group_key))
                    continue
                # Relaxed re-check: high containment → accept lower score
                if body_anchor is not None:
                    containment = self._bbox_containment_score(det_box, body_anchor)
                    if containment > 0.50 and score >= self.PERSON_MATCH_MIN_SCORE * 0.70:
                        boost = score + containment * self.PERSON_CONTAINMENT_BONUS
                        if boost >= self.PERSON_MATCH_MIN_SCORE:
                            candidates.append((boost, track_id, group_key))
                            containment_matches[(track_id, group_key)] = True

        # Greedy assignment (highest score first)
        used_tracks = set()
        used_groups = set()
        for score, track_id, group_key in sorted(candidates, reverse=True):
            if track_id in used_tracks or group_key in used_groups:
                continue
            box = group_boxes[group_key]
            previous_box = tracks[track_id].get("box", box)
            tracks[track_id]["box"] = self._smooth_box(previous_box, box, self.PERSON_TRACK_SMOOTHING)
            tracks[track_id]["last_detection_box"] = box
            tracks[track_id]["last_frame"] = frame_index
            tracks[track_id]["missed"] = 0
            tracks[track_id]["hits"] = int(tracks[track_id].get("hits", 0)) + 1

            # Update body_anchor_box if this model provides body-level detections
            # (Model 4 / OpenVINO person detector)
            is_body_model = (model_num == 4)
            if is_body_model and body_detections:
                group_indices = detection_groups[group_key]
                body_idx = detections[group_indices[0]].get("body_detection_index")
                if body_idx is not None and int(body_idx) < len(body_detections):
                    body_box = self._detection_box(body_detections[int(body_idx)])
                    tracks[track_id]["body_anchor_box"] = body_box
            elif is_body_model:
                # Model 4 detection boxes are already body-level
                tracks[track_id]["body_anchor_box"] = box

            for det_index in detection_groups[group_key]:
                detection = detections[det_index]
                detection["person_id"] = track_id
                detection["track_score"] = round(float(score), 3)
                self._add_person_behavior_record(model_num, track_id, detection, frame_index, box)

            used_tracks.add(track_id)
            used_groups.add(group_key)

        # Phase 2: Create new tracks for unmatched groups
        # Pre-compute unused track boxes for NMS merge check
        unused_track_items = []
        for track_id, track in tracks.items():
            if track_id not in used_tracks:
                tb = track.get("box")
                if tb is not None:
                    unused_track_items.append((track_id, tb))

        for group_key, det_indices in sorted(
            detection_groups.items(),
            key=lambda item: self._tracking_sort_key((item[1][0], detections[item[1][0]])),
        ):
            if group_key in used_groups:
                continue
            box = group_boxes[group_key]

            # Check if this detection overlaps too much with an existing
            # unmatched track → merge instead of creating a duplicate
            merged_track_id = None
            for tid, tb in unused_track_items:
                if self._bbox_iou(box, tb) >= self.PERSON_TRACK_NMS_IOU:
                    merged_track_id = tid
                    break
            if merged_track_id is None:
                # Also check containment: if detection center is inside an
                # existing track's box, it's likely the same person
                for tid, tb in unused_track_items:
                    if self._center_inside_box(box, tb, margin_ratio=0.20):
                        merged_track_id = tid
                        break

            if merged_track_id is not None:
                # Merge into existing track
                track = tracks[merged_track_id]
                track["box"] = self._smooth_box(track.get("box", box), box, self.PERSON_TRACK_SMOOTHING)
                track["last_detection_box"] = box
                track["last_frame"] = frame_index
                track["missed"] = 0
                track["hits"] = int(track.get("hits", 0)) + 1
                for det_index in det_indices:
                    detection = detections[det_index]
                    detection["person_id"] = merged_track_id
                    detection["track_score"] = 0.85
                    self._add_person_behavior_record(model_num, merged_track_id, detection, frame_index, box)
                used_tracks.add(merged_track_id)
                # Remove from unused list so subsequent groups can't merge to it again
                unused_track_items = [(tid, tb) for tid, tb in unused_track_items if tid != merged_track_id]
                continue

            track_id = next_id
            next_id += 1
            track_data = {
                "box": box,
                "anchor_box": box,
                "last_detection_box": box,
                "last_frame": frame_index,
                "missed": 0,
                "hits": 1,
            }
            # Store body anchor for Model 4 (body-level detections)
            if model_num == 4 and body_detections:
                body_idx = detections[det_indices[0]].get("body_detection_index")
                if body_idx is not None and int(body_idx) < len(body_detections):
                    track_data["body_anchor_box"] = self._detection_box(
                        body_detections[int(body_idx)]
                    )
                else:
                    track_data["body_anchor_box"] = box
            elif model_num == 4:
                track_data["body_anchor_box"] = box

            # Try to recover identity from PersonRegistry using body position
            # (critical for model restart / new session identity continuity)
            if int(track_id) not in CameraState._person_identity_map:
                try:
                    registry = get_registry()
                    bx1, by1, bx2, by2 = box
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                    else:
                        fh, fw = 1080, 1920  # sensible defaults
                    recovered_id, is_new = registry.find_or_create(
                        None,
                        body_center=(cx, cy),
                        frame_size=(fw, fh),
                    )
                    if not is_new:
                        CameraState._person_identity_map[int(track_id)] = recovered_id
                except Exception:
                    pass  # registry not available

            tracks[track_id] = track_data
            for det_index in det_indices:
                detection = detections[det_index]
                detection["person_id"] = track_id
                detection["track_score"] = 1.0
                self._add_person_behavior_record(model_num, track_id, detection, frame_index, box)
            used_tracks.add(track_id)

        for track_id in list(tracks.keys()):
            if track_id in used_tracks:
                continue
            tracks[track_id]["missed"] = int(tracks[track_id].get("missed", 0)) + 1
            if tracks[track_id]["missed"] > CameraState._person_track_max_missed:
                del tracks[track_id]

        tracker["next_id"] = next_id
        return detections

    def _write_person_summary(self, model_num: int, current_date: str):
        model_log = CameraState._person_behavior_log.get(model_num, {})
        if not model_log:
            return

        # Look up track hit counts to filter out fleeting detections
        tracker = self._ensure_person_tracker(model_num)
        tracks = tracker.get("tracks", {})
        track_hits: Dict[int, int] = {}
        for tid, track in tracks.items():
            track_hits[int(tid)] = int(track.get("hits", 0))

        model_folder = f"Model_{self._display_model_num(model_num)}"
        base_dir = os.path.join("detections", current_date, model_folder)
        os.makedirs(base_dir, exist_ok=True)
        summary_csv = os.path.join(base_dir, "person_summary.csv")
        summary_json = os.path.join(base_dir, "person_summary.json")

        rows = []
        json_rows = []
        skipped_count = 0
        for person_id in sorted(model_log.keys()):
            records = model_log[person_id]
            if not records:
                continue
            hits = track_hits.get(int(person_id), 0)
            # Skip tracks that haven't been seen enough times —
            # these are likely false positives or fleeting detections
            if hits < self.PERSON_MIN_HITS_TO_PERSIST:
                skipped_count += 1
                continue
            counts = Counter(record["behavior"] for record in records)
            dominant_behavior, dominant_count = counts.most_common(1)[0]
            identity_id = CameraState._person_identity_map.get(int(person_id), "")
            row = {
                "person_id": f"Person_{person_id:03d}",
                "identity_id": identity_id,
                "total_events": len(records),
                "track_hits": hits,
                "dominant_behavior": dominant_behavior,
                "dominant_count": dominant_count,
                "behavior_counts": json.dumps(dict(counts), ensure_ascii=False),
            }
            rows.append(row)
            json_rows.append(row)

        if skipped_count > 0:
            print(f"Model {self._display_model_num(model_num)}: skipped {skipped_count} "
                  f"low-quality tracks (hits < {self.PERSON_MIN_HITS_TO_PERSIST}), "
                  f"kept {len(rows)}")

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["person_id", "identity_id", "total_events", "track_hits", "dominant_behavior", "dominant_count", "behavior_counts"])
            writer.writeheader()
            writer.writerows(rows)

        Path(summary_json).write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    async def _save_detection_image(self, frame, model_num: int, detections: list):
        """Save each detected bounding box as a separate cropped image."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            model_folder = f"Model_{self._display_model_num(model_num)}"
            base_dir = os.path.join("detections", current_date, model_folder)
            os.makedirs(base_dir, exist_ok=True)

            if detections and all("person_id" not in detection for detection in detections):
                detections = self._assign_person_ids_to_detections(model_num, detections, frame=frame)

            timestamp = datetime.now().strftime("%H-%M-%S")
            class_name_counts: Dict[str, int] = {}

            for idx, detection in enumerate(detections):
                class_name = detection["class_name"]
                safe_class_name = self._safe_filename_part(str(class_name))
                person_id = int(detection.get("person_id", 0) or 0)
                person_folder = f"Person_{person_id:03d}" if person_id > 0 else "Person_000"
                person_dir = os.path.join(base_dir, person_folder)
                os.makedirs(person_dir, exist_ok=True)

                coords = detection["coords"]
                x1, y1, x2, y2 = coords["xmin"], coords["ymin"], coords["xmax"], coords["ymax"]

                height, width = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bounding box for {class_name}: skipping save")
                    continue

                cropped_image = frame[y1:y2, x1:x2]
                class_name_counts[safe_class_name] = class_name_counts.get(safe_class_name, 0) + 1
                duplicate_suffix = "" if class_name_counts[safe_class_name] == 1 else f"_{class_name_counts[safe_class_name]}"
                filename = f"{timestamp}_{safe_class_name}{duplicate_suffix}.jpg"
                filepath = os.path.join(person_dir, filename)

                cv2.imwrite(filepath, cropped_image)
                print(f"Saved cropped detection image to: {filepath}")

                # Keep the original Model_X/Person_XXX layout, and also archive by person
                # so cross-model detections for one student can be reviewed together.
                people_model_dir = os.path.join("detections", current_date, "People", person_folder, model_folder)
                os.makedirs(people_model_dir, exist_ok=True)
                people_filepath = os.path.join(people_model_dir, filename)
                cv2.imwrite(people_filepath, cropped_image)

                coordinate = f"[{x1},{y1},{x2},{y2}]"
                async with self:
                    self.table_entry_counter += 1
                    new_entry = {
                        "no": str(self.table_entry_counter),
                        "location_file": os.path.join("detections", current_date, model_folder, person_folder, filename),
                        "behaviour": class_name,
                        "coordinate": coordinate,
                        "person_id": person_folder,
                    }
                    self.table_data.append(new_entry)
                    print(f"Added entry to table_data: {new_entry}")

            self._write_person_summary(model_num, current_date)

        except Exception as e:
            print(f"Error in _save_detection_image: {str(e)}")
            
    async def _should_save_detection(self) -> bool:
        """Check if we should save based on rate limiting."""
        current_time = time.time()
        
        # Check if enough time has passed since last save (rate limiting)
        if current_time - self._last_save_time < (60 / self.MAX_SAVES_PER_MINUTE):
            print(f"Rate limiting: Not saving. Time since last save: {current_time - self._last_save_time:.2f} seconds")
            return False
            
        async with self:
            self._last_save_time = current_time
            print(f"Updated last save time: {self._last_save_time}")
            
        return True
    
    @rx.event
    async def handle_image_upload(self, files: list[rx.UploadFile]):
        """Handle image upload from local computer."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Convert image bytes to numpy array
            nparr = np.frombuffer(upload_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Store original frame using the new method
            self.set_original_frame(frame)
            
            # Convert to base64 for display
            img_base64 = base64.b64encode(upload_data).decode('utf-8')
            content_type = file.content_type or "image/jpeg"
            
            # Update state
            self.uploaded_image = f"data:{content_type};base64,{img_base64}"
            self.current_frame = self.uploaded_image
            self.camera_active = False
            self._reset_person_tracking()
            
            # Proses gambar jika deteksi aktif
            if self.detection_enabled:
                return CameraState.process_uploaded_image
            
        except Exception as e:
            self.error_message = f"Upload error: {str(e)}"
                
    @rx.event
    async def toggle_detection(self, enabled: bool):
        """Toggle detection and process uploaded image if exists"""
        # Set state without async with
        self.detection_enabled = enabled
        self.eye_alerts = []
        self.eye_alert_counter = 0
        self.eye_frame_counter = 0
        
        if enabled and self._original_frame_bytes:
            # If enabled, process image with detection
            return CameraState.process_uploaded_image
        elif not enabled and self.uploaded_image:
            # If disabled, restore original uploaded image
            self.current_frame = self.uploaded_image

    @rx.event(background=True)
    async def process_uploaded_image(self):
        """Process uploaded image with selected model detection"""
        try:
            frame = self.original_frame
            if frame is None:
                return
            
            processed_frame = frame.copy()

            if self.detection_enabled:
                if self.cross_model_enabled:
                    (
                        processed_frame,
                        total_detections,
                        process_time,
                        highest_class,
                        highest_conf,
                        coords,
                        detections_by_model,
                        alerts,
                    ) = self._run_cross_model_detection(frame, is_video=False)

                    for model_num, detections in detections_by_model.items():
                        if detections:
                            await self._save_detection_image(frame, model_num, detections)

                    await self._update_detection_metrics(
                        total_detections,
                        process_time,
                        highest_class,
                        highest_conf,
                        coords,
                    )
                    if alerts:
                        async with self:
                            self.eye_alerts = alerts

                elif self.active_model == 1:
                    # Model 1: YOLOv8 for classroom behavior
                    yolo_model = self.get_yolo_model()
                    processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True, model_num=1)
                    
                    if total_detections > 0:
                        await self._save_detection_image(frame, self.active_model, all_detections)
      
                    # Update stats inside context manager
                    async with self:
                        self.detection_count = total_detections
                        self.processing_time = process_time
                        self.highest_confidence_class = highest_class
                        self.highest_confidence = highest_conf
                        self.highest_conf_xmin = coords["xmin"]
                        self.highest_conf_ymin = coords["ymin"]
                        self.highest_conf_xmax = coords["xmax"]
                        self.highest_conf_ymax = coords["ymax"]
                
                elif self.active_model == 2:
                    # Model 2: YOLOv8 for cheating detection
                    yolo_model = self.get_yolo_model_2()
                    processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, False, model_num=2)
                    
                    if total_detections > 0:
                        await self._save_detection_image(frame, self.active_model, all_detections)
                    
                    # Update stats inside context manager
                    async with self:
                        self.detection_count = total_detections
                        self.processing_time = process_time
                        self.highest_confidence_class = highest_class
                        self.highest_confidence = highest_conf
                        self.highest_conf_xmin = coords["xmin"]
                        self.highest_conf_ymin = coords["ymin"]
                        self.highest_conf_xmax = coords["xmax"]
                        self.highest_conf_ymax = coords["ymax"]
                
                elif self.active_model == 3:
                # Model 3: Eye tracking with current thresholds
                    eye_tracker = EyeTracker()
                    try:
                        processed_frame, alerts, total_detections, process_time, highest_class, highest_conf, coords = eye_tracker.process_eye_detections(
                            processed_frame,
                            0,
                            0,
                            cnn_threshold=self.confidence_threshold,  # Use threshold from settings
                            duration_threshold=self.duration_threshold, 
                            is_video=False,
                            selected_targets=self._active_selected_targets()
                        )
                        
                        # Add automatic capture for eye tracking
                        if total_detections > 0:
                            all_detections = [{
                                "class_name": highest_class,
                                "coords": coords
                            }]
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        # Update stats
                        async with self:
                            self.detection_count = total_detections
                            self.processing_time = process_time
                            self.highest_confidence_class = highest_class
                            self.highest_confidence = highest_conf 
                            self.highest_conf_xmin = coords["xmin"]
                            self.highest_conf_ymin = coords["ymin"]
                            self.highest_conf_xmax = coords["xmax"]
                            self.highest_conf_ymax = coords["ymax"]
                            if alerts:
                                self.eye_alerts = alerts
                    except Exception as e:
                        print(f"Eye tracking error: {str(e)}")
                        async with self:
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

                elif self.active_model == 4:
                    try:
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_openvino_prediction(frame, model_num=4)

                        if total_detections > 0:
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        await self._update_detection_metrics(
                            total_detections,
                            process_time,
                            highest_class,
                            highest_conf,
                            coords,
                        )
                    except Exception as e:
                        print(f"OpenVINO model error: {str(e)}")
                        async with self:
                            self.error_message = f"OpenVINO Model 4 error: {str(e)}"
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

                elif self.active_model == 5:
                    try:
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_openvino_emotion_prediction(frame, model_num=5)

                        if total_detections > 0:
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        await self._update_detection_metrics(
                            total_detections,
                            process_time,
                            highest_class,
                            highest_conf,
                            coords,
                        )
                    except Exception as e:
                        print(f"OpenVINO emotion model error: {str(e)}")
                        async with self:
                            self.error_message = f"OpenVINO Model 5 error: {str(e)}"
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

                elif self.active_model == 6:
                    try:
                        slbd_model = self.get_yolo_model_6()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                            slbd_model,
                            frame,
                            False,
                            model_num=6,
                            confidence_threshold=self.model6_confidence_threshold,
                            iou_threshold=self.model6_iou_threshold,
                            class_color_getter=self.get_model6_class_color,
                        )

                        if total_detections > 0:
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        await self._update_detection_metrics(
                            total_detections,
                            process_time,
                            highest_class,
                            highest_conf,
                            coords,
                        )
                    except Exception as e:
                        print(f"SLBD Model 6 error: {str(e)}")
                        async with self:
                            self.error_message = f"SLBD Model 6 error: {str(e)}"
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

                elif self.active_model == 7:
                    try:
                        scb_model = self.get_yolo_model_7()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                            scb_model,
                            frame,
                            False,
                            model_num=7,
                            confidence_threshold=self.model7_confidence_threshold,
                            iou_threshold=self.model7_iou_threshold,
                            class_color_getter=self.get_model7_class_color,
                        )

                        if total_detections > 0:
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        await self._update_detection_metrics(
                            total_detections,
                            process_time,
                            highest_class,
                            highest_conf,
                            coords,
                        )
                    except Exception as e:
                        print(f"SCB YOLO Model 7 error: {str(e)}")
                        async with self:
                            self.error_message = f"SCB YOLO Model 7 error: {str(e)}"
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

                elif self.active_model == 8:
                    try:
                        high_model = self.get_yolo_model_8()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                            high_model,
                            frame,
                            False,
                            model_num=8,
                            confidence_threshold=self.model8_confidence_threshold,
                            iou_threshold=self.model8_iou_threshold,
                            class_color_getter=self.get_model8_class_color,
                        )

                        if total_detections > 0:
                            await self._save_detection_image(frame, self.active_model, all_detections)

                        await self._update_detection_metrics(
                            total_detections,
                            process_time,
                            highest_class,
                            highest_conf,
                            coords,
                        )
                    except Exception as e:
                        print(f"High YOLO Model 8 error: {str(e)}")
                        async with self:
                            self.error_message = f"High YOLO Model 8 error: {str(e)}"
                            self.detection_count = 0
                            self.processing_time = 0.0
                            self.highest_confidence_class = "N/A"
                            self.highest_confidence = 0

            # Convert processed frame to base64
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update display
            async with self:
                self.current_frame = f"data:image/jpeg;base64,{img_base64}"
        
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            async with self:
                self.error_message = f"Image processing error: {str(e)}"

    @rx.event
    async def handle_video_upload(self, files: list[rx.UploadFile]):
        """Handle video upload."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Save video to a writable temp folder with a unique name
            upload_dir = self._upload_dir()
            safe_name = self._safe_upload_name(file.name)
            self.video_path = os.path.join(upload_dir, safe_name)
            with open(self.video_path, "wb") as f:
                f.write(upload_data)
            
            # Stop other media sources and start video processing
            self.camera_active = False
            self.uploaded_image = ""
            self.current_frame = ""
            self.video_playing = True
            self.detection_enabled = False  # Reset detection state
            self.eye_alerts = []  # Clear any existing alerts
            self._reset_person_tracking()
            
            return CameraState.process_video_frames
            
        except Exception as e:
            self.error_message = f"Video upload error: {str(e)}"

    @rx.event(background=True)
    async def process_video_frames(self):
        """Process and display video frames."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                async with self:
                    self.error_message = "Failed to open video file"
                    self.video_playing = False
                return

            # Initialize trackers and models
            eye_tracker = None
            yolo_model = None
            yolo_model_2 = None
            yolo_model_6 = None
            yolo_model_7 = None
            yolo_model_8 = None
            openvino_action_model = None
            openvino_emotion_model = None
            frame_count = 0
            all_detections = []
            local_eye_alert_counter = 0
            local_eye_frame_counter = 0
            last_time = time.time()
            cross_model_cache = {}

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:  # Reset video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                processed_frame = frame.copy()

                # Process detections if enabled
                if self.detection_enabled:
                    if self.cross_model_enabled:
                        try:
                            (
                                processed_frame,
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                detections_by_model,
                                alerts,
                            ) = self._run_cross_model_detection(
                                frame,
                                model_cache=cross_model_cache,
                                is_video=True,
                                alert_counter=local_eye_alert_counter,
                                frame_counter=local_eye_frame_counter,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    for model_num, detections in detections_by_model.items():
                                        if detections:
                                            await self._save_detection_image(frame, model_num, detections)

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                            if alerts:
                                async with self:
                                    self.eye_alerts = alerts
                        except Exception as e:
                            print(f"Cross-model detection error in video: {str(e)}")
                            async with self:
                                self.error_message = f"Cross-model detection error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 1:
                        # Model 1: YOLOv8 for classroom behavior
                        if yolo_model is None:
                            yolo_model = self.get_yolo_model()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True, model_num=1)
                            
                        # Only save on interval frames and with rate limiting
                        if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                            if await self._should_save_detection():
                                try:
                                    await self._save_detection_image(frame, self.active_model, all_detections)
                                except Exception as e:
                                    print(f"Error saving detection: {str(e)}")
                                
                        # Calculate FPS
                        current_time = time.time()
                        time_diff = current_time - last_time
                        current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                        last_time = current_time

                        # Update stats
                        async with self:
                            self.detection_count = total_detections
                            self.processing_time = process_time
                            self.fps = current_fps  
                            self.highest_confidence_class = highest_class
                            self.highest_confidence = highest_conf
                            self.highest_conf_xmin = coords["xmin"]
                            self.highest_conf_ymin = coords["ymin"]
                            self.highest_conf_xmax = coords["xmax"]
                            self.highest_conf_ymax = coords["ymax"]

                    elif self.active_model == 2:
                        # Model 2: YOLOv8 for cheating detection
                        if yolo_model_2 is None:
                            yolo_model_2 = self.get_yolo_model_2()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model_2, frame, False, model_num=2)

                        # Only save on interval frames and with rate limiting
                        if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                            if await self._should_save_detection():
                                try:
                                    await self._save_detection_image(frame, self.active_model, all_detections)
                                except Exception as e:
                                    print(f"Error saving detection: {str(e)}")
                                                            
                        # Calculate FPS
                        current_time = time.time()
                        time_diff = current_time - last_time
                        current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                        last_time = current_time

                        # Update stats
                        async with self:
                            self.detection_count = total_detections
                            self.processing_time = process_time
                            self.fps = current_fps
                            self.highest_confidence_class = highest_class
                            self.highest_confidence = highest_conf
                            self.highest_conf_xmin = coords["xmin"]
                            self.highest_conf_ymin = coords["ymin"]
                            self.highest_conf_xmax = coords["xmax"]
                            self.highest_conf_ymax = coords["ymax"]

                    elif self.active_model == 3:
                        # Model 3: Eye tracking
                        if eye_tracker is None:
                            eye_tracker = EyeTracker()
                        
                        try:
                            processed_frame, alerts, total_detections, process_time, highest_class, highest_conf, coords = eye_tracker.process_eye_detections(
                                processed_frame,
                                local_eye_alert_counter,
                                local_eye_frame_counter,
                                cnn_threshold=self.confidence_threshold,
                                duration_threshold=self.duration_threshold,
                                is_video=True,
                                selected_targets=self._active_selected_targets()
                            )
                            
                            # Add automatic capture for eye tracking with interval and rate limiting
                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        all_detections = [{
                                            "class_name": highest_class,
                                            "coords": coords
                                        }]
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving eye detection: {str(e)}")
                            
                            # Hitung FPS
                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            # Update stats
                            async with self:
                                self.detection_count = total_detections
                                self.processing_time = process_time
                                self.fps = current_fps
                                self.highest_confidence_class = highest_class
                                self.highest_confidence = highest_conf
                                self.highest_conf_xmin = coords["xmin"]
                                self.highest_conf_ymin = coords["ymin"]
                                self.highest_conf_xmax = coords["xmax"]
                                self.highest_conf_ymax = coords["ymax"]
                                if alerts:
                                    self.eye_alerts = alerts
                                    self.eye_alert_counter = local_eye_alert_counter
                                    self.eye_frame_counter = local_eye_frame_counter
                        except Exception as e:
                            print(f"Eye tracking error in video: {str(e)}")
                            async with self:
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 4:
                        try:
                            if openvino_action_model is None:
                                openvino_action_model = self.get_openvino_action_model()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = openvino_action_model.predict(
                                frame,
                                confidence_threshold=self.confidence_threshold,
                                iou_threshold=self.iou_threshold,
                                selected_targets=self._active_selected_targets(),
                            )
                            if all_detections:
                                all_detections = self._assign_person_ids_to_detections(4, all_detections, frame_index=frame_count, frame=frame)
                                self._link_body_identities(frame, 4)

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving OpenVINO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"OpenVINO model error in video: {str(e)}")
                            async with self:
                                self.error_message = f"OpenVINO Model 4 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 5:
                        try:
                            if openvino_emotion_model is None:
                                openvino_emotion_model = self.get_openvino_emotion_model()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = openvino_emotion_model.predict(
                                frame,
                                face_confidence_threshold=self.confidence_threshold,
                                emotion_confidence_threshold=self.iou_threshold,
                                selected_targets=self._active_selected_targets(),
                            )
                            if all_detections:
                                all_detections = self._assign_person_ids_to_detections(5, all_detections, frame_index=frame_count, frame=frame)
                                self._link_face_identities(frame, all_detections, frame_count)

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving OpenVINO emotion detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"OpenVINO emotion model error in video: {str(e)}")
                            async with self:
                                self.error_message = f"OpenVINO Model 5 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 6:
                        try:
                            if yolo_model_6 is None:
                                yolo_model_6 = self.get_yolo_model_6()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_6,
                                frame,
                                False,
                                model_num=6,
                                confidence_threshold=self.model6_confidence_threshold,
                                iou_threshold=self.model6_iou_threshold,
                                class_color_getter=self.get_model6_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving SLBD detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"SLBD Model 6 error in video: {str(e)}")
                            async with self:
                                self.error_message = f"SLBD Model 6 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 7:
                        try:
                            if yolo_model_7 is None:
                                yolo_model_7 = self.get_yolo_model_7()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_7,
                                frame,
                                False,
                                model_num=7,
                                frame_index=frame_count,
                                confidence_threshold=self.model7_confidence_threshold,
                                iou_threshold=self.model7_iou_threshold,
                                class_color_getter=self.get_model7_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving SCB YOLO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"SCB YOLO Model 7 error in video: {str(e)}")
                            async with self:
                                self.error_message = f"SCB YOLO Model 7 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 8:
                        try:
                            if yolo_model_8 is None:
                                yolo_model_8 = self.get_yolo_model_8()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_8,
                                frame,
                                False,
                                model_num=8,
                                confidence_threshold=self.model8_confidence_threshold,
                                iou_threshold=self.model8_iou_threshold,
                                class_color_getter=self.get_model8_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving High YOLO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"High YOLO Model 8 error in video: {str(e)}")
                            async with self:
                                self.error_message = f"High YOLO Model 8 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Update display
                async with self:
                    self.current_frame = f"data:image/jpeg;base64,{img_base64}"

                await asyncio.sleep(1/30)  # ~30 fps

            cap.release()
            
        except Exception as e:
            async with self:
                self.error_message = f"Video processing error: {str(e)}"
            
        finally:
            async with self:
                self.processing_active = False
                self.video_playing = False
                self.current_frame = ""

    @rx.event
    async def clear_camera(self):
        """Clear the camera state and stop the camera if it's running."""
        # First, disable detection to reset the switch state
        self.detection_enabled = False
        
        # Wait a brief moment for the switch to update
        await asyncio.sleep(0.1)
        
        # Then clear all other states
        self.camera_active = False
        self.video_playing = False 
        self.current_frame = ""
        self.uploaded_image = ""
        self._original_frame_bytes = b""  # Clear stored original frame
        self.detection_results = []
        self.face_count = 0
        self.error_message = ""
        self.eye_alerts = []
        self._reset_person_tracking()
        self.eye_alert_counter = 0
        self.eye_frame_counter = 0
        self.detection_count = 0
        self.processing_time = 0.0
        self.fps = 0.0
        
        self.table_data: List[Dict[str, str]] = []
        self.table_entry_counter: int = 0
        self._reset_person_tracking()
         
    @rx.event
    def toggle_face_detection(self):
        self.face_detection_active = not self.face_detection_active
        
    @rx.event
    def update_min_neighbors(self, value: str):
        self.min_neighbors = int(value)
        
    @rx.event
    def update_scale_factor(self, value: str):
        self.scale_factor = float(value) / 10

    @rx.event(background=True)
    async def process_camera_feed(self):
        """Process and display webcam frames."""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                async with self:
                    self.error_message = "Failed to open camera"
                    self.camera_active = False
                return

            # Initialize variables outside the loop
            eye_tracker = None
            yolo_model = None
            yolo_model_2 = None
            yolo_model_6 = None
            yolo_model_7 = None
            yolo_model_8 = None
            openvino_action_model = None
            openvino_emotion_model = None
            frame_count = 0
            all_detections = []
            local_eye_alert_counter = 0
            local_eye_frame_counter = 0
            last_time = time.time()
            cross_model_cache = {}

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_frame = frame.copy()

                # Process detections if enabled
                if self.detection_enabled:
                    if self.cross_model_enabled:
                        try:
                            (
                                processed_frame,
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                detections_by_model,
                                alerts,
                            ) = self._run_cross_model_detection(
                                frame,
                                model_cache=cross_model_cache,
                                is_video=True,
                                alert_counter=local_eye_alert_counter,
                                frame_counter=local_eye_frame_counter,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    for model_num, detections in detections_by_model.items():
                                        if detections:
                                            await self._save_detection_image(frame, model_num, detections)

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                            if alerts:
                                async with self:
                                    self.eye_alerts = alerts
                        except Exception as e:
                            print(f"Cross-model detection error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"Cross-model detection error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 1:
                        # Model 1: YOLOv8 for classroom behavior
                        if yolo_model is None:
                            yolo_model = self.get_yolo_model()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True, model_num=1)

                        if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                            if await self._should_save_detection():
                                try:
                                    await self._save_detection_image(frame, self.active_model, all_detections)
                                except Exception as e:
                                    print(f"Error saving detection: {str(e)}") 
                                                               
                        # Calculate FPS
                        current_time = time.time()
                        time_diff = current_time - last_time
                        current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                        last_time = current_time

                        # Update stats
                        async with self:
                            self.detection_count = total_detections
                            self.processing_time = process_time
                            self.fps = current_fps 
                            self.highest_confidence_class = highest_class
                            self.highest_confidence = highest_conf
                            self.highest_conf_xmin = coords["xmin"]
                            self.highest_conf_ymin = coords["ymin"]
                            self.highest_conf_xmax = coords["xmax"]
                            self.highest_conf_ymax = coords["ymax"]

                    elif self.active_model == 2:
                        # Model 2: YOLOv8 for cheating detection
                        if yolo_model_2 is None:
                            yolo_model_2 = self.get_yolo_model_2()
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model_2, frame, False, model_num=2)
                        
                        if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                            if await self._should_save_detection():
                                try:
                                    await self._save_detection_image(frame, self.active_model, all_detections)
                                except Exception as e:
                                    print(f"Error saving detection: {str(e)}")
                                    
                        # Calculate FPS
                        current_time = time.time()
                        time_diff = current_time - last_time
                        current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                        last_time = current_time

                        # Update stats
                        async with self:
                            self.detection_count = total_detections
                            self.processing_time = process_time
                            self.fps = current_fps 
                            self.highest_confidence_class = highest_class
                            self.highest_confidence = highest_conf
                            self.highest_conf_xmin = coords["xmin"]
                            self.highest_conf_ymin = coords["ymin"]
                            self.highest_conf_xmax = coords["xmax"]
                            self.highest_conf_ymax = coords["ymax"]

                    elif self.active_model == 3:
                        # Model 3: Eye tracking
                        if eye_tracker is None:
                            eye_tracker = EyeTracker()
                        
                        try:
                            processed_frame, alerts, total_detections, process_time, highest_class, highest_conf, coords = eye_tracker.process_eye_detections(
                                processed_frame,
                                local_eye_alert_counter,
                                local_eye_frame_counter,
                                cnn_threshold=self.confidence_threshold,
                                duration_threshold=self.duration_threshold,
                                is_video=True,
                                selected_targets=self._active_selected_targets()
                            )
                            
                            # Add automatic capture for eye tracking with interval and rate limiting
                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        all_detections = [{
                                            "class_name": highest_class,
                                            "coords": coords
                                        }]
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving eye detection: {str(e)}")                            
                            
                            # Hitung FPS
                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            # Update stats
                            async with self:
                                self.detection_count = total_detections
                                self.processing_time = process_time
                                self.fps = current_fps
                                self.highest_confidence_class = highest_class
                                self.highest_confidence = highest_conf
                                self.highest_conf_xmin = coords["xmin"]
                                self.highest_conf_ymin = coords["ymin"]
                                self.highest_conf_xmax = coords["xmax"]
                                self.highest_conf_ymax = coords["ymax"]
                                if alerts:
                                    self.eye_alerts = alerts
                                    self.eye_alert_counter = local_eye_alert_counter
                                    self.eye_frame_counter = local_eye_frame_counter
                        except Exception as e:
                            print(f"Eye tracking error in video: {str(e)}")
                            async with self:
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 4:
                        try:
                            if openvino_action_model is None:
                                openvino_action_model = self.get_openvino_action_model()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = openvino_action_model.predict(
                                frame,
                                confidence_threshold=self.confidence_threshold,
                                iou_threshold=self.iou_threshold,
                                selected_targets=self._active_selected_targets(),
                            )
                            if all_detections:
                                all_detections = self._assign_person_ids_to_detections(4, all_detections, frame_index=frame_count, frame=frame)
                                self._link_body_identities(frame, 4)

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving OpenVINO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"OpenVINO model error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"OpenVINO Model 4 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 5:
                        try:
                            if openvino_emotion_model is None:
                                openvino_emotion_model = self.get_openvino_emotion_model()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = openvino_emotion_model.predict(
                                frame,
                                face_confidence_threshold=self.confidence_threshold,
                                emotion_confidence_threshold=self.iou_threshold,
                                selected_targets=self._active_selected_targets(),
                            )
                            if all_detections:
                                all_detections = self._assign_person_ids_to_detections(5, all_detections, frame_index=frame_count, frame=frame)
                                self._link_face_identities(frame, all_detections, frame_count)

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving OpenVINO emotion detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"OpenVINO emotion model error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"OpenVINO Model 5 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 6:
                        try:
                            if yolo_model_6 is None:
                                yolo_model_6 = self.get_yolo_model_6()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_6,
                                frame,
                                False,
                                model_num=6,
                                confidence_threshold=self.model6_confidence_threshold,
                                iou_threshold=self.model6_iou_threshold,
                                class_color_getter=self.get_model6_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving SLBD detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"SLBD Model 6 error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"SLBD Model 6 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 7:
                        try:
                            if yolo_model_7 is None:
                                yolo_model_7 = self.get_yolo_model_7()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_7,
                                frame,
                                False,
                                model_num=7,
                                frame_index=frame_count,
                                confidence_threshold=self.model7_confidence_threshold,
                                iou_threshold=self.model7_iou_threshold,
                                class_color_getter=self.get_model7_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving SCB YOLO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"SCB YOLO Model 7 error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"SCB YOLO Model 7 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                    elif self.active_model == 8:
                        try:
                            if yolo_model_8 is None:
                                yolo_model_8 = self.get_yolo_model_8()
                            processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(
                                yolo_model_8,
                                frame,
                                False,
                                model_num=8,
                                confidence_threshold=self.model8_confidence_threshold,
                                iou_threshold=self.model8_iou_threshold,
                                class_color_getter=self.get_model8_class_color,
                            )

                            if total_detections > 0 and frame_count % self.FRAME_CAPTURE_INTERVAL == 0:
                                if await self._should_save_detection():
                                    try:
                                        await self._save_detection_image(frame, self.active_model, all_detections)
                                    except Exception as e:
                                        print(f"Error saving High YOLO detection: {str(e)}")

                            current_time = time.time()
                            time_diff = current_time - last_time
                            current_fps = round(1.0 / time_diff, 1) if time_diff > 0 else 0.0
                            last_time = current_time

                            await self._update_detection_metrics(
                                total_detections,
                                process_time,
                                highest_class,
                                highest_conf,
                                coords,
                                fps=current_fps,
                            )
                        except Exception as e:
                            print(f"High YOLO Model 8 error in camera: {str(e)}")
                            async with self:
                                self.error_message = f"High YOLO Model 8 error: {str(e)}"
                                self.detection_count = 0
                                self.processing_time = 0.0
                                self.fps = 0.0

                # Convert and display frame
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                async with self:
                    self.current_frame = f"data:image/jpeg;base64,{img_base64}"
                    self.frame_count += 1
                
                await asyncio.sleep(1/30)
                
        except Exception as e:
            async with self:
                self.error_message = f"Camera error: {str(e)}"
                self.camera_active = False
                self.processing_active = False
        
        finally:
            if 'cap' in locals():
                cap.release()
            async with self:
                self.processing_active = False
                self.detection_results = []
                self.current_frame = ""





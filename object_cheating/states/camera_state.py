import reflex as rx
from typing import TypedDict, List
import cv2
import base64
import numpy as np
import asyncio
import os
import time
import re
from datetime import datetime
from typing import List, Dict
from object_cheating.utils.eye_tracker import EyeTracker
from object_cheating.utils.openvino_action_detector import OpenVINOActionDetector
from object_cheating.utils.openvino_emotion_detector import OpenVINOEmotionDetector
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
    }
    
    # Constants for frame capture
    FRAME_CAPTURE_INTERVAL = 10  # Capture every 10th frame
    MAX_SAVES_PER_MINUTE = 6  # Maximum 6 saves per minute (1 every 10 seconds)
    
    # Add timestamp tracking for rate limiting
    _last_save_time: float = 0
    
    @rx.event
    def prev_model(self):
        if self.active_model > 1:
            self.active_model -= 1

    @rx.event
    def next_model(self):
        if self.active_model < 5:
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
            self.set_model_defaults(target)  # Removed await, calling directly
                
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
    
    # Add new YOLO model for Model 2
    _yolo_model_2 = None
    _openvino_action_model = None
    _openvino_emotion_model = None
    
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
    
    def __init__(self, *args, **kwargs):
        """Initialize state with parent initialization."""
        super().__init__(*args, **kwargs)
        
    @rx.event
    def set_active_model(self, model_num: int):
        if 1 <= model_num <= 5:
            self.active_model = model_num
            self.selected_target = "All"
            self.selected_targets = ["All"]
        else:
            print(f"Nomor model tidak valid: {model_num}. Harus antara 1 dan 5.")
        
    
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
            
    def _apply_yolo_prediction(
        self,
        model,
        frame,
        is_model_1=True,
        selected_targets=None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
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
                
                if is_model_1:
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
    ):
        """Run Open Model Zoo Smart Classroom person/action detection."""
        openvino_model = self.get_openvino_action_model()
        return openvino_model.predict(
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

    def _apply_openvino_emotion_prediction(
        self,
        frame,
        selected_targets=None,
        face_confidence_threshold: float | None = None,
        emotion_confidence_threshold: float | None = None,
    ):
        """Run Open Model Zoo face detection + emotion recognition."""
        emotion_model = self.get_openvino_emotion_model()
        return emotion_model.predict(
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
                detections_by_model[3] = [
                    {"class_name": class_name, "conf": confidence_ratio, "coords": coords}
                ]
                update_highest(class_name, float(confidence), coords)

        for model_num in (1, 2, 4, 5):
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
                    confidence_threshold=self.model1_confidence_threshold,
                    iou_threshold=self.model1_iou_threshold,
                )
            elif model_num == 2:
                model = model_cache.get("model2") or self.get_yolo_model_2()
                model_cache["model2"] = model
                _, model_detections, process_time, class_name, confidence, coords, detections = self._apply_yolo_prediction(
                    model,
                    frame,
                    False,
                    selected_targets=active_targets,
                    confidence_threshold=self.model2_confidence_threshold,
                    iou_threshold=self.model2_iou_threshold,
                )
            else:
                if model_num == 5:
                    model = model_cache.get("model5") or self.get_openvino_emotion_model()
                    model_cache["model5"] = model
                    _, model_detections, process_time, class_name, confidence, coords, detections = model.predict(
                        frame,
                        face_confidence_threshold=self.model5_face_confidence_threshold,
                        emotion_confidence_threshold=self.model5_emotion_confidence_threshold,
                        selected_targets=active_targets,
                    )
                else:
                    model = model_cache.get("model4") or self.get_openvino_action_model()
                    model_cache["model4"] = model
                    _, model_detections, process_time, class_name, confidence, coords, detections = model.predict(
                        frame,
                        confidence_threshold=self.model4_confidence_threshold,
                        iou_threshold=self.model4_iou_threshold,
                        selected_targets=active_targets,
                        action_confidence_threshold=self.model4_action_confidence_threshold,
                    )

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
            "location_file": location_file,
            "behaviour": behaviour,
            "coordinate": coordinate,
        }
        self.table_data.insert(0, new_entry)
        print(f"Added entry to table_data: {new_entry}")

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "unknown"
    
    async def _save_detection_image(self, frame, model_num: int, detections: list):
        """Save each detected bounding box as a separate cropped image."""
        try:
            # Create directory structure: detections/YYYY-MM-DD/Model_X/
            current_date = datetime.now().strftime("%Y-%m-%d")
            model_folder = f"Model_{model_num}"
            base_dir = os.path.join("detections", current_date, model_folder)
            os.makedirs(base_dir, exist_ok=True)
            
            # Generate unique timestamp for this batch of detections
            timestamp = datetime.now().strftime("%H-%M-%S")
            base_filename = f"{timestamp}.jpg"
            class_name_counts: Dict[str, int] = {}
            
            # Process each detection
            for idx, detection in enumerate(detections):
                class_name = detection["class_name"]
                safe_class_name = self._safe_filename_part(str(class_name))
                coords = detection["coords"]
                x1, y1, x2, y2 = coords["xmin"], coords["ymin"], coords["xmax"], coords["ymax"]
                
                # Ensure coordinates are within image bounds
                height, width = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bounding box for {class_name}: skipping save")
                    continue
                
                # Crop the bounding box area
                cropped_image = frame[y1:y2, x1:x2]
                
                # Generate readable filenames: time_class.jpg, time_class_2.jpg, ...
                class_name_counts[safe_class_name] = class_name_counts.get(safe_class_name, 0) + 1
                duplicate_suffix = (
                    "" if class_name_counts[safe_class_name] == 1 else f"_{class_name_counts[safe_class_name]}"
                )
                filename = f"{timestamp}_{safe_class_name}{duplicate_suffix}.jpg"
                filepath = os.path.join(base_dir, filename)
                
                # Save the cropped image
                cv2.imwrite(filepath, cropped_image)
                print(f"Saved cropped detection image to: {filepath}")
                
                # Add entry to table within context manager
                coordinate = f"[{x1},{y1},{x2},{y2}]"
                async with self:
                    self.table_entry_counter += 1
                    new_entry = {
                        "no": str(self.table_entry_counter),  # Start numbering from 1
                        "location_file": os.path.join("detections", current_date, model_folder, filename),
                        "behaviour": class_name,
                        "coordinate": coordinate,
                    }
                    # Append to end instead of insert at beginning
                    self.table_data.append(new_entry)
                    print(f"Added entry to table_data: {new_entry}")
                    
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
                    processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True)
                    
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
                    processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, False)
                    
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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_openvino_prediction(frame)

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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_openvino_emotion_prediction(frame)

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
            
            # Save video to temporary file
            self.video_path = os.path.join(rx.get_upload_dir(), file.name)  # Use .name instead of .filename
            with open(self.video_path, "wb") as f:
                f.write(upload_data)
            
            # Stop other media sources and start video processing
            self.camera_active = False
            self.uploaded_image = ""
            self.current_frame = ""
            self.video_playing = True
            self.detection_enabled = False  # Reset detection state
            self.eye_alerts = []  # Clear any existing alerts
            
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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True)
                            
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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model_2, frame, False)

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
        self.eye_alert_counter = 0
        self.eye_frame_counter = 0
        self.detection_count = 0
        self.processing_time = 0.0
        self.fps = 0.0
        
        self.table_data: List[Dict[str, str]] = []
        self.table_entry_counter: int = 0
         
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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model, frame, True)

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
                        processed_frame, total_detections, process_time, highest_class, highest_conf, coords, all_detections = self._apply_yolo_prediction(yolo_model_2, frame, False)
                        
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

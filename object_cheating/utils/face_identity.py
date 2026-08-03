"""
Face Identity Manager — persistent cross-session student identity system.

Uses face_recognition library (dlib) to:
1. Extract 128-d face embeddings from face crops
2. Match against a persistent registry of known students
3. Assign stable identity IDs that survive across detection sessions

Spatial fallback: when faces are not visible, uses body position similarity
to match students across sessions (same seat → same student in classrooms).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Registry paths ─────────────────────────────────────────────────

DATA_DIR = Path("data/faces")
REGISTRY_FILE = DATA_DIR / "person_registry.json"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Face encoding extraction ──────────────────────────────────────

def extract_face_encoding(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """Extract 128-d face encoding from a BGR face crop image.
    Returns None if no face found or encoding failed.
    """
    if face_crop is None or face_crop.size == 0:
        return None
    try:
        # face_recognition expects RGB
        rgb = face_crop[:, :, ::-1] if face_crop.shape[-1] == 3 else face_crop
        # Resize very small faces
        h, w = rgb.shape[:2]
        if h < 48 or w < 48:
            return None
        encodings = _face_recognition().face_encodings(rgb)
        return encodings[0] if encodings else None
    except Exception:
        return None


def compare_faces(known_encoding: np.ndarray, candidate_encoding: np.ndarray) -> float:
    """Compute cosine similarity between two face encodings. Returns 0-1."""
    try:
        # face_recognition uses Euclidean distance internally for compare_faces
        # We use cosine similarity (higher = better match)
        return _face_recognition().face_distance([known_encoding], candidate_encoding)[0]
    except Exception:
        return 1.0


# ── Lazy import ───────────────────────────────────────────────────

_FR_MODULE = None


def _face_recognition():
    global _FR_MODULE
    if _FR_MODULE is None:
        import face_recognition as fr
        _FR_MODULE = fr
    return _FR_MODULE


# ── Person Registry ───────────────────────────────────────────────

class PersonRegistry:
    """Persistent registry of known students with face encodings and metadata."""

    def __init__(self):
        _ensure_dir()
        self._data: Dict[str, dict] = {}
        self._encoding_cache: Dict[str, List[np.ndarray]] = {}
        self._load()

    # ── IO ──

    def _load(self):
        if REGISTRY_FILE.exists():
            try:
                self._data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        # Load cached encodings from disk
        for person_id, info in self._data.items():
            enc_path = DATA_DIR / person_id / "encodings.npy"
            if enc_path.exists():
                try:
                    self._encoding_cache[person_id] = list(
                        np.load(enc_path, allow_pickle=True)
                    )
                except Exception:
                    self._encoding_cache[person_id] = []

    def _save(self):
        _ensure_dir()
        REGISTRY_FILE.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_encodings(self, person_id: str):
        if person_id in self._encoding_cache:
            person_dir = DATA_DIR / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                str(person_dir / "encodings.npy"),
                np.array(self._encoding_cache[person_id], dtype=object),
                allow_pickle=True,
            )

    # ── Identity matching ──

    def find_or_create(
        self,
        face_encoding: Optional[np.ndarray],
        body_center: Optional[Tuple[float, float]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, bool]:
        """Find matching person or create new identity.

        Args:
            face_encoding: 128-d face embedding, or None if no face detected
            body_center: (cx, cy) center of the person body box, normalized 0-1
            frame_size: (width, height) of the frame

        Returns:
            (person_id, is_new) tuple
        """
        # Phase 1: Face matching (highest confidence)
        if face_encoding is not None:
            person_id = self._match_face(face_encoding)
            if person_id is not None:
                self._update_person(person_id, face_encoding, body_center, frame_size)
                return person_id, False

        # Phase 2: Spatial matching (same seat → same student)
        if body_center is not None and frame_size is not None:
            person_id = self._match_position(body_center, frame_size)
            if person_id is not None:
                if face_encoding is not None:
                    self._update_person(person_id, face_encoding, body_center, frame_size)
                return person_id, False

        # Phase 3: Create new identity
        new_id = f"stu_{uuid.uuid4().hex[:12]}"
        self._create_person(new_id, face_encoding, body_center, frame_size)
        return new_id, True

    def _match_face(self, encoding: np.ndarray) -> Optional[str]:
        """Match face encoding against all known persons. Returns person_id or None."""
        best_id = None
        best_distance = float("inf")

        for person_id, stored_encodings in self._encoding_cache.items():
            if not stored_encodings:
                continue
            try:
                distances = _face_recognition().face_distance(
                    stored_encodings, encoding
                )
                min_dist = float(np.min(distances))
            except Exception:
                continue

            # dlib face_distance: lower = better match. < 0.5 = same person typically
            if min_dist < 0.50 and min_dist < best_distance:
                best_distance = min_dist
                best_id = person_id

        return best_id

    def _match_position(self, body_center: Tuple[float, float], frame_size: Tuple[int, int]) -> Optional[str]:
        """Match by normalized body position (classroom seats are relatively fixed)."""
        cx, cy = body_center
        fw, fh = frame_size

        best_id = None
        best_distance = float("inf")

        for person_id, info in self._data.items():
            last_pos = info.get("last_position")
            if last_pos is None:
                continue
            lx = last_pos.get("x", -1)
            ly = last_pos.get("y", -1)
            if lx < 0 or ly < 0:
                continue

            # Euclidean distance in normalized coordinates (0-1)
            dx = cx / fw - lx
            dy = cy / fh - ly
            distance = (dx * dx + dy * dy) ** 0.5

            # Threshold: within ~15% of frame size
            if distance < 0.15 and distance < best_distance:
                best_distance = distance
                best_id = person_id

        return best_id

    # ── CRUD ──

    def _create_person(
        self,
        person_id: str,
        face_encoding: Optional[np.ndarray],
        body_center: Optional[Tuple[float, float]],
        frame_size: Optional[Tuple[int, int]],
    ):
        now = time.time()
        info: dict = {
            "id": person_id,
            "name": person_id,  # Default name = ID, user can customize
            "created_at": now,
            "last_seen": now,
            "face_count": 1 if face_encoding is not None else 0,
            "total_detections": 0,
            "last_position": {
                "x": round(body_center[0] / frame_size[0], 4) if body_center and frame_size else -1,
                "y": round(body_center[1] / frame_size[1], 4) if body_center and frame_size else -1,
            },
        }
        self._data[person_id] = info
        if face_encoding is not None:
            self._encoding_cache[person_id] = [face_encoding]
            self._save_encodings(person_id)
        self._save()

    def _update_person(
        self,
        person_id: str,
        face_encoding: Optional[np.ndarray],
        body_center: Optional[Tuple[float, float]],
        frame_size: Optional[Tuple[int, int]],
    ):
        if person_id not in self._data:
            self._create_person(person_id, face_encoding, body_center, frame_size)
            return

        info = self._data[person_id]
        info["last_seen"] = time.time()
        info["total_detections"] = info.get("total_detections", 0) + 1

        if face_encoding is not None:
            info["face_count"] = info.get("face_count", 0) + 1
            if person_id not in self._encoding_cache:
                self._encoding_cache[person_id] = []
            self._encoding_cache[person_id].append(face_encoding)
            # Keep last 50 encodings max
            if len(self._encoding_cache[person_id]) > 50:
                self._encoding_cache[person_id] = self._encoding_cache[person_id][-50:]
            self._save_encodings(person_id)

        if body_center is not None and frame_size is not None:
            info["last_position"] = {
                "x": round(body_center[0] / frame_size[0], 4),
                "y": round(body_center[1] / frame_size[1], 4),
            }

        self._save()

    def get_name(self, person_id: str) -> str:
        info = self._data.get(person_id, {})
        return info.get("name", person_id)

    def set_name(self, person_id: str, name: str):
        if person_id in self._data:
            self._data[person_id]["name"] = name
            self._save()

    def all_persons(self) -> List[dict]:
        return sorted(
            [{"id": pid, **info} for pid, info in self._data.items()],
            key=lambda x: x.get("last_seen", 0),
            reverse=True,
        )

    def get_person(self, person_id: str) -> Optional[dict]:
        info = self._data.get(person_id)
        if info is None:
            return None
        return {"id": person_id, **info}

    def has_face(self, person_id: str) -> bool:
        return person_id in self._encoding_cache and len(self._encoding_cache[person_id]) > 0

    def get_face_image_path(self, person_id: str) -> Optional[str]:
        person_dir = DATA_DIR / person_id
        # Return the first saved face image
        for ext in (".jpg", ".png", ".jpeg"):
            for img_path in sorted(person_dir.glob(f"face_*{ext}")):
                return str(img_path)
        return None

    def get_face_image_base64(self, person_id: str) -> Optional[str]:
        """Return face image as data URI string for display in UI."""
        path = self.get_face_image_path(person_id)
        if path is None:
            return None
        try:
            import base64
            with open(path, "rb") as f:
                raw = f.read()
            return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
        except Exception:
            return None

    def get_face_for_identity(self, identity_id: str) -> Optional[str]:
        """Look up face image for an identity_id. Falls back to checking
        if the identity_id is a person folder directly."""
        img = self.get_face_image_base64(identity_id)
        if img:
            return img
        # Also try matching: identity_id might be in a different format
        # e.g. "2026-08-02__Person_001" -> strip date prefix
        if "__" in identity_id:
            _, short_id = identity_id.split("__", 1)
            img = self.get_face_image_base64(short_id)
            if img:
                return img
        return None

    def save_face_image(self, person_id: str, face_crop: np.ndarray):
        """Save a face crop image for this person."""
        person_dir = DATA_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        timestamp = int(time.time() * 1000)
        img_path = person_dir / f"face_{timestamp}.jpg"
        cv2.imwrite(str(img_path), face_crop)


# ── Singleton ─────────────────────────────────────────────────────

_registry_instance: Optional[PersonRegistry] = None


def get_registry() -> PersonRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PersonRegistry()
    return _registry_instance

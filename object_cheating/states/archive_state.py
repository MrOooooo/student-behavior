import base64
from pathlib import Path
from typing import Dict, List

import reflex as rx


class ArchiveState(rx.State):
    archive_view: str = "person"
    archive_group: str = "All"
    archive_date: str = ""
    archive_dates: List[str] = []
    archive_groups: List[Dict[str, str]] = []
    archive_items: List[Dict[str, str]] = []
    archive_message: str = ""
    preview_open: bool = False
    preview_image_src: str = ""
    preview_person: str = ""
    preview_model: str = ""
    preview_filename: str = ""
    preview_path: str = ""
    max_archive_items: int = 120

    # Cross-page pending filters (set by evaluation page before navigation)
    pending_person_filter: str = ""
    pending_date_filter: str = ""

    def apply_person_filter(self, person_id: str, date: str):
        """Called from evaluation page before navigating to archive."""
        self.pending_person_filter = person_id
        self.pending_date_filter = date

    def load_archive(self):
        base_dir = Path("detections")
        if not base_dir.exists():
            self.archive_dates = []
            self.archive_groups = []
            self.archive_items = []
            self.archive_message = "\u8fd8\u6ca1\u6709\u68c0\u6d4b\u4fdd\u5b58\u7ed3\u679c\uff0c\u8bf7\u5148\u8fd0\u884c\u4e00\u6b21\u68c0\u6d4b\u3002"
            return

        dates = sorted(
            [path.name for path in base_dir.iterdir() if path.is_dir()],
            reverse=True,
        )
        self.archive_dates = dates
        if not dates:
            self.archive_groups = []
            self.archive_items = []
            self.archive_message = "\u8fd8\u6ca1\u6709\u68c0\u6d4b\u4fdd\u5b58\u7ed3\u679c\uff0c\u8bf7\u5148\u8fd0\u884c\u4e00\u6b21\u68c0\u6d4b\u3002"
            return

        # Apply pending person filter from evaluation page
        if self.pending_person_filter:
            if self.pending_date_filter and self.pending_date_filter in dates:
                self.archive_date = self.pending_date_filter
            else:
                self.archive_date = dates[0]
            self.archive_view = "person"
            self.archive_group = self.pending_person_filter
            # Clear pending filter after applying
            self.pending_person_filter = ""
            self.pending_date_filter = ""
        elif not self.archive_date or self.archive_date not in dates:
            self.archive_date = dates[0]

        date_dir = base_dir / self.archive_date
        self.archive_groups = self._build_archive_groups(date_dir)
        valid_groups = [group["name"] for group in self.archive_groups]
        if valid_groups and self.archive_group not in valid_groups:
            self.archive_group = valid_groups[0]
        if not valid_groups:
            self.archive_group = ""

        if self.archive_view == "model":
            items = self._load_by_model(date_dir)
        else:
            items = self._load_by_person(date_dir)
            if not items:
                items = self._load_people_from_model_folders(date_dir)

        self.archive_items = items[: self.max_archive_items]
        if self.archive_items:
            selected = self.archive_group if self.archive_group else "All"
            self.archive_message = f"{selected}: \u5f53\u524d\u663e\u793a {len(self.archive_items)} \u6761\u8bb0\u5f55\u3002"
        else:
            self.archive_message = "\u5f53\u524d\u5206\u7c7b\u4e0b\u6ca1\u6709\u53ef\u663e\u793a\u7684\u68c0\u6d4b\u56fe\u7247\u3002"

    def set_archive_view(self, view: str):
        self.archive_view = view
        self.archive_group = "All"
        self.load_archive()

    def set_archive_group(self, group_name: str):
        self.archive_group = group_name
        self.load_archive()

    def set_archive_date(self, date: str):
        self.archive_date = date
        self.archive_group = "All"
        self.load_archive()

    def refresh_archive(self):
        self.load_archive()

    def open_preview(self, image_src: str, person: str, model: str, filename: str, path: str):
        self.preview_image_src = image_src
        self.preview_person = person
        self.preview_model = model
        self.preview_filename = filename
        self.preview_path = path
        self.preview_open = True

    def set_preview_open(self, is_open: bool):
        self.preview_open = is_open

    def close_preview(self):
        self.preview_open = False

    def _build_archive_groups(self, date_dir: Path) -> List[Dict[str, str]]:
        if self.archive_view == "model":
            groups = self._model_groups(date_dir)
        else:
            groups = self._person_groups(date_dir)
        total = sum(int(group["count"]) for group in groups)
        if not groups:
            return []
        return [{"name": "All", "label": "\u5168\u90e8", "count": str(total)}] + groups

    def _person_groups(self, date_dir: Path) -> List[Dict[str, str]]:
        people_dir = date_dir / "People"
        counts: Dict[str, int] = {}
        if people_dir.exists():
            for person_dir in sorted(people_dir.glob("Person_*")):
                if not person_dir.is_dir():
                    continue
                counts[person_dir.name] = len([path for path in person_dir.glob("Model_*/*.*") if self._is_image(path)])
        else:
            for image_path in date_dir.glob("Model_*/Person_*/*.*"):
                if self._is_image(image_path):
                    person = image_path.parent.name
                    counts[person] = counts.get(person, 0) + 1
        return [
            {"name": name, "label": name, "count": str(count)}
            for name, count in sorted(counts.items())
            if count > 0
        ]

    def _model_groups(self, date_dir: Path) -> List[Dict[str, str]]:
        groups = []
        for model_dir in sorted(date_dir.glob("Model_*")):
            if not model_dir.is_dir():
                continue
            count = len([path for path in model_dir.glob("Person_*/*.*") if self._is_image(path)])
            if count > 0:
                groups.append({"name": model_dir.name, "label": model_dir.name, "count": str(count)})
        return groups

    def _load_by_person(self, date_dir: Path) -> List[Dict[str, str]]:
        people_dir = date_dir / "People"
        if not people_dir.exists():
            return []

        person_dirs = sorted([path for path in people_dir.glob("Person_*") if path.is_dir()])
        if self.archive_group and self.archive_group != "All":
            person_dirs = [path for path in person_dirs if path.name == self.archive_group]

        image_paths = []
        for person_dir in person_dirs:
            image_paths.extend([path for path in person_dir.glob("Model_*/*.*") if self._is_image(path)])
        image_paths = sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)

        items = []
        for image_path in image_paths:
            person = image_path.parents[1].name
            model = image_path.parent.name
            items.append(self._make_item(image_path, group=person, person=person, model=model))
        return items

    def _load_by_model(self, date_dir: Path) -> List[Dict[str, str]]:
        model_dirs = sorted([path for path in date_dir.glob("Model_*") if path.is_dir()])
        if self.archive_group and self.archive_group != "All":
            model_dirs = [path for path in model_dirs if path.name == self.archive_group]

        image_paths = []
        for model_dir in model_dirs:
            image_paths.extend([path for path in model_dir.glob("Person_*/*.*") if self._is_image(path)])
        image_paths = sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)

        return [
            self._make_item(
                image_path,
                group=image_path.parents[1].name,
                person=image_path.parent.name,
                model=image_path.parents[1].name,
            )
            for image_path in image_paths
        ]

    def _load_people_from_model_folders(self, date_dir: Path) -> List[Dict[str, str]]:
        image_paths = [path for path in date_dir.glob("Model_*/Person_*/*.*") if self._is_image(path)]
        if self.archive_group and self.archive_group != "All":
            image_paths = [path for path in image_paths if path.parent.name == self.archive_group]
        image_paths = sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)
        return [
            self._make_item(
                image_path,
                group=image_path.parent.name,
                person=image_path.parent.name,
                model=image_path.parents[1].name,
            )
            for image_path in image_paths
        ]

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

    def _make_item(self, image_path: Path, group: str, person: str, model: str) -> Dict[str, str]:
        return {
            "group": group,
            "person": person,
            "model": model,
            "filename": image_path.name,
            "path": str(image_path),
            "image_src": self._image_to_data_uri(image_path),
        }

    @staticmethod
    def _image_to_data_uri(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix == "jpg" else suffix
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/{mime};base64,{encoded}"
        except Exception:
            return ""

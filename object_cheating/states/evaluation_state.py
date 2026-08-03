"""
学生行为识别成绩评价系统 - 评分引擎

基于课堂行为检测数据 + 队友模块数据，计算每位学生的行为成绩评分。
"""

import csv
import json
import io
from pathlib import Path
from typing import Any, Dict, List

import reflex as rx


# ── 默认行为→分数映射 ─────────────────────────────────────────────

DEFAULT_PARTICIPATION_MAP: Dict[str, int] = {
    "raising_hand": 95, "hand_raising": 95,
    "writing": 85,
    "reading": 75,
    "Normal": 60, "sitting": 60,
    "standing": 40, "Stand Up": 40,
    "turned_around": -20, "Turned Around": -20,
    "lie_on_the_desk": -50, "Lie On The Desk": -50,
    "Wave": -10,
}

DEFAULT_FOCUS_MAP: Dict[str, int] = {
    "center": 95,
    "Normal": 80, "sitting": 80,
    "reading": 85, "writing": 85,
    "neutral": 75,
    "hand_raising": 85, "raising_hand": 85,
    "standing": 60,
    "left": -30, "right": -30,
    "Look Around": -50,
    "bowing_head": -40, "Bowing Head": -40,
    "turned_around": -40, "Turned Around": -40,
    "leaning_over_table": -10, "Leaning Over Table": -10,
}

DEFAULT_ANOMALY_PENALTY: Dict[str, int] = {
    "cheating": -30, "Cheating": -30,
    "using_phone": -20, "Using Phone": -20,
    "Hand Under Table": -15,
    "lie_on_the_desk": -10, "Lie On The Desk": -10,
    "Wave": -5,
    "Bend Over The Desk": -5,
}

DEFAULT_EMOTION_MAP: Dict[str, int] = {
    "happy": 90, "Happy": 90,
    "neutral": 70, "Neutral": 70,
    "surprise": 50, "Surprise": 50,
    "sad": 30, "Sad": 30,
    "anger": 10, "Anger": 10,
}

# ── 默认因子权重 ───────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "participation": 0.20,
    "focus": 0.20,
    "anomaly": 0.15,
    "emotion": 0.10,
    "assignment": 0.15,
    "self_test": 0.10,
    "lab": 0.10,
}

FACTOR_LABELS: Dict[str, str] = {
    "participation": "课堂参与度",
    "focus": "课堂专注度",
    "anomaly": "行为规范",
    "emotion": "情绪状态",
    "assignment": "作业成绩(智教)",
    "self_test": "自测正确率(慧学)",
    "lab": "场景实践(智能教学)",
}

# ── 行为→因子归属（哪些行为参与哪些因子的计算） ─────────────────

BEHAVIOR_FACTOR_MAP: Dict[str, List[str]] = {
    # 参与度相关行为
    "raising_hand": ["participation", "focus"],
    "hand_raising": ["participation", "focus"],
    "writing": ["participation", "focus"],
    "reading": ["participation", "focus"],
    "Normal": ["participation", "focus"],
    "sitting": ["participation", "focus"],
    "standing": ["participation", "focus"],
    "Stand Up": ["participation"],
    "turned_around": ["participation", "focus"],
    "Turned Around": ["participation", "focus"],
    "lie_on_the_desk": ["participation", "anomaly"],
    "Lie On The Desk": ["participation", "anomaly"],
    "Wave": ["participation", "anomaly"],
    # 专注度相关
    "center": ["focus"],
    "neutral": ["focus"],
    "left": ["focus"],
    "right": ["focus"],
    "Look Around": ["focus"],
    "bowing_head": ["focus"],
    "Bowing Head": ["focus"],
    "leaning_over_table": ["focus"],
    "Leaning Over Table": ["focus"],
    # 异常行为
    "cheating": ["anomaly"],
    "Cheating": ["anomaly"],
    "using_phone": ["anomaly"],
    "Using Phone": ["anomaly"],
    "Hand Under Table": ["anomaly"],
    "Bend Over The Desk": ["anomaly"],
    # 情绪
    "happy": ["emotion"],
    "Happy": ["emotion"],
    "sad": ["emotion"],
    "Sad": ["emotion"],
    "surprise": ["emotion"],
    "Surprise": ["emotion"],
    "anger": ["emotion"],
    "Anger": ["emotion"],
    "Neutral": ["emotion"],
}


def _parse_behavior_counts(counts_raw: Any) -> Dict[str, int]:
    """将 person_summary.json 中的 behavior_counts 统一解析为 dict。"""
    if isinstance(counts_raw, str):
        try:
            return json.loads(counts_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(counts_raw, dict):
        return counts_raw
    return {}


def _normalize_score(raw: float, total_events: int) -> float:
    """将原始加权和归一化到 0-100，同时考虑检测次数。"""
    if total_events <= 0:
        return 50.0  # 无数据默认中位分
    # 因子分 = 原始加权和 / 总检测次数
    # 由于各行为分值范围 -50 ~ 95，需要缩放到 0-100
    normalized = raw / total_events
    # 归一化：将 [-50, 95] 映射到 [0, 100]
    score = (normalized + 50) / 145 * 100
    return max(0.0, min(100.0, round(score, 1)))


def _normalize_anomaly(total_penalty: float) -> float:
    """异常行为扣分归一化：起始 100，直接扣除异常分值。"""
    # total_penalty 是负值（各异常行为扣分的加权和）
    # 直接用 100 + total_penalty（如 -60 → 40 分）
    score = 100.0 + total_penalty
    return max(0.0, min(100.0, round(score, 1)))


class EvaluationState(rx.State):
    """平时成绩综合评价状态管理"""

    # ── 权重配置 ──
    participation_weight: float = DEFAULT_WEIGHTS["participation"]
    focus_weight: float = DEFAULT_WEIGHTS["focus"]
    anomaly_weight: float = DEFAULT_WEIGHTS["anomaly"]
    emotion_weight: float = DEFAULT_WEIGHTS["emotion"]
    assignment_weight: float = DEFAULT_WEIGHTS["assignment"]
    self_test_weight: float = DEFAULT_WEIGHTS["self_test"]
    lab_weight: float = DEFAULT_WEIGHTS["lab"]

    # ── 行为→分数映射（JSON 字符串，兼容 Reflex 状态序列化） ──
    participation_map_json: str = json.dumps(DEFAULT_PARTICIPATION_MAP, ensure_ascii=False)
    focus_map_json: str = json.dumps(DEFAULT_FOCUS_MAP, ensure_ascii=False)
    anomaly_penalty_json: str = json.dumps(DEFAULT_ANOMALY_PENALTY, ensure_ascii=False)
    emotion_map_json: str = json.dumps(DEFAULT_EMOTION_MAP, ensure_ascii=False)

    # ── 队友模块数据（JSON 字符串） ──
    assignment_scores_json: str = "{}"
    self_test_scores_json: str = "{}"
    lab_scores_json: str = "{}"

    # ── 使用模拟数据标记 ──
    use_mock_teammate_data: bool = True

    # ── 评估结果 ──
    evaluation_date: str = ""
    evaluation_dates: List[str] = []
    person_evaluations: List[Dict[str, Any]] = []
    class_average: float = 0.0
    student_count: int = 0
    evaluation_message: str = ""

    # ── 跨天聚合评估 ──
    show_all_dates: bool = False  # False = single date, True = all dates aggregated
    aggregated_evaluations: List[Dict[str, Any]] = []
    aggregated_student_count: int = 0
    aggregated_class_average: float = 0.0

    # ── 历史汇总 ──
    all_dates_summary: List[Dict[str, Any]] = []

    # ── UI 状态 ──
    show_weight_config: bool = False
    show_behavior_mapping: bool = False
    selected_student: str = "All"
    loading_evaluation: bool = False

    # ── 行为映射编辑器状态 ──
    selected_mapping_factor: str = "participation"  # 当前编辑哪个因子
    new_behavior_name: str = ""
    new_behavior_score: str = ""
    editing_behavior: str = ""  # 非空表示正在编辑已有行为（存储行为名）

    # ── 导出 ──
    export_ready: bool = False

    # ═══════════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════════

    def load_dates(self):
        """加载所有可评估的日期列表。"""
        base_dir = Path("detections")
        if not base_dir.exists():
            self.evaluation_dates = []
            self.evaluation_message = "暂无检测数据，请先运行检测后查看评估。"
            return

        dates = sorted(
            [p.name for p in base_dir.iterdir() if p.is_dir()],
            reverse=True,
        )
        self.evaluation_dates = dates

        if not dates:
            self.evaluation_message = "暂无检测数据，请先运行检测后查看评估。"
            return

        if not self.evaluation_date or self.evaluation_date not in dates:
            self.evaluation_date = dates[0]

        if self.show_all_dates:
            self.load_all_dates_aggregated()
        else:
            self.load_evaluation(self.evaluation_date)

    def load_evaluation(self, date: str):
        """按日期加载所有模型的 person_summary 并计算评分。"""
        self.loading_evaluation = True
        self.evaluation_date = date

        date_dir = Path("detections") / date
        if not date_dir.exists():
            self.person_evaluations = []
            self.class_average = 0.0
            self.student_count = 0
            self.evaluation_message = f"日期 {date} 下没有检测数据。"
            self.loading_evaluation = False
            return

        # 1. 读取所有 Model_*/person_summary.json
        merged: Dict[str, Dict[str, Any]] = {}  # person_id -> {behavior: count}
        identity_map: Dict[str, str] = {}  # person_id -> identity_id
        model_count = 0

        for model_dir in sorted(date_dir.glob("Model_*")):
            if not model_dir.is_dir():
                continue
            summary_file = model_dir / "person_summary.json"
            if not summary_file.exists():
                continue
            model_count += 1
            try:
                records = json.loads(summary_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            for record in records:
                pid = record.get("person_id", "")
                if not pid:
                    continue
                iid = record.get("identity_id", "")
                if iid and pid not in identity_map:
                    identity_map[pid] = iid
                counts = _parse_behavior_counts(record.get("behavior_counts", {}))
                total = record.get("total_events", 0)

                if pid not in merged:
                    merged[pid] = {"_total_events": 0, "_behaviors": {}}

                merged[pid]["_total_events"] += total
                for behavior, count in counts.items():
                    merged[pid]["_behaviors"][behavior] = (
                        merged[pid]["_behaviors"].get(behavior, 0) + count
                    )

        if not merged:
            self.person_evaluations = []
            self.class_average = 0.0
            self.student_count = 0
            self.evaluation_message = f"日期 {date} 下没有检测到学生行为记录。"
            self.loading_evaluation = False
            return

        # 2. 加载队友数据
        teammate_scores = self._load_teammate_scores(date_dir)

        # 3. 计算评分
        evaluations = self._calculate_all_scores(merged, teammate_scores)

        # 4. 附加人脸照片
        self._attach_face_images(evaluations, identity_map)

        self.person_evaluations = evaluations
        self.student_count = len(evaluations)

        if evaluations:
            self.class_average = round(
                sum(e["weighted_total"] for e in evaluations) / len(evaluations), 1
            )
        else:
            self.class_average = 0.0

        self.evaluation_message = (
            f"日期 {date}：{model_count} 个模型，{len(evaluations)} 名学生，"
            f"班级均分 {self.class_average}"
        )

        self.loading_evaluation = False

    def load_all_dates_aggregated(self):
        """Aggregate behavior data across ALL dates by persistent identity_id."""
        self.loading_evaluation = True
        base_dir = Path("detections")
        if not base_dir.exists():
            self.aggregated_evaluations = []
            self.aggregated_student_count = 0
            self.aggregated_class_average = 0.0
            self.evaluation_message = "暂无任何检测数据。"
            self.loading_evaluation = False
            return

        all_records: Dict[str, Dict[str, Any]] = {}
        date_set: set[str] = set()

        for date_dir in sorted(base_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("evaluation"):
                continue
            date = date_dir.name
            date_set.add(date)

            for model_dir in sorted(date_dir.glob("Model_*")):
                if not model_dir.is_dir():
                    continue
                summary_file = model_dir / "person_summary.json"
                if not summary_file.exists():
                    continue
                try:
                    records = json.loads(summary_file.read_text(encoding="utf-8"))
                except Exception:
                    continue

                for record in records:
                    identity_id = record.get("identity_id", "")
                    person_id = record.get("person_id", "Unknown")

                    if not identity_id:
                        identity_id = f"{date}__{person_id}"

                    if identity_id not in all_records:
                        all_records[identity_id] = {
                            "_identity_id": identity_id,
                            "_display_name": person_id,
                            "_total_events": 0,
                            "_behaviors": {},
                            "_dates": set(),
                        }

                    entry = all_records[identity_id]
                    entry["_dates"].add(date)
                    entry["_total_events"] += record.get("total_events", 0)
                    counts = _parse_behavior_counts(record.get("behavior_counts", {}))
                    for b, c in counts.items():
                        entry["_behaviors"][b] = entry["_behaviors"].get(b, 0) + c

        if not all_records:
            self.aggregated_evaluations = []
            self.aggregated_student_count = 0
            self.aggregated_class_average = 0.0
            self.evaluation_message = "未找到任何学生行为记录。"
            self.loading_evaluation = False
            return

        # Resolve display names from registry
        try:
            from object_cheating.utils.face_identity import get_registry
            registry = get_registry()
            for identity_id in all_records:
                name = registry.get_name(identity_id)
                if name and name != identity_id:
                    all_records[identity_id]["_display_name"] = name
        except Exception:
            pass

        evaluations = self._calculate_all_scores(
            all_records,
            {"assignment": {}, "self_test": {}, "lab": {}},
        )

        for e in evaluations:
            identity_id = e["person_id"]
            entry = all_records.get(identity_id, {})
            e["date_count"] = len(entry.get("_dates", set()))
            e["all_dates"] = sorted(entry.get("_dates", set()))
            display_name = entry.get("_display_name", "")
            if display_name:
                e["display_name"] = display_name

        # Attach face images (person_id is already identity_id in aggregated view)
        self._attach_face_images(evaluations, {})

        self.aggregated_evaluations = evaluations
        self.aggregated_student_count = len(evaluations)

        if evaluations:
            self.aggregated_class_average = round(
                sum(e["weighted_total"] for e in evaluations) / len(evaluations), 1
            )
        else:
            self.aggregated_class_average = 0.0

        self.evaluation_message = (
            f"全部 {len(evaluations)} 名学生 · {len(date_set)} 天数据，"
            f"班级均分 {self.aggregated_class_average}"
        )

        self.loading_evaluation = False

    def toggle_view_mode(self):
        """Switch between single-date and all-dates aggregated view."""
        self.show_all_dates = not self.show_all_dates
        if self.show_all_dates:
            self.load_all_dates_aggregated()
        elif self.evaluation_date:
            self.load_evaluation(self.evaluation_date)
        else:
            self.load_dates()

    def goto_student_archive(self, person_id: str):
        """Set archive filter and navigate to archive page for this student.

        Extracts the correct person folder name (Person_XXX) for single-date view,
        and falls back to identity lookup for aggregated view.
        """
        # Determine the archive person group name and date
        if self.show_all_dates:
            # Aggregated view: identity_id may look like "stu_abc123" or "2026-08-02__Person_001"
            archive_person = person_id
            if "__" in person_id:
                # Extract Person_XXX from fallback format like "2026-08-02__Person_001"
                _, archive_person = person_id.split("__", 1)
            # Find most recent date for this student
            evals = self.aggregated_evaluations or self.person_evaluations
            eval_dates = []
            for e in evals:
                if e.get("person_id") == person_id:
                    eval_dates = e.get("all_dates", [])
                    break
            archive_date = eval_dates[0] if eval_dates else self.evaluation_date
        else:
            # Single-date view: person_id is already like "Person_001"
            archive_person = person_id
            archive_date = self.evaluation_date

        # Set pending filter on ArchiveState and redirect
        from object_cheating.states.archive_state import ArchiveState
        ArchiveState.apply_person_filter(archive_person, archive_date)  # type: ignore[attr-defined]
        return rx.redirect("/archive")

    def _load_teammate_scores(self, date_dir: Path) -> Dict[str, Dict[str, float]]:
        """加载队友模块数据，若不存在则使用模拟数据。"""
        scores: Dict[str, Dict[str, float]] = {
            "assignment": {},
            "self_test": {},
            "lab": {},
        }

        teammate_dir = date_dir / "teammate_scores"
        if teammate_dir.exists():
            for score_file in teammate_dir.glob("*.json"):
                try:
                    data = json.loads(score_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                if "智教" in score_file.stem or "assignment" in score_file.stem.lower():
                    scores["assignment"] = data
                elif "慧学" in score_file.stem or "self" in score_file.stem.lower():
                    scores["self_test"] = data
                elif "智能教学" in score_file.stem or "lab" in score_file.stem.lower():
                    scores["lab"] = data

        # 模拟数据兜底
        if self.use_mock_teammate_data:
            import random
            import hashlib

            if not scores["assignment"]:
                scores["assignment"] = {}
            if not scores["self_test"]:
                scores["self_test"] = {}
            if not scores["lab"]:
                scores["lab"] = {}

        return scores

    # ═══════════════════════════════════════════════════════════════
    # 评分计算
    # ═══════════════════════════════════════════════════════════════

    def _calculate_all_scores(
        self,
        merged: Dict[str, Dict[str, Any]],
        teammate_scores: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """对所有学生计算四项行为因子分及加权总分。"""
        participation_map = json.loads(self.participation_map_json)
        focus_map = json.loads(self.focus_map_json)
        anomaly_map = json.loads(self.anomaly_penalty_json)
        emotion_map = json.loads(self.emotion_map_json)

        evaluations = []
        for pid in sorted(merged.keys()):
            data = merged[pid]
            total_events = data["_total_events"]
            behaviors = data["_behaviors"]

            # 计算四个行为因子
            participation_raw = self._factor_raw_score(
                behaviors, participation_map
            )
            focus_raw = self._factor_raw_score(behaviors, focus_map)
            anomaly_penalty = self._factor_raw_score(behaviors, anomaly_map)
            emotion_raw = self._factor_raw_score(behaviors, emotion_map)

            participation_score = _normalize_score(participation_raw, total_events)
            focus_score = _normalize_score(focus_raw, total_events)
            anomaly_score = _normalize_anomaly(anomaly_penalty)
            emotion_score = _normalize_score(emotion_raw, total_events)

            # 队友数据
            assignment_score = float(teammate_scores["assignment"].get(pid, 0))
            self_test_score = float(teammate_scores["self_test"].get(pid, 0))
            lab_score = float(teammate_scores["lab"].get(pid, 0))

            # 若模拟数据：基于行为分数生成合理模拟值
            if self.use_mock_teammate_data:
                import random
                import hashlib

                seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
                rng = random.Random(seed)

                if assignment_score == 0:
                    base = (participation_score + focus_score) / 2
                    assignment_score = round(
                        max(30, min(98, base + rng.uniform(-10, 15))), 1
                    )
                if self_test_score == 0:
                    base = (participation_score + focus_score) / 2
                    self_test_score = round(
                        max(30, min(98, base + rng.uniform(-12, 12))), 1
                    )
                if lab_score == 0:
                    base = (participation_score + focus_score) / 2
                    lab_score = round(
                        max(30, min(98, base + rng.uniform(-8, 18))), 1
                    )

            # 加权总分
            weights = {
                "participation": self.participation_weight,
                "focus": self.focus_weight,
                "anomaly": self.anomaly_weight,
                "emotion": self.emotion_weight,
                "assignment": self.assignment_weight,
                "self_test": self.self_test_weight,
                "lab": self.lab_weight,
            }

            weighted_total = round(
                participation_score * weights["participation"]
                + focus_score * weights["focus"]
                + anomaly_score * weights["anomaly"]
                + emotion_score * weights["emotion"]
                + assignment_score * weights["assignment"]
                + self_test_score * weights["self_test"]
                + lab_score * weights["lab"],
                1,
            )

            # 找出 dominant behaviors（前3）
            sorted_behaviors = sorted(
                behaviors.items(), key=lambda x: x[1], reverse=True
            )
            dominant = {b: c for b, c in sorted_behaviors[:3]}

            evaluations.append({
                "person_id": pid,
                "participation_score": participation_score,
                "focus_score": focus_score,
                "anomaly_score": anomaly_score,
                "emotion_score": emotion_score,
                "assignment_score": assignment_score,
                "self_test_score": self_test_score,
                "lab_score": lab_score,
                "weighted_total": weighted_total,
                "grade": self._get_grade_label(weighted_total),
                "total_detections": total_events,
                "dominant_behaviors": dominant,
            })

        # 按总分降序排列
        evaluations.sort(key=lambda e: e["weighted_total"], reverse=True)
        return evaluations

    @staticmethod
    def _attach_face_images(
        evaluations: List[Dict[str, Any]],
        identity_map: Dict[str, str],
    ):
        """Attach face image data URI to each evaluation if available.

        For aggregated view (identity_map is empty), person_id is the identity_id.
        For single-date view, identity_map maps person_id -> identity_id.
        """
        try:
            from object_cheating.utils.face_identity import get_registry
            registry = get_registry()
        except Exception:
            return

        for e in evaluations:
            pid = e.get("person_id", "")
            # Try identity_map first (single-date: Person_XXX -> stu_xxxxx)
            iid = identity_map.get(pid, pid)
            # Look up face from the registry
            face_base64 = registry.get_face_for_identity(iid)
            if not face_base64 and iid != pid:
                face_base64 = registry.get_face_for_identity(pid)
            e["face_image"] = face_base64 or ""

    @staticmethod
    def _factor_raw_score(behaviors: Dict[str, int], score_map: Dict[str, int]) -> float:
        """计算一个因子的原始加权分：Σ(行为次数 × 映射分值)。"""
        total = 0.0
        for behavior, count in behaviors.items():
            if behavior in score_map:
                total += count * score_map[behavior]
        return total

    @staticmethod
    def _get_grade_label(score: float) -> str:
        """分数 → 等级映射。"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        return "F"

    # ═══════════════════════════════════════════════════════════════
    # 权重配置
    # ═══════════════════════════════════════════════════════════════

    def toggle_weight_config(self):
        self.show_weight_config = not self.show_weight_config

    def toggle_behavior_mapping(self):
        self.show_behavior_mapping = not self.show_behavior_mapping

    def set_weight(self, factor: str, value: str):
        """更新单个因子权重（从滑块字符串值）。"""
        try:
            val = float(value)
            val = max(0.0, min(1.0, round(val, 2)))
        except ValueError:
            return

        attr_map = {
            "participation": "participation_weight",
            "focus": "focus_weight",
            "anomaly": "anomaly_weight",
            "emotion": "emotion_weight",
            "assignment": "assignment_weight",
            "self_test": "self_test_weight",
            "lab": "lab_weight",
        }
        if factor in attr_map:
            setattr(self, attr_map[factor], val)
            self._recalculate()

    def reset_weights(self):
        """重置权重为默认值。"""
        self.participation_weight = DEFAULT_WEIGHTS["participation"]
        self.focus_weight = DEFAULT_WEIGHTS["focus"]
        self.anomaly_weight = DEFAULT_WEIGHTS["anomaly"]
        self.emotion_weight = DEFAULT_WEIGHTS["emotion"]
        self.assignment_weight = DEFAULT_WEIGHTS["assignment"]
        self.self_test_weight = DEFAULT_WEIGHTS["self_test"]
        self.lab_weight = DEFAULT_WEIGHTS["lab"]
        self._recalculate()

    def _recalculate(self):
        """权重变更后重新计算当前评估结果。"""
        if self.show_all_dates:
            self.load_all_dates_aggregated()
        elif self.evaluation_date and self.person_evaluations:
            self.load_evaluation(self.evaluation_date)

    def set_evaluation_date(self, date: str):
        """切换评估日期（切换到单日视图）。"""
        self.show_all_dates = False
        self.load_evaluation(date)

    def set_selected_student(self, student_id: str):
        self.selected_student = student_id

    def toggle_mock_data(self, value: bool):
        self.use_mock_teammate_data = value
        if self.evaluation_date:
            self.load_evaluation(self.evaluation_date)

    # ═══════════════════════════════════════════════════════════════
    # 行为映射编辑器
    # ═══════════════════════════════════════════════════════════════

    _MAPPING_ATTRS = {
        "participation": "participation_map_json",
        "focus": "focus_map_json",
        "anomaly": "anomaly_penalty_json",
        "emotion": "emotion_map_json",
    }

    _MAPPING_DEFAULTS = {
        "participation": DEFAULT_PARTICIPATION_MAP,
        "focus": DEFAULT_FOCUS_MAP,
        "anomaly": DEFAULT_ANOMALY_PENALTY,
        "emotion": DEFAULT_EMOTION_MAP,
    }

    @rx.var
    def current_behavior_mapping(self) -> List[Dict[str, Any]]:
        """Return the current factor's behavior→score mapping as a list for foreach."""
        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        try:
            data = json.loads(getattr(self, attr, "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        return [{"behavior": k, "score": v} for k, v in sorted(data.items())]

    @rx.var
    def selected_mapping_label(self) -> str:
        return FACTOR_LABELS.get(self.selected_mapping_factor, "课堂参与度")

    @rx.var
    def behavior_factors_for_ui(self) -> List[Dict[str, str]]:
        """Available factors (only the 4 behavior ones) for the mapping editor tabs."""
        return [
            {"key": "participation", "label": FACTOR_LABELS["participation"]},
            {"key": "focus", "label": FACTOR_LABELS["focus"]},
            {"key": "anomaly", "label": FACTOR_LABELS["anomaly"]},
            {"key": "emotion", "label": FACTOR_LABELS["emotion"]},
        ]

    def select_mapping_factor(self, factor: str):
        self.selected_mapping_factor = factor
        self.new_behavior_name = ""
        self.new_behavior_score = ""
        self.editing_behavior = ""

    def set_new_behavior_name(self, name: str):
        self.new_behavior_name = name

    def set_new_behavior_score(self, score: str):
        self.new_behavior_score = score

    def add_behavior_entry(self):
        """Add or update a behavior→score entry for the current factor."""
        name = self.new_behavior_name.strip()
        if not name:
            return
        try:
            score = float(self.new_behavior_score)
        except ValueError:
            score = 0

        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        try:
            data = json.loads(getattr(self, attr, "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}

        data[name] = score
        setattr(self, attr, json.dumps(data, ensure_ascii=False))
        self.new_behavior_name = ""
        self.new_behavior_score = ""
        self.editing_behavior = ""
        self._recalculate()

    def start_edit_behavior(self, behavior: str):
        """Populate the add-new form with an existing behavior's data for editing."""
        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        try:
            data = json.loads(getattr(self, attr, "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if behavior in data:
            self.editing_behavior = behavior
            self.new_behavior_name = behavior
            self.new_behavior_score = str(data[behavior])

    def cancel_edit_behavior(self):
        self.editing_behavior = ""
        self.new_behavior_name = ""
        self.new_behavior_score = ""

    def remove_behavior_entry(self, behavior: str):
        """Remove a behavior entry from the current factor's mapping."""
        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        try:
            data = json.loads(getattr(self, attr, "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if behavior in data:
            del data[behavior]
            setattr(self, attr, json.dumps(data, ensure_ascii=False))
            self._recalculate()

    def update_behavior_score(self, behavior: str, score_val: str):
        """Update a single behavior's score in the current factor mapping."""
        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        try:
            score = float(score_val)
        except ValueError:
            score = 0
        try:
            data = json.loads(getattr(self, attr, "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        data[behavior] = score
        setattr(self, attr, json.dumps(data, ensure_ascii=False))
        self._recalculate()

    def reset_behavior_mapping(self):
        """Reset the current factor's mapping to defaults."""
        key = self.selected_mapping_factor
        attr = self._MAPPING_ATTRS.get(key, "participation_map_json")
        default = self._MAPPING_DEFAULTS.get(key, {})
        setattr(self, attr, json.dumps(default, ensure_ascii=False))
        self._recalculate()

    # ═══════════════════════════════════════════════════════════════
    # 历史汇总
    # ═══════════════════════════════════════════════════════════════

    def load_all_dates_summary(self):
        """汇总所有日期的评估数据。"""
        base_dir = Path("detections")
        if not base_dir.exists():
            self.all_dates_summary = []
            return

        summaries = []
        for date_dir in sorted(base_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            date = date_dir.name

            # 统计该日期的学生数量
            merged = self._quick_merge(date_dir)
            if not merged:
                summaries.append({
                    "date": date,
                    "student_count": 0,
                    "class_average": 0,
                    "has_data": False,
                })
                continue

            evaluations = self._calculate_all_scores(merged, self._load_teammate_scores(date_dir))
            avg = round(sum(e["weighted_total"] for e in evaluations) / len(evaluations), 1) if evaluations else 0

            summaries.append({
                "date": date,
                "student_count": len(evaluations),
                "class_average": avg,
                "has_data": True,
                "grade_distribution": self._grade_distribution(evaluations),
            })

        self.all_dates_summary = summaries

    def _quick_merge(self, date_dir: Path) -> Dict[str, Dict[str, Any]]:
        """快速合并 person_summary（用于汇总，不含详细计算）。"""
        merged: Dict[str, Dict[str, Any]] = {}
        for model_dir in date_dir.glob("Model_*"):
            summary_file = model_dir / "person_summary.json"
            if not summary_file.exists():
                continue
            try:
                records = json.loads(summary_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for record in records:
                pid = record.get("person_id", "")
                if not pid:
                    continue
                counts = _parse_behavior_counts(record.get("behavior_counts", {}))
                total = record.get("total_events", 0)
                if pid not in merged:
                    merged[pid] = {"_total_events": 0, "_behaviors": {}}
                merged[pid]["_total_events"] += total
                for b, c in counts.items():
                    merged[pid]["_behaviors"][b] = merged[pid]["_behaviors"].get(b, 0) + c
        return merged

    @staticmethod
    def _grade_distribution(evaluations: List[Dict]) -> Dict[str, int]:
        """统计各等级人数。"""
        dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for e in evaluations:
            grade = e.get("grade", "F")
            if grade in dist:
                dist[grade] += 1
        return dist

    # ═══════════════════════════════════════════════════════════════
    # 持久化与导出
    # ═══════════════════════════════════════════════════════════════

    def save_evaluation(self):
        """将当前评估结果保存到 detections/<date>/evaluation/ 目录。"""
        if not self.evaluation_date or not self.person_evaluations:
            return

        eval_dir = Path("detections") / self.evaluation_date / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # JSON 完整数据
        json_path = eval_dir / "person_evaluations.json"
        json_path.write_text(
            json.dumps(self.person_evaluations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # CSV 表格
        csv_path = eval_dir / "person_evaluations.csv"
        fieldnames = [
            "person_id", "participation_score", "focus_score",
            "anomaly_score", "emotion_score", "assignment_score",
            "self_test_score", "lab_score", "weighted_total", "grade",
            "total_detections",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.person_evaluations)

        # 配置快照
        config = {
            "weights": {
                "participation": self.participation_weight,
                "focus": self.focus_weight,
                "anomaly": self.anomaly_weight,
                "emotion": self.emotion_weight,
                "assignment": self.assignment_weight,
                "self_test": self.self_test_weight,
                "lab": self.lab_weight,
            },
            "participation_map": json.loads(self.participation_map_json),
            "focus_map": json.loads(self.focus_map_json),
            "anomaly_penalty": json.loads(self.anomaly_penalty_json),
            "emotion_map": json.loads(self.emotion_map_json),
            "use_mock_teammate_data": self.use_mock_teammate_data,
        }
        (eval_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.evaluation_message = f"评估结果已保存至 {eval_dir}"

    def export_csv(self) -> str:
        """导出评估结果 CSV 文本（供 rx.download 使用）。"""
        if not self.person_evaluations:
            return ""

        output = io.StringIO()
        fieldnames = [
            "person_id", "participation_score", "focus_score",
            "anomaly_score", "emotion_score", "assignment_score",
            "self_test_score", "lab_score", "weighted_total", "grade",
            "total_detections",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.person_evaluations)

        return output.getvalue()

    # ═══════════════════════════════════════════════════════════════
    # 当前视图的计算属性
    # ═══════════════════════════════════════════════════════════════

    @rx.var
    def current_evaluations(self) -> List[Dict[str, Any]]:
        """Return evaluations for the active view (single-date or aggregated)."""
        if self.show_all_dates:
            return self.aggregated_evaluations
        return self.person_evaluations

    @rx.var
    def current_student_count(self) -> int:
        if self.show_all_dates:
            return self.aggregated_student_count
        return self.student_count

    @rx.var
    def current_class_average(self) -> float:
        if self.show_all_dates:
            return self.aggregated_class_average
        return self.class_average

    @rx.var
    def current_view_label(self) -> str:
        return "全部日期汇总" if self.show_all_dates else "单日评估"

    # ═══════════════════════════════════════════════════════════════
    # 权重的百分比显示（用于 UI）
    # ═══════════════════════════════════════════════════════════════

    @rx.var
    def weight_sum(self) -> float:
        return round(
            self.participation_weight
            + self.focus_weight
            + self.anomaly_weight
            + self.emotion_weight
            + self.assignment_weight
            + self.self_test_weight
            + self.lab_weight,
            2,
        )

    @rx.var
    def weight_valid(self) -> bool:
        return abs(self.weight_sum - 1.0) < 0.02

    @rx.var
    def grade_color(self) -> str:
        """根据当前视图均分返回颜色。"""
        avg = self.current_class_average
        if avg >= 90:
            return "#22c55e"  # green
        elif avg >= 80:
            return "#22d3ee"  # cyan
        elif avg >= 70:
            return "#eab308"  # yellow
        elif avg >= 60:
            return "#f97316"  # orange
        return "#ef4444"  # red

    @rx.var
    def filtered_evaluations(self) -> List[Dict[str, Any]]:
        """根据 selected_student 筛选后的评估列表。"""
        evals = self.current_evaluations
        if self.selected_student == "All":
            return evals
        return [
            e for e in evals
            if e["person_id"] == self.selected_student
        ]

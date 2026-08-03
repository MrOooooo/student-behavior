"""
学生行为识别成绩评价页面 UI 组件
"""

import reflex as rx

from object_cheating.states.evaluation_state import (
    EvaluationState,
)

# ── 权重预设按钮 ──────────────────────────────────────────────────

_WEIGHT_PRESETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _weight_buttons(factor_key: str) -> rx.Component:
    return rx.el.div(
        *[
            rx.button(
                f"{int(v*100)}%",
                on_click=lambda v=v: EvaluationState.set_weight(factor_key, str(v)),
                class_name=(
                    "rounded border border-cyan-300/25 bg-cyan-300/8 px-1.5 py-0.5 text-xs "
                    "text-cyan-200 hover:bg-cyan-300/20"
                ),
            )
            for v in _WEIGHT_PRESETS
        ],
        class_name="flex flex-wrap gap-1 mt-1",
    )


def _weight_row(factor_key: str, label: str, weight_value) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.span(label, class_name="text-sm text-cyan-100 font-medium w-44 flex-shrink-0"),
            rx.input(
                value=weight_value,
                type="number",
                min=0,
                max=1.0,
                step=0.05,
                width="64px",
                height="28px",
                text_align="center",
                color="#dffbff",
                background_color="#03142f",
                border="1px solid rgba(34,211,238,0.45)",
                border_radius="md",
                on_change=lambda v, k=factor_key: EvaluationState.set_weight(k, v),
            ),
            class_name="flex items-center gap-2",
        ),
        _weight_buttons(factor_key),
        class_name="py-2 border-b border-cyan-400/10 last:border-0",
    )


# ── 子组件 ───────────────────────────────────────────────────────

def _empty_state() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.icon("clipboard-list", size=48, class_name="text-slate-500 mx-auto mb-4"),
            rx.el.p("暂无评估数据", class_name="text-xl text-slate-400 font-semibold mb-2"),
            rx.el.p(
                "请先在主页运行行为检测，然后回到此页面查看学生行为成绩评估。",
                class_name="text-sm text-slate-500 max-w-md",
            ),
            class_name="text-center py-20",
        ),
    )


def _summary_stats() -> rx.Component:
    return rx.cond(
        EvaluationState.current_student_count > 0,
        rx.el.div(
            rx.el.div(
                rx.el.p("学生人数", class_name="text-xs tracking-wide text-cyan-300/70 uppercase"),
                rx.el.p(
                    EvaluationState.current_student_count,
                    class_name="text-3xl font-bold text-cyan-100",
                ),
                class_name="text-center px-6 py-3",
            ),
            rx.el.div(
                rx.el.p("班级均分", class_name="text-xs tracking-wide text-cyan-300/70 uppercase"),
                rx.el.p(
                    EvaluationState.current_class_average,
                    class_name="text-3xl font-bold",
                    style={"color": EvaluationState.grade_color},
                ),
                class_name="text-center px-6 py-3",
            ),
            rx.el.div(
                rx.el.p("等级分布", class_name="text-xs tracking-wide text-cyan-300/70 uppercase mb-1"),
                rx.el.div(
                    rx.foreach(
                        EvaluationState.current_evaluations,
                        lambda e: rx.el.span(
                            e["grade"],
                            class_name=rx.cond(
                                e["grade"] == "A",
                                "inline-block rounded px-1.5 py-0.5 text-xs font-bold border border-emerald-400/60 bg-emerald-400/15 text-emerald-300 mx-0.5",
                                rx.cond(
                                    e["grade"] == "B",
                                    "inline-block rounded px-1.5 py-0.5 text-xs font-bold border border-cyan-400/60 bg-cyan-400/15 text-cyan-300 mx-0.5",
                                    rx.cond(
                                        e["grade"] == "C",
                                        "inline-block rounded px-1.5 py-0.5 text-xs font-bold border border-yellow-400/60 bg-yellow-400/15 text-yellow-300 mx-0.5",
                                        rx.cond(
                                            e["grade"] == "D",
                                            "inline-block rounded px-1.5 py-0.5 text-xs font-bold border border-orange-400/60 bg-orange-400/15 text-orange-300 mx-0.5",
                                            "inline-block rounded px-1.5 py-0.5 text-xs font-bold border border-red-400/60 bg-red-400/15 text-red-300 mx-0.5",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    class_name="flex flex-wrap justify-center gap-1",
                ),
                class_name="text-center px-6 py-3",
            ),
            class_name="mb-6 grid grid-cols-3 gap-4 rounded-2xl border border-cyan-400/30 bg-[#061b3d]/80 p-4 shadow-[0_0_24px_rgba(34,211,238,0.16)] backdrop-blur",
        ),
        rx.fragment(),
    )


def _student_card(person: dict) -> rx.Component:
    """单学生评分卡片。支持显示跨天信息。"""
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                # 人脸照片 — 有小图显示小图，没有就用默认头像
                rx.cond(
                    person["face_image"] != "",
                    rx.el.img(
                        src=person["face_image"],
                        class_name="h-10 w-10 rounded-full border border-cyan-400/40 object-cover shadow-[0_0_10px_rgba(34,211,238,0.25)]",
                    ),
                    rx.el.div(
                        rx.el.span("👤", class_name="text-lg"),
                        class_name="flex h-10 w-10 items-center justify-center rounded-full bg-cyan-400/10 border border-cyan-400/30",
                    ),
                ),
                rx.el.div(
                    rx.el.p(person["person_id"], class_name="font-semibold text-cyan-100 text-sm"),
                    rx.el.p(
                        rx.el.span("检测: ", class_name="text-xs text-cyan-300/50"),
                        person["total_detections"],
                        rx.el.span(" 次", class_name="text-xs text-cyan-300/50"),
                        class_name="text-xs text-cyan-300/50",
                    ),
                    class_name="ml-2",
                ),
                class_name="flex items-center",
            ),
            rx.el.span(
                person["grade"],
                class_name=rx.cond(
                    person["grade"] == "A",
                    "rounded-lg border px-3 py-1 text-lg font-extrabold border-emerald-400/60 bg-emerald-400/15 text-emerald-300",
                    rx.cond(
                        person["grade"] == "B",
                        "rounded-lg border px-3 py-1 text-lg font-extrabold border-cyan-400/60 bg-cyan-400/15 text-cyan-300",
                        rx.cond(
                            person["grade"] == "C",
                            "rounded-lg border px-3 py-1 text-lg font-extrabold border-yellow-400/60 bg-yellow-400/15 text-yellow-300",
                            rx.cond(
                                person["grade"] == "D",
                                "rounded-lg border px-3 py-1 text-lg font-extrabold border-orange-400/60 bg-orange-400/15 text-orange-300",
                                "rounded-lg border px-3 py-1 text-lg font-extrabold border-red-400/60 bg-red-400/15 text-red-300",
                            ),
                        ),
                    ),
                ),
            ),
            class_name="flex items-center justify-between mb-3",
        ),
        # ── 总分（可点击跳转归档查看行为图片） ──
        rx.el.div(
            rx.el.p(
                person["weighted_total"],
                class_name=rx.cond(
                    person["grade"] == "A", "text-3xl font-extrabold text-emerald-400 group-hover:scale-110 transition-transform",
                    rx.cond(
                        person["grade"] == "B", "text-3xl font-extrabold text-cyan-400 group-hover:scale-110 transition-transform",
                        rx.cond(
                            person["grade"] == "C", "text-3xl font-extrabold text-yellow-400 group-hover:scale-110 transition-transform",
                            rx.cond(
                                person["grade"] == "D", "text-3xl font-extrabold text-orange-400 group-hover:scale-110 transition-transform",
                                "text-3xl font-extrabold text-red-400 group-hover:scale-110 transition-transform",
                            ),
                        ),
                    ),
                ),
            ),
            rx.el.p(
                rx.el.span("行为成绩总分", class_name="text-xs text-cyan-300/50"),
                rx.el.span(" 🔍", class_name="text-xs text-cyan-300/30 ml-1"),
                class_name="mt-1",
            ),
            on_click=EvaluationState.goto_student_archive(person["person_id"]),
            class_name="group text-center mb-3 cursor-pointer rounded-xl py-2 -mx-2 hover:bg-cyan-400/5 transition-all duration-200",
            title="点击查看该学生行为检测图片",
        ),
        # ── 分项得分（纯文字，不做颜色条件判断，避免 foreach item 比较限制） ──
        rx.el.div(
            _factor_text("参与度", person["participation_score"]),
            _factor_text("专注度", person["focus_score"]),
            _factor_text("行为规范", person["anomaly_score"]),
            _factor_text("情绪状态", person["emotion_score"]),
            _factor_text("作业成绩", person["assignment_score"]),
            _factor_text("自测正确率", person["self_test_score"]),
            _factor_text("场景实践", person["lab_score"]),
            class_name="space-y-1",
        ),
        class_name=(
            "rounded-2xl border border-cyan-400/25 bg-[#08264f]/80 p-4 "
            "shadow-[0_0_20px_rgba(34,211,238,0.12)] backdrop-blur "
            "hover:shadow-[0_0_30px_rgba(34,211,238,0.22)] transition-all duration-200"
        ),
    )


def _factor_text(label: str, score) -> rx.Component:
    """因子分文字行 — 纯文本，不做比较运算避免 ObjectItemOperation 限制。"""
    return rx.el.div(
        rx.el.span(label, class_name="text-xs text-cyan-300/60 w-20 flex-shrink-0"),
        rx.el.span(score, class_name="text-xs text-cyan-200/80 font-mono"),
        class_name="flex items-center gap-1",
    )


# ── 行为映射编辑器子面板 ──────────────────────────────────────────

def _mapping_entry_row(item: dict) -> rx.Component:
    """Single behavior→score row — click score to edit."""
    return rx.el.div(
        # Behavior name
        rx.el.span(
            item["behavior"],
            class_name="inline-block w-[130px] truncate rounded-md border px-2 py-1 text-xs",
            style={
                "color": "#94a3b8",
                "background_color": "#020817",
                "border": "1px solid rgba(34,211,238,0.20)",
            },
        ),
        rx.el.span("→", class_name="text-cyan-300/40 mx-1 text-xs"),
        # Score — click to populate edit form
        rx.el.span(
            rx.el.span(item["score"], class_name="text-xs text-cyan-200 font-mono"),
            rx.el.span(" 分", class_name="text-[10px] text-cyan-300/40"),
            class_name="inline-block w-[64px] text-center rounded-md border px-1 py-1 cursor-pointer hover:bg-cyan-400/10 transition-colors",
            style={
                "background_color": "#03142f",
                "border": "1px solid rgba(34,211,238,0.35)",
            },
            on_click=lambda b=item["behavior"]: EvaluationState.start_edit_behavior(b),
            title="点击编辑分值",
        ),
        rx.button(
            "✕",
            on_click=lambda b=item["behavior"]: EvaluationState.remove_behavior_entry(b),
            class_name="ml-2 rounded border border-red-400/30 bg-red-400/8 px-2 py-0.5 text-xs text-red-300 hover:bg-red-400/20",
            title="删除此行为映射",
        ),
        class_name="flex items-center gap-1 py-1",
    )


def _behavior_mapping_panel() -> rx.Component:
    """Collapsible sub-panel for editing per-factor behavior→score mappings."""
    return rx.el.div(
        # ── 分隔线 + 标题 ──
        rx.el.div(
            rx.el.span("行为分数映射", class_name="text-base font-bold text-cyan-100"),
            rx.el.span(
                "（每种行为参与计分时的分值，0 表示不计分）",
                class_name="text-xs text-cyan-300/50 ml-2",
            ),
            rx.button(
                rx.cond(
                    EvaluationState.show_behavior_mapping,
                    "收起 ⬆",
                    "展开 ⬇",
                ),
                on_click=EvaluationState.toggle_behavior_mapping,
                class_name="ml-auto rounded-lg border border-cyan-300/30 bg-cyan-300/8 px-3 py-1 text-xs text-cyan-200 hover:bg-cyan-300/20",
            ),
            class_name="flex items-center mb-3 pt-3 border-t border-cyan-400/25",
        ),
        # ── 因子选择 tabs ──
        rx.cond(
            EvaluationState.show_behavior_mapping,
            rx.el.div(
                rx.el.div(
                    rx.foreach(
                        EvaluationState.behavior_factors_for_ui,
                        lambda fct: rx.button(
                            fct["label"],
                            on_click=EvaluationState.select_mapping_factor(fct["key"]),
                            class_name=rx.cond(
                                EvaluationState.selected_mapping_factor == fct["key"],
                                "rounded-lg border border-emerald-400/50 bg-emerald-400/12 px-3 py-1.5 text-xs font-semibold text-emerald-200",
                                "rounded-lg border border-cyan-300/25 bg-cyan-300/6 px-3 py-1.5 text-xs text-cyan-300/70 hover:bg-cyan-300/15",
                            ),
                        ),
                    ),
                    class_name="flex flex-wrap gap-2 mb-3",
                ),
                # ── 映射列表 ──
                rx.el.div(
                    rx.foreach(
                        EvaluationState.current_behavior_mapping,
                        _mapping_entry_row,
                    ),
                    class_name="max-h-48 overflow-y-auto space-y-0.5 rounded-lg bg-slate-900/40 p-3 border border-cyan-400/15",
                ),
                # ── 添加/编辑行为 ──
                rx.el.div(
                    # Edit mode indicator
                    rx.cond(
                        EvaluationState.editing_behavior != "",
                        rx.el.div(
                            rx.el.span("正在编辑: ", class_name="text-xs text-amber-300/80"),
                            rx.el.span(EvaluationState.editing_behavior, class_name="text-xs font-semibold text-amber-200"),
                            rx.el.span("（修改分值后点击保存）", class_name="text-xs text-amber-300/50 ml-1"),
                            class_name="mb-2",
                        ),
                        rx.fragment(),
                    ),
                    rx.el.div(
                        rx.input(
                            value=EvaluationState.new_behavior_name,
                            placeholder="行为名称",
                            on_change=EvaluationState.set_new_behavior_name,
                            disabled=EvaluationState.editing_behavior != "",
                            width="130px",
                            height="28px",
                            color="#dffbff",
                            background_color="#03142f",
                            border="1px solid rgba(34,211,238,0.45)",
                            border_radius="md",
                            class_name="text-xs",
                        ),
                        rx.el.span("→", class_name="text-cyan-300/40 mx-1 text-xs"),
                        rx.input(
                            value=EvaluationState.new_behavior_score,
                            placeholder="分值",
                            type="number",
                            on_change=EvaluationState.set_new_behavior_score,
                            width="60px",
                            height="28px",
                            text_align="center",
                            color="#dffbff",
                            background_color="#03142f",
                            border="1px solid rgba(34,211,238,0.45)",
                            border_radius="md",
                            class_name="text-xs",
                        ),
                        rx.cond(
                            EvaluationState.editing_behavior != "",
                            rx.el.div(
                                rx.button(
                                    "💾 保存",
                                    on_click=EvaluationState.add_behavior_entry,
                                    class_name="ml-2 rounded-lg border border-amber-400/50 bg-amber-400/10 px-3 py-1 text-xs text-amber-200 hover:bg-amber-400/20",
                                ),
                                rx.button(
                                    "取消",
                                    on_click=EvaluationState.cancel_edit_behavior,
                                    class_name="ml-1 rounded-lg border border-slate-500 bg-slate-700/50 px-3 py-1 text-xs text-slate-300 hover:bg-slate-600/60",
                                ),
                                class_name="flex items-center gap-1",
                            ),
                            rx.button(
                                "＋ 添加",
                                on_click=EvaluationState.add_behavior_entry,
                                class_name="ml-2 rounded-lg border border-emerald-400/50 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200 hover:bg-emerald-400/20",
                            ),
                        ),
                        rx.button(
                            "重置默认",
                            on_click=EvaluationState.reset_behavior_mapping,
                            class_name="ml-auto rounded-lg border border-amber-400/35 bg-amber-400/8 px-3 py-1 text-xs text-amber-300/80 hover:bg-amber-400/18",
                        ),
                        class_name="flex items-center gap-1",
                    ),
                    class_name="mt-2",
                ),
                class_name="mb-4",
            ),
            rx.fragment(),
        ),
        class_name="mb-4",
    )


# ── 权重配置弹窗 ─────────────────────────────────────────────────

def _weight_config_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("权重与行为分数配置", class_name="text-lg font-bold text-cyan-100"),
            rx.dialog.description(
                "调整各因子权重比例（总和 1.0）及每个因子下的行为→分值映射。",
                class_name="text-sm text-cyan-300/70 mb-4",
            ),
            rx.el.div(
                _weight_row("participation", "课堂参与度", EvaluationState.participation_weight),
                _weight_row("focus", "课堂专注度", EvaluationState.focus_weight),
                _weight_row("anomaly", "行为规范", EvaluationState.anomaly_weight),
                _weight_row("emotion", "情绪状态", EvaluationState.emotion_weight),
                _weight_row("assignment", "作业成绩(智教)", EvaluationState.assignment_weight),
                _weight_row("self_test", "自测正确率(慧学)", EvaluationState.self_test_weight),
                _weight_row("lab", "场景实践(智能教学)", EvaluationState.lab_weight),
                class_name="space-y-1 mb-2 max-h-72 overflow-y-auto",
            ),
            rx.el.div(
                rx.el.span("权重总和：", class_name="text-sm text-cyan-200"),
                rx.el.span(
                    EvaluationState.weight_sum,
                    class_name=rx.cond(
                        EvaluationState.weight_valid,
                        "text-sm font-bold text-emerald-400",
                        "text-sm font-bold text-red-400",
                    ),
                ),
                rx.cond(
                    ~EvaluationState.weight_valid,
                    rx.el.span("  ⚠ 总和不为 1.0", class_name="text-xs text-red-400 ml-1"),
                ),
                class_name="mb-4 rounded-lg bg-slate-900/60 px-4 py-2",
            ),
            # ── 行为分数映射（二级面板） ──
            _behavior_mapping_panel(),
            rx.el.div(
                rx.el.span("使用模拟队友数据：", class_name="text-sm text-cyan-200 mr-3"),
                rx.switch(
                    checked=EvaluationState.use_mock_teammate_data,
                    on_change=EvaluationState.toggle_mock_data,
                    color_scheme="cyan",
                ),
                class_name="flex items-center mb-4",
            ),
            rx.el.div(
                rx.dialog.close(
                    rx.button("关闭", class_name="rounded-lg border border-cyan-300/35 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100 hover:bg-cyan-300/20"),
                ),
                rx.button(
                    "保存评估",
                    on_click=EvaluationState.save_evaluation,
                    class_name="rounded-lg border border-emerald-300/50 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-100 hover:bg-emerald-300/20 ml-2",
                ),
                rx.button(
                    "重置默认",
                    on_click=EvaluationState.reset_weights,
                    class_name="rounded-lg border border-slate-500 bg-slate-700/50 px-4 py-2 text-sm text-slate-300 hover:bg-slate-600/60 ml-2",
                ),
                class_name="flex items-center gap-2",
            ),
            class_name=(
                "rounded-2xl border border-cyan-400/30 bg-[#061b3d]/95 p-6 "
                "shadow-[0_0_48px_rgba(34,211,238,0.22)] backdrop-blur max-w-2xl max-h-[92vh] overflow-y-auto"
            ),
        ),
        open=EvaluationState.show_weight_config,
        on_open_change=EvaluationState.toggle_weight_config,
    )


# ── 历史汇总面板 ─────────────────────────────────────────────────

def _history_summary_panel() -> rx.Component:
    return rx.el.div(
        rx.el.h3("历史评估汇总", class_name="text-lg font-bold text-cyan-100 mb-3"),
        rx.cond(
            EvaluationState.all_dates_summary.length() > 0,
            rx.el.div(
                rx.foreach(
                    EvaluationState.all_dates_summary,
                    lambda item: rx.el.div(
                        rx.el.div(
                            rx.el.span(item["date"], class_name="text-sm font-semibold text-cyan-200"),
                            rx.button(
                                "查看",
                                on_click=EvaluationState.set_evaluation_date(item["date"]),
                                class_name="ml-auto rounded-md border border-cyan-300/30 bg-cyan-300/8 px-3 py-1 text-xs text-cyan-200 hover:bg-cyan-300/20",
                            ),
                            class_name="flex items-center",
                        ),
                        class_name="py-2 border-b border-cyan-400/10 last:border-0",
                    ),
                ),
                class_name="rounded-xl border border-cyan-400/20 bg-[#061b3d]/60 p-4 max-h-80 overflow-y-auto",
            ),
            rx.el.p("暂无历史数据", class_name="text-sm text-slate-500 text-center py-6"),
        ),
        class_name="rounded-2xl border border-cyan-400/30 bg-[#061b3d]/80 p-4 shadow-[0_0_24px_rgba(34,211,238,0.14)] backdrop-blur",
    )


# ═════════════════════════════════════════════════════════════════
# 主页面
# ═════════════════════════════════════════════════════════════════

def evaluation_page() -> rx.Component:
    return rx.box(
        _weight_config_dialog(),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.link(
                        rx.button(
                            "← 返回主页",
                            class_name="rounded-lg border border-cyan-300/35 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 shadow-[0_0_14px_rgba(34,211,238,0.18)] hover:bg-cyan-300/20",
                        ),
                        href="/",
                    ),
                    rx.el.h1(
                        "学生行为成绩总评",
                        class_name="text-2xl font-bold text-cyan-50 drop-shadow-[0_0_14px_rgba(34,211,238,0.45)]",
                    ),
                    class_name="flex items-center gap-4",
                ),
                rx.el.div(
                    rx.el.div(
                        rx.el.span(
                            EvaluationState.current_view_label,
                            class_name="text-sm text-emerald-300/80 font-medium mr-2",
                        ),
                        rx.select(
                            EvaluationState.evaluation_dates,
                            value=EvaluationState.evaluation_date,
                            on_change=EvaluationState.set_evaluation_date,
                            disabled=EvaluationState.show_all_dates,
                            class_name=(
                                "rounded-lg border border-cyan-300/35 bg-cyan-300/10 "
                                "px-3 py-1.5 text-sm text-cyan-100 "
                                "disabled:opacity-40 disabled:cursor-not-allowed"
                            ),
                        ),
                        class_name="flex items-center",
                    ),
                    rx.button(
                        "刷新",
                        on_click=EvaluationState.load_dates,
                        class_name="rounded-lg border border-cyan-300/35 bg-cyan-300/10 px-3 py-1.5 text-sm text-cyan-200 hover:bg-cyan-300/20 ml-2",
                    ),
                    rx.button(
                        rx.cond(
                            EvaluationState.show_all_dates,
                            rx.el.span("📅 单日评估"),
                            rx.el.span("📊 全部汇总"),
                        ),
                        on_click=EvaluationState.toggle_view_mode,
                        class_name=rx.cond(
                            EvaluationState.show_all_dates,
                            "rounded-lg border border-amber-300/50 bg-amber-400/10 px-4 py-2 text-sm font-semibold text-amber-100 shadow-[0_0_18px_rgba(234,179,8,0.22)] hover:bg-amber-300/20 ml-3",
                            "rounded-lg border border-emerald-300/50 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 shadow-[0_0_18px_rgba(16,185,129,0.22)] hover:bg-emerald-300/20 ml-3",
                        ),
                    ),
                    rx.button(
                        "⚙ 权重配置",
                        on_click=EvaluationState.toggle_weight_config,
                        class_name="rounded-lg border border-emerald-300/50 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 shadow-[0_0_18px_rgba(16,185,129,0.22)] hover:bg-emerald-300/20 ml-3",
                    ),
                    rx.button(
                        "📥 导出 CSV",
                        on_click=EvaluationState.save_evaluation,
                        class_name="rounded-lg border border-slate-400/50 bg-slate-500/10 px-4 py-2 text-sm text-slate-300 hover:bg-slate-400/20 ml-2",
                    ),
                    class_name="flex items-center gap-2",
                ),
                class_name="mb-6 flex items-center justify-between gap-4 border-b border-cyan-400/30 pb-4",
            ),
            rx.cond(
                EvaluationState.evaluation_message != "",
                rx.el.div(
                    rx.el.p(EvaluationState.evaluation_message, class_name="text-sm text-cyan-300/80"),
                    class_name="mb-4 rounded-lg bg-cyan-400/5 border border-cyan-400/20 px-4 py-2",
                ),
                rx.fragment(),
            ),
            _summary_stats(),
            rx.el.div(
                rx.el.div(
                    rx.cond(
                        EvaluationState.current_student_count > 0,
                        rx.cond(
                            EvaluationState.loading_evaluation,
                            rx.el.div(
                                rx.el.p("正在计算评分...", class_name="text-cyan-300/70 text-center py-12"),
                            ),
                            rx.el.div(
                                rx.foreach(
                                    EvaluationState.current_evaluations,
                                    _student_card,
                                ),
                                class_name="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
                            ),
                        ),
                        _empty_state(),
                    ),
                    class_name="flex-1",
                ),
                rx.el.div(
                    _history_summary_panel(),
                    class_name="w-72 ml-6 flex-shrink-0",
                ),
                class_name="flex",
            ),
            class_name="mx-auto max-w-[1500px] px-5 py-6",
        ),
        rx.el.div(class_name="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top,#0b4f9f55,transparent_34%),linear-gradient(135deg,#020817_0%,#061a3d_48%,#020817_100%)]"),
        rx.el.div(class_name="pointer-events-none fixed inset-0 -z-10 opacity-30 [background-image:linear-gradient(rgba(34,211,238,0.18)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.18)_1px,transparent_1px)] [background-size:42px_42px]"),
    )

"""
主页学生行为成绩快速概览横条组件
"""

import reflex as rx

from object_cheating.states.evaluation_state import EvaluationState


def evaluation_summary() -> rx.Component:
    """主页快速评估概览横条 - 仅在有评估数据时显示。"""
    return rx.cond(
        (EvaluationState.current_student_count > 0) & (EvaluationState.evaluation_date != ""),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.icon("bar-chart-3", size=18, class_name="text-emerald-400"),
                    rx.el.span("学生行为成绩概要", class_name="text-sm font-semibold text-cyan-100 ml-2"),
                    rx.el.span(
                        EvaluationState.current_view_label,
                        class_name="text-xs text-cyan-300/50 ml-2 px-1.5 py-0.5 rounded border border-cyan-300/20 bg-cyan-300/5",
                    ),
                    class_name="flex items-center",
                ),
                rx.el.div(
                    rx.el.span("日期: ", class_name="text-sm text-cyan-300/60 mx-3"),
                    rx.el.span(EvaluationState.evaluation_date, class_name="text-sm text-cyan-300/60"),
                    rx.el.span(" | ", class_name="text-cyan-400/30"),
                    rx.el.span(EvaluationState.current_student_count, class_name="text-sm text-cyan-300/80 mx-3"),
                    rx.el.span(" 名学生  ", class_name="text-sm text-cyan-300/80"),
                    rx.el.span(" | ", class_name="text-cyan-400/30"),
                    rx.el.span("  班级均分: ", class_name="text-sm text-cyan-300/80 ml-3"),
                    rx.el.span(
                        EvaluationState.current_class_average,
                        class_name="text-sm font-extrabold",
                        style={"color": EvaluationState.grade_color},
                    ),
                    class_name="flex items-center",
                ),
                rx.link(
                    rx.button(
                        "查看完整评价 →",
                        class_name="rounded-lg border border-emerald-300/50 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-100 hover:bg-emerald-300/20",
                    ),
                    href="/evaluation",
                ),
                class_name="flex items-center justify-between px-4 py-2",
            ),
            class_name=(
                "rounded-xl border border-emerald-400/25 bg-emerald-400/5 "
                "shadow-[0_0_18px_rgba(16,185,129,0.12)] backdrop-blur mb-4"
            ),
        ),
        rx.fragment(),
    )

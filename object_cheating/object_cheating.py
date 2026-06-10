import reflex as rx

from object_cheating.components.camera_feed import camera_feed
from object_cheating.components.controls import controls
from object_cheating.components.treshold import threshold
from object_cheating.components.stats_panel import stats_panel
from object_cheating.components.behavior_panel import behavior_panel
from object_cheating.components.coordinate_panel import coordinate_panel
from object_cheating.components.table import tables_v2
from object_cheating.components.input_panel import input_panel
from object_cheating.components.warning_dialog import warning_dialog
from object_cheating.components.delete_dialog import delete_dialog
from object_cheating.states.camera_state import CameraState

def index() -> rx.Component:
    return rx.box(
        # Warning dialog at root level for proper overlay
        warning_dialog(),
        delete_dialog(),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.el.div(
                        rx.el.p(
                            "视觉教室监视器",
                            class_name="text-xs tracking-[0.45em] text-cyan-300/80 uppercase"
                        ),
                        rx.el.h1(
                            "学生行为智能检测",
                            class_name="text-3xl font-bold text-cyan-50 drop-shadow-[0_0_14px_rgba(34,211,238,0.55)]"
                        ),
                    ),
                    rx.button(
                        rx.cond(
                            CameraState.right_panel_collapsed,
                            "显示右侧栏",
                            "隐藏右侧栏"
                        ),
                        on_click=CameraState.toggle_right_panel,
                        class_name="rounded-lg border border-cyan-300/50 bg-cyan-400/10 px-4 py-2 text-sm font-semibold text-cyan-100 shadow-[0_0_18px_rgba(34,211,238,0.25)] hover:bg-cyan-300/20",
                    ),
                    class_name="mb-6 flex items-center justify-between gap-4 border-b border-cyan-400/30 pb-4"
                ),
                rx.el.div(
                    # Left Section: Camera Feed, Controls, and Table in separate sections
                    rx.el.div(
                        rx.el.div(
                            camera_feed(),
                            controls(),
                            class_name="rounded-2xl border border-cyan-400/30 bg-[#061b3d]/80 p-4 shadow-[0_0_32px_rgba(34,211,238,0.18)] backdrop-blur space-y-4"
                        ),
                        rx.el.div(
                            tables_v2(),
                            class_name="rounded-2xl border border-cyan-400/30 bg-[#061b3d]/80 p-4 shadow-[0_0_32px_rgba(34,211,238,0.14)] backdrop-blur space-y-4"
                        ),
                        class_name=rx.cond(
                            CameraState.right_panel_collapsed,
                            "w-full space-y-4 transition-all duration-300",
                            "w-2/3 pr-4 space-y-4 transition-all duration-300"
                        )
                    ),
                    # Right Section: Threshold, Show Label Name, Behavior, Coordinate Panels
                    rx.cond(
                        CameraState.right_panel_collapsed,
                        rx.fragment(),
                        rx.el.div(
                            threshold(),
                            stats_panel(),
                            behavior_panel(),
                            coordinate_panel(),
                            input_panel(),
                            class_name="w-1/3 space-y-4 transition-all duration-300"
                        )
                    ),
                    class_name="flex"
                ),
                class_name="mx-auto max-w-[1500px] px-5 py-6"
            ),
            class_name="min-h-screen bg-[#020817] text-cyan-50"
        ),
        rx.el.div(class_name="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top,#0b4f9f55,transparent_34%),linear-gradient(135deg,#020817_0%,#061a3d_48%,#020817_100%)]"),
        rx.el.div(class_name="pointer-events-none fixed inset-0 -z-10 opacity-30 [background-image:linear-gradient(rgba(34,211,238,0.18)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.18)_1px,transparent_1px)] [background-size:42px_42px]"),
    )

app = rx.App(
    theme=rx.theme(
        accent_color="grass",
    )
)
app.add_page(index)

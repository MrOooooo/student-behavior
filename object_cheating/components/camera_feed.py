import reflex as rx
from object_cheating.states.camera_state import CameraState


def camera_feed() -> rx.Component:
    display_class = rx.cond(
        CameraState.right_panel_collapsed,
        "w-full h-[680px] rounded-xl overflow-hidden border border-cyan-400/40 bg-[#020817] shadow-inner shadow-cyan-950",
        "w-full h-[520px] rounded-xl overflow-hidden border border-cyan-400/40 bg-[#020817] shadow-inner shadow-cyan-950",
    )
    empty_class = rx.cond(
        CameraState.right_panel_collapsed,
        "w-full h-[680px] bg-[#020817] flex items-center justify-center text-cyan-300/60 text-xl font-medium rounded-xl",
        "w-full h-[520px] bg-[#020817] flex items-center justify-center text-cyan-300/60 text-xl font-medium rounded-xl",
    )
    return rx.el.div(
        rx.el.div(
            rx.cond(
                CameraState.current_frame,
                rx.el.img(src=CameraState.current_frame, class_name="w-full h-full object-contain rounded-xl shadow-[0_0_34px_rgba(34,211,238,0.24)]"),
                rx.el.div("未加载画面", class_name=empty_class),
            ),
            class_name=display_class,
        ),
        rx.cond(CameraState.error_message, rx.el.p(CameraState.error_message, class_name="text-red-500 mt-2 text-sm")),
        class_name="w-full",
    )

import reflex as rx
from object_cheating.states.camera_state import CameraState


def coordinate_panel() -> rx.Component:
    return rx.el.div(
        rx.el.h3("边界框坐标", class_name="text-lg font-semibold mb-2 text-cyan-50"),
        rx.el.div(
            rx.el.div(
                rx.el.span(rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), f"xmin: {CameraState.highest_conf_xmin}", "xmin: N/A"), class_name="text-cyan-100"),
                rx.el.span(rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), f"ymin: {CameraState.highest_conf_ymin}", "ymin: N/A"), class_name="text-cyan-100"),
                class_name="flex justify-between",
            ),
            rx.el.div(
                rx.el.span(rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), f"xmax: {CameraState.highest_conf_xmax}", "xmax: N/A"), class_name="text-cyan-100"),
                rx.el.span(rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), f"ymax: {CameraState.highest_conf_ymax}", "ymax: N/A"), class_name="text-cyan-100"),
                class_name="flex justify-between mt-2",
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl border border-cyan-400/25",
        ),
        class_name="border border-cyan-400/30 bg-[#061b3d]/80 p-4 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] w-full backdrop-blur",
    )

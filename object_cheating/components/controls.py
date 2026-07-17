import reflex as rx
from object_cheating.states.camera_state import CameraState


def model_navigation() -> rx.Component:
    """Komponen navigasi model."""
    return rx.hstack(
        rx.icon_button(
            rx.icon("chevron-left"),
            on_click=lambda: CameraState.try_change_model(CameraState.active_model - 1),
            variant="surface",
            height="30px",
            width="30px",
            disabled=CameraState.active_model == 1,
        ),
        rx.badge(
            rx.center(
                rx.text(f"Model {CameraState.active_model}"),
                width="100%",
                height="28px",
            ),
            variant="surface",
            min_width="100px",
            text_align="center",
            class_name="border border-cyan-300/50 bg-cyan-300/10 text-cyan-50",
        ),
        rx.icon_button(
            rx.icon("chevron-right"),
            on_click=lambda: CameraState.try_change_model(CameraState.active_model + 1),
            variant="surface",
            height="30px",
            width="30px",
            disabled=CameraState.active_model == 7,
        ),
        spacing="2",
        align="center",
    )


def controls() -> rx.Component:
    """Controls component for detection and model selection."""
    media_active = rx.cond(
        (CameraState.camera_active |
         (CameraState.current_frame != "") |
         CameraState.video_playing),
        True,
        False
    )
    return rx.hstack(
        rx.hstack(
            rx.text("Enable Detection", class_name="text-cyan-100"),
            rx.switch(
                checked=CameraState.detection_enabled,
                on_change=CameraState.toggle_detection,
                color_scheme="grass",
                variant="surface",
                disabled=~media_active,
                transition="all 0.2s ease-in-out",
            ),
            spacing="2",
            align="center",
        ),
        model_navigation(),
        spacing="2",
        class_name="flex justify-between rounded-xl border border-cyan-400/20 bg-cyan-950/30 p-3"
    )

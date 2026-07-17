import reflex as rx
from object_cheating.states.camera_state import CameraState


def stats_panel() -> rx.Component:
    model1_classes = ["All", "Bend Over The Desk", "Hand Under Table", "Look Around", "Normal", "Stand Up", "Wave"]
    model2_classes = ["All", "cheating", "normal"]
    model3_classes = ["All", "center", "left", "right"]
    model4_classes = ["All", "sitting", "writing", "raising_hand", "standing", "turned_around", "lie_on_the_desk"]
    model5_classes = ["All", "neutral", "happy", "sad", "surprise", "anger"]
    model6_classes = ["All", "hand_raising", "reading", "writing", "using_phone", "bowing_head", "leaning_over_table"]
    model7_classes = ["All", "hand_raising", "reading", "writing", "using_phone", "bowing_head", "leaning_over_table"]

    target_classes = rx.cond(
        CameraState.active_model == 1,
        model1_classes,
        rx.cond(
            CameraState.active_model == 2,
            model2_classes,
            rx.cond(
                CameraState.active_model == 3,
                model3_classes,
                rx.cond(
                    CameraState.active_model == 4,
                    model4_classes,
                    rx.cond(
                        CameraState.active_model == 5,
                        model5_classes,
                        rx.cond(
                            CameraState.active_model == 6,
                            model6_classes,
                            model7_classes,
                        ),
                    ),
                ),
            ),
        ),
    )

    def target_checkbox(target: str) -> rx.Component:
        return rx.hstack(
            rx.checkbox(
                checked=CameraState.selected_targets.contains(target),
                on_change=lambda checked: CameraState.toggle_selected_target(target, checked),
                color_scheme="grass",
                size="2",
            ),
            rx.text(target, class_name="text-sm font-medium text-cyan-100"),
            spacing="1",
            align="center",
            class_name="mr-2 mb-1",
        )

    def cross_target_checkbox(model_num: int, targets_state, target: str) -> rx.Component:
        return rx.hstack(
            rx.checkbox(
                checked=targets_state.contains(target),
                on_change=lambda checked: CameraState.toggle_cross_model_target(model_num, target, checked),
                color_scheme="grass",
                size="2",
            ),
            rx.text(target, class_name="text-sm font-medium text-cyan-100"),
            spacing="1",
            align="center",
            class_name="mr-2 mb-1",
        )

    def cross_model_group(title: str, model_num: int, classes: list[str], targets_state) -> rx.Component:
        return rx.box(
            rx.text(title, class_name="text-sm font-semibold text-cyan-50"),
            rx.flex(*[cross_target_checkbox(model_num, targets_state, target) for target in classes], wrap="wrap", gap="1", margin_top="2"),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full border border-cyan-400/25",
        )

    single_model_selector = rx.vstack(
        rx.text("Target Selection:", class_name="text-cyan-100"),
        rx.flex(rx.foreach(target_classes, target_checkbox), wrap="wrap", gap="2", max_width="320px"),
        rx.text(f"Selected: {CameraState.selected_target}", class_name="text-xs text-cyan-300/75"),
        spacing="2",
        align="start",
    )

    cross_model_selector = rx.vstack(
        rx.hstack(
            rx.text("Cross Model Detection", class_name="text-cyan-100 font-medium"),
            rx.switch(
                checked=CameraState.cross_model_enabled,
                on_change=CameraState.toggle_cross_model,
                color_scheme="grass",
            ),
            spacing="2",
            align="center",
        ),
        rx.cond(
            CameraState.cross_model_enabled,
            rx.vstack(
                cross_model_group("Model 1", 1, model1_classes, CameraState.model1_cross_targets),
                cross_model_group("Model 2", 2, model2_classes, CameraState.model2_cross_targets),
                cross_model_group("Model 3", 3, model3_classes, CameraState.model3_cross_targets),
                cross_model_group("Model 4", 4, model4_classes, CameraState.model4_cross_targets),
                cross_model_group("Model 5", 5, model5_classes, CameraState.model5_cross_targets),
                cross_model_group("Model 6", 6, model6_classes, CameraState.model6_cross_targets),
                cross_model_group("Model 7", 7, model7_classes, CameraState.model7_cross_targets),
                spacing="3",
                width="100%",
            ),
            single_model_selector,
        ),
        spacing="3",
        align="start",
        width="100%",
    )

    return rx.vstack(
        rx.el.h3("Detection Summary", class_name="text-lg font-semibold mb-2 text-cyan-50"),
        rx.hstack(
            rx.text(f"Total Target: {CameraState.detection_count}", class_name="text-cyan-100"),
            rx.text(
                rx.cond((CameraState.camera_active | CameraState.video_playing), f"FPS: {CameraState.fps}", "FPS: N/A"),
                class_name="text-cyan-100"
            ),
            justify="between",
            width="100%",
        ),
        rx.hstack(
            rx.text(f"Runtime: {CameraState.processing_time}s", class_name="text-cyan-100"),
            cross_model_selector,
            justify="between",
            width="100%",
        ),
        spacing="4",
        class_name="border border-cyan-400/30 bg-[#061b3d]/80 p-4 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] w-full backdrop-blur"
    )

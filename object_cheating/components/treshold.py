import reflex as rx
from object_cheating.states.threshold_state import ThresholdState
from object_cheating.states.camera_state import CameraState

def threshold() -> rx.Component:
    def threshold_input(value, on_change, step=0.01, min_value=0, max_value=1):
        return rx.input(
            value=value,
            type="number",
            min=min_value,
            max=max_value,
            step=step,
            width="72px",
            height="32px",
            text_align="center",
            color="#dffbff",
            background_color="#03142f",
            border="1px solid rgba(34,211,238,0.45)",
            border_radius="md",
            on_change=on_change,
        )

    def threshold_row(label: str, value, on_change, step=0.01, min_value=0, max_value=1):
        return rx.hstack(
            rx.text(label, class_name="text-sm font-medium text-cyan-100"),
            rx.spacer(),
            threshold_input(value, on_change, step, min_value, max_value),
            width="100%",
            align="center",
        )

    def cross_model_thresholds() -> rx.Component:
        model1_card = rx.box(
            rx.text("Model 1", class_name="text-sm font-semibold text-cyan-50"),
            threshold_row(
                "Confidence",
                CameraState.model1_confidence_threshold,
                lambda value: CameraState.set_model_confidence_from_str(value, 1),
            ),
            threshold_row(
                "IoU",
                CameraState.model1_iou_threshold,
                lambda value: CameraState.set_model_second_threshold_from_str(value, 1),
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full space-y-2 border border-cyan-400/25",
        )
        model2_card = rx.box(
            rx.text("Model 2", class_name="text-sm font-semibold text-cyan-50"),
            threshold_row(
                "Confidence",
                CameraState.model2_confidence_threshold,
                lambda value: CameraState.set_model_confidence_from_str(value, 2),
            ),
            threshold_row(
                "IoU",
                CameraState.model2_iou_threshold,
                lambda value: CameraState.set_model_second_threshold_from_str(value, 2),
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full space-y-2 border border-cyan-400/25",
        )
        model3_card = rx.box(
            rx.text("Model 3", class_name="text-sm font-semibold text-cyan-50"),
            threshold_row(
                "CNN Confidence",
                CameraState.model3_confidence_threshold,
                lambda value: CameraState.set_model_confidence_from_str(value, 3),
            ),
            threshold_row(
                "Duration (s)",
                CameraState.model3_duration_threshold,
                lambda value: CameraState.set_model_second_threshold_from_str(value, 3),
                step=0.1,
                min_value=1,
                max_value=10,
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full space-y-2 border border-cyan-400/25",
        )
        model4_card = rx.box(
            rx.text("Model 4", class_name="text-sm font-semibold text-cyan-50"),
            threshold_row(
                "Detection Conf",
                CameraState.model4_confidence_threshold,
                lambda value: CameraState.set_model_confidence_from_str(value, 4),
            ),
            threshold_row(
                "IoU",
                CameraState.model4_iou_threshold,
                lambda value: CameraState.set_model_second_threshold_from_str(value, 4),
            ),
            threshold_row(
                "Action Conf",
                CameraState.model4_action_confidence_threshold,
                CameraState.set_model4_action_confidence_from_str,
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full space-y-2 border border-cyan-400/25",
        )
        model5_card = rx.box(
            rx.text("Model 5", class_name="text-sm font-semibold text-cyan-50"),
            threshold_row(
                "Face Conf",
                CameraState.model5_face_confidence_threshold,
                lambda value: CameraState.set_model_confidence_from_str(value, 5),
            ),
            threshold_row(
                "Emotion Conf",
                CameraState.model5_emotion_confidence_threshold,
                lambda value: CameraState.set_model_second_threshold_from_str(value, 5),
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl w-full space-y-2 border border-cyan-400/25",
        )

        return rx.vstack(
            rx.hstack(
                rx.icon_button(
                    rx.icon("chevron-left"),
                    on_click=CameraState.prev_cross_threshold_model,
                    disabled=CameraState.cross_threshold_model == 1,
                    variant="surface",
                    height="30px",
                    width="30px",
                ),
                rx.badge(
                    rx.center(
                        rx.text(f"Model {CameraState.cross_threshold_model}"),
                        width="100%",
                    ),
                    variant="surface",
                    min_width="100px",
                    text_align="center",
                ),
                rx.icon_button(
                    rx.icon("chevron-right"),
                    on_click=CameraState.next_cross_threshold_model,
                    disabled=CameraState.cross_threshold_model == 5,
                    variant="surface",
                    height="30px",
                    width="30px",
                ),
                justify="between",
                align="center",
                width="100%",
            ),
            rx.match(
                CameraState.cross_threshold_model,
                (1, model1_card),
                (2, model2_card),
                (3, model3_card),
                (4, model4_card),
                (5, model5_card),
                model1_card,
            ),
            spacing="3",
            width="100%",
        )

    return rx.box(
        rx.vstack(
            rx.cond(
                CameraState.cross_model_enabled,
                rx.el.h3("模型阈值", class_name="text-lg font-semibold mb-2 text-cyan-50"),
                rx.el.h3("阈值设置", class_name="text-lg font-semibold mb-2 text-cyan-50"),
            ),
            rx.cond(
                CameraState.cross_model_enabled,
                cross_model_thresholds(),
                rx.fragment(
                    # Confidence Threshold Section
                    rx.hstack(
                        rx.text(
                            rx.cond(
                                CameraState.active_model == 5,
                                "Face Confidence:",
                                "Confidence Threshold:"
                            ),
                            class_name="font-medium text-cyan-100"
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.input(
                                value=ThresholdState.confidence_threshold,
                                type="number",
                                min=0,
                                max=1,
                                step=0.01,
                                width="70px",
                                height="36px",
                                text_align="center",
                                color="#dffbff",
                                background_color="#03142f",
                                border="1px solid rgba(34,211,238,0.45)",
                                border_radius="md",
                                on_change=ThresholdState.set_confidence_from_str,
                            ),
                            rx.vstack(
                                rx.icon_button(
                                    rx.icon("chevron-up", size=15),
                                    on_click=ThresholdState.increment_confidence,
                                    border="1px solid rgba(34,211,238,0.35)",
                                    border_radius="md",
                                    height="18px",
                                    width="30px",
                                    px="1",
                                    ml="1",
                                ),
                                rx.icon_button(
                                    rx.icon("chevron-down", size=15),
                                    on_click=ThresholdState.decrement_confidence,
                                    border="1px solid rgba(34,211,238,0.35)",
                                    border_radius="md",
                                    height="18px",
                                    width="30px",
                                    px="1",
                                    ml="1",
                                ),
                                spacing="0",
                            ),
                            align="center",
                        ),
                        width="100%",
                        justify="between",
                        align="center",
                    ),
                    # Second Threshold Section (IoU for Model 1 & 2, Duration for Model 3)
                    rx.hstack(
                        rx.text(
                            rx.cond(
                                CameraState.active_model == 3,
                                "Duration Threshold (s):",
                                rx.cond(
                                    CameraState.active_model == 5,
                                    "Emotion Confidence:",
                                    "IoU Threshold:"
                                )
                            ),
                            class_name="font-medium text-cyan-100"
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.input(
                                value=rx.cond(
                                    CameraState.active_model == 3,
                                    ThresholdState.duration_threshold,
                                    ThresholdState.iou_threshold
                                ),
                                type="number",
                                min=rx.cond(CameraState.active_model == 3, 1, 0),
                                max=rx.cond(CameraState.active_model == 3, 10, 1),
                                step=rx.cond(CameraState.active_model == 3, 0.1, 0.01),
                                width="70px",
                                height="36px",
                                text_align="center",
                                color="#dffbff",
                                background_color="#03142f",
                                border="1px solid rgba(34,211,238,0.45)",
                                border_radius="md",
                                on_change=lambda value: ThresholdState.set_second_threshold_from_str(value, CameraState.active_model),
                            ),
                            rx.vstack(
                                rx.icon_button(
                                    rx.icon("chevron-up", size=15),
                                    on_click=lambda: ThresholdState.increment_second_threshold(CameraState.active_model),
                                    border="1px solid rgba(34,211,238,0.35)",
                                    border_radius="md",
                                    height="18px",
                                    width="30px",
                                    px="1",
                                    ml="1",
                                ),
                                rx.icon_button(
                                    rx.icon("chevron-down", size=15),
                                    on_click=lambda: ThresholdState.decrement_second_threshold(CameraState.active_model),
                                    border="1px solid rgba(34,211,238,0.35)",
                                    border_radius="md",
                                    height="18px",
                                    width="30px",
                                    px="1",
                                    ml="1",
                                ),
                                spacing="0",
                            ),
                            align="center",
                        ),
                        width="100%",
                        justify="between",
                        align="center",
                        mt="4",
                    ),
                ),
            ),
            class_name="border border-cyan-400/30 bg-[#061b3d]/80 p-4 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] w-full backdrop-blur"
        ),
        width="100%",
    )

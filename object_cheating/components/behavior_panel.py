import reflex as rx
from object_cheating.states.camera_state import CameraState


def behavior_panel() -> rx.Component:
    """Behavior panel component."""
    color_map_model1 = {
        "Normal": "text-[#7CFC00]",
        "Bend Over The Desk": "text-[#00FFFF]",
        "Hand Under Table": "text-[#4169E1]",
        "Look Around": "text-[#EE82EE]",
        "Stand Up": "text-[#B0C4DE]",
        "Wave": "text-[#FFB6C1]",
    }

    color_map_model3 = {
        "center": "text-[#7CFC00]",
        "left": "text-[#7B68EE]",
        "right": "text-[#7B68EE]",
        "closed": "text-[#808080]",
    }

    color_map_model4 = {
        "sitting": "text-[#7CFC00]",
        "writing": "text-[#2563EB]",
        "raising_hand": "text-[#FF6347]",
        "standing": "text-[#EAB308]",
        "turned_around": "text-[#8B5CF6]",
        "lie_on_the_desk": "text-[#DC143C]",
    }

    color_map_model5 = {
        "neutral": "text-[#7CFC00]",
        "happy": "text-[#EAB308]",
        "sad": "text-[#2563EB]",
        "surprise": "text-[#00CED1]",
        "anger": "text-[#FF6347]",
    }

    color_map_model6 = {
        "hand_raising": "text-[#FF6347]",
        "reading": "text-[#7CFC00]",
        "writing": "text-[#2563EB]",
        "using_phone": "text-[#EAB308]",
        "bowing_head": "text-[#8B5CF6]",
        "leaning_over_table": "text-[#00CED1]",
    }

    color_map_model8 = {
        "leaning_over_table": "text-[#00CED1]",
        "Hand Under Table": "text-[#4169E1]",
        "Look Around": "text-[#EE82EE]",
        "Normal": "text-[#7CFC00]",
        "standing": "text-[#EAB308]",
        "Wave": "text-[#FFB6C1]",
        "sitting": "text-[#7CFC00]",
        "writing": "text-[#2563EB]",
        "hand_raising": "text-[#FF6347]",
        "turned_around": "text-[#8B5CF6]",
        "lie_on_the_desk": "text-[#DC143C]",
        "reading": "text-[#7CFC00]",
        "using_phone": "text-[#EAB308]",
        "bowing_head": "text-[#8B5CF6]",
    }

    default_color = "text-cyan-200"

    def get_behavior_color() -> str:
        return rx.cond(
            CameraState.detection_enabled & (CameraState.detection_count > 0),
            rx.cond(
                CameraState.active_model == 1,
                rx.match(
                    CameraState.highest_confidence_class,
                    ("Normal", color_map_model1["Normal"]),
                    ("Bend Over The Desk", color_map_model1["Bend Over The Desk"]),
                    ("Hand Under Table", color_map_model1["Hand Under Table"]),
                    ("Look Around", color_map_model1["Look Around"]),
                    ("Stand Up", color_map_model1["Stand Up"]),
                    ("Wave", color_map_model1["Wave"]),
                    default_color,
                ),
                rx.cond(
                    CameraState.active_model == 2,
                    rx.cond(CameraState.highest_confidence_class == "cheating", "text-[#FF6347]", "text-[#7CFC00]"),
                    rx.cond(
                        CameraState.active_model == 3,
                        rx.match(
                            CameraState.highest_confidence_class,
                            ("center", color_map_model3["center"]),
                            ("left", color_map_model3["left"]),
                            ("right", color_map_model3["right"]),
                            ("closed", color_map_model3["closed"]),
                            default_color,
                        ),
                        rx.cond(
                            CameraState.active_model == 4,
                            rx.match(
                                CameraState.highest_confidence_class,
                                ("sitting", color_map_model4["sitting"]),
                                ("writing", color_map_model4["writing"]),
                                ("raising_hand", color_map_model4["raising_hand"]),
                                ("standing", color_map_model4["standing"]),
                                ("turned_around", color_map_model4["turned_around"]),
                                ("lie_on_the_desk", color_map_model4["lie_on_the_desk"]),
                                default_color,
                            ),
                            rx.cond(
                                CameraState.active_model == 5,
                                rx.match(
                                    CameraState.highest_confidence_class,
                                    ("neutral", color_map_model5["neutral"]),
                                    ("happy", color_map_model5["happy"]),
                                    ("sad", color_map_model5["sad"]),
                                    ("surprise", color_map_model5["surprise"]),
                                    ("anger", color_map_model5["anger"]),
                                    default_color,
                                ),
                                rx.cond(
                                    CameraState.active_model == 8,
                                    rx.match(
                                        CameraState.highest_confidence_class,
                                        ("leaning_over_table", color_map_model8["leaning_over_table"]),
                                        ("Hand Under Table", color_map_model8["Hand Under Table"]),
                                        ("Look Around", color_map_model8["Look Around"]),
                                        ("Normal", color_map_model8["Normal"]),
                                        ("standing", color_map_model8["standing"]),
                                        ("Wave", color_map_model8["Wave"]),
                                        ("sitting", color_map_model8["sitting"]),
                                        ("writing", color_map_model8["writing"]),
                                        ("hand_raising", color_map_model8["hand_raising"]),
                                        ("turned_around", color_map_model8["turned_around"]),
                                        ("lie_on_the_desk", color_map_model8["lie_on_the_desk"]),
                                        ("reading", color_map_model8["reading"]),
                                        ("using_phone", color_map_model8["using_phone"]),
                                        ("bowing_head", color_map_model8["bowing_head"]),
                                        default_color,
                                    ),
                                    rx.match(
                                        CameraState.highest_confidence_class,
                                        ("hand_raising", color_map_model6["hand_raising"]),
                                        ("reading", color_map_model6["reading"]),
                                        ("writing", color_map_model6["writing"]),
                                        ("using_phone", color_map_model6["using_phone"]),
                                        ("bowing_head", color_map_model6["bowing_head"]),
                                        ("leaning_over_table", color_map_model6["leaning_over_table"]),
                                        default_color,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            default_color,
        ) + " font-semibold"

    def get_confidence_color() -> str:
        return rx.cond(
            CameraState.highest_confidence >= 90,
            "text-[#00FF00]",
            rx.cond(
                CameraState.highest_confidence >= 80,
                "text-[#ADFF2F]",
                rx.cond(
                    CameraState.highest_confidence >= 70,
                    "text-[#9ACD32]",
                    rx.cond(
                        CameraState.highest_confidence >= 60,
                        "text-[#EBC40E]",
                        rx.cond(
                            CameraState.highest_confidence >= 50,
                            "text-[#FFD700]",
                            rx.cond(
                                CameraState.highest_confidence >= 40,
                                "text-[#FFA500]",
                                rx.cond(
                                    CameraState.highest_confidence >= 30,
                                    "text-[#FF8C00]",
                                    rx.cond(
                                        CameraState.highest_confidence >= 20,
                                        "text-[#FF7F50]",
                                        "text-[#DC143C]",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ) + " font-semibold"

    return rx.el.div(
        rx.el.h3("行为分析", class_name="text-lg font-semibold mb-2 text-cyan-50"),
        rx.el.div(
            rx.el.div(
                rx.el.span("行为：", class_name="text-cyan-100"),
                rx.el.span(
                    rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), CameraState.highest_confidence_class, "N/A"),
                    class_name=get_behavior_color(),
                ),
                class_name="flex justify-between",
            ),
            rx.el.div(
                rx.el.span("置信度：", class_name="text-cyan-100"),
                rx.el.span(
                    rx.cond(CameraState.detection_enabled & (CameraState.detection_count > 0), f"{CameraState.highest_confidence}%", "0%"),
                    class_name=rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        get_confidence_color(),
                        "text-[#DC143C]",
                    ),
                ),
                class_name="flex justify-between mt-2",
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl border border-cyan-400/25",
        ),
        class_name="border border-cyan-400/30 bg-[#061b3d]/80 p-4 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] w-full backdrop-blur",
    )

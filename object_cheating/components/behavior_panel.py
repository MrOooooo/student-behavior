import reflex as rx
from object_cheating.states.camera_state import CameraState

def behavior_panel() -> rx.Component:
    """Behavior panel component."""
    # Define color maps for Model 1
    color_map_model1 = {
        "Normal": "text-[#7CFC00]",        # Lawn Green
        "Bend Over The Desk": "text-[#00FFFF]",    # Aqua
        "Hand Under Table": "text-[#4169E1]",      # Royal Blue
        "Look Around": "text-[#EE82EE]",         # Violet
        "Stand Up": "text-[#B0C4DE]",           # Light Steel Blue
        "Wave": "text-[#FFB6C1]"                # Light Pink
    }
    
    # Define color maps for Model 3 (Eye Tracking)
    color_map_model3 = {
        "center": "text-[#7CFC00]",  # Lawn Green 
        "left": "text-[#7B68EE]",    # Medium Slate Blue  
        "right": "text-[#7B68EE]",   # Medium Slate Blue 
        "closed": "text-[#808080]"   # Gray
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

    # Default color for N/A or unknown class
    default_color = "text-cyan-200"

    # Function to dynamically get the color class for the behavior text
    def get_behavior_color() -> str:
        return rx.cond(
            CameraState.detection_enabled & (CameraState.detection_count > 0),
            rx.cond(
                CameraState.active_model == 1,
                # Check if the highest_confidence_class exists in color_map_model1
                rx.match(
                    CameraState.highest_confidence_class,
                    ("Normal", color_map_model1["Normal"]),
                    ("Bend Over The Desk", color_map_model1["Bend Over The Desk"]),
                    ("Hand Under Table", color_map_model1["Hand Under Table"]),
                    ("Look Around", color_map_model1["Look Around"]),
                    ("Stand Up", color_map_model1["Stand Up"]),
                    ("Wave", color_map_model1["Wave"]),
                    default_color  # Default color if class not in color_map_model1
                ),
                rx.cond(
                    CameraState.active_model == 2,
                    rx.cond(
                        CameraState.highest_confidence_class == "cheating",
                        "text-[#FF6347]",  # Tomato for cheating
                        "text-[#7CFC00]"   # Lawn green for normal
                    ),
                    rx.cond(
                        CameraState.active_model == 3,
                        # Check if the highest_confidence_class exists in color_map_model3
                        rx.match(
                            CameraState.highest_confidence_class,
                            ("center", color_map_model3["center"]),
                            ("left", color_map_model3["left"]),
                            ("right", color_map_model3["right"]),
                            ("closed", color_map_model3["closed"]),
                            default_color  
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
                                default_color
                            ),
                            rx.match(
                                CameraState.highest_confidence_class,
                                ("neutral", color_map_model5["neutral"]),
                                ("happy", color_map_model5["happy"]),
                                ("sad", color_map_model5["sad"]),
                                ("surprise", color_map_model5["surprise"]),
                                ("anger", color_map_model5["anger"]),
                                default_color
                            )
                        )
                    )
                )
            ),
            default_color
        ) + " font-semibold"

    # Function to dynamically get the color class for the confidence level text
    def get_confidence_color() -> str:
        return rx.cond(
            CameraState.highest_confidence >= 90,
            "text-[#00FF00]",  # lime
            rx.cond(
                CameraState.highest_confidence >= 80,
                "text-[#ADFF2F]",  # green yellow
                rx.cond(
                    CameraState.highest_confidence >= 70,
                    "text-[#9ACD32]",  # yellow green
                    rx.cond(
                        CameraState.highest_confidence >= 60,
                        "text-[#EBC40E]",  # yellow
                        rx.cond(
                            CameraState.highest_confidence >= 50,
                            "text-[#FFD700]",  # gold
                            rx.cond(
                                CameraState.highest_confidence >= 40,
                                "text-[#FFA500]",  # orange
                                rx.cond(
                                    CameraState.highest_confidence >= 30,
                                    "text-[#FF8C00]",  # dark orange
                                    rx.cond(
                                        CameraState.highest_confidence >= 20,
                                        "text-[#FF7F50]",  # coral
                                        "text-[#DC143C]"  # crimson
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ) + " font-semibold"

    return rx.el.div(
        rx.el.h3("行为分析", class_name="text-lg font-semibold mb-2 text-cyan-50"),
        rx.el.div(
            rx.el.div(
                rx.el.span("行为: ", class_name="text-cyan-100"),
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        CameraState.highest_confidence_class,
                        "N/A"
                    ),
                    class_name=get_behavior_color()
                ),
                class_name="flex justify-between"
            ),
            rx.el.div(
                rx.el.span("置信度: ", class_name="text-cyan-100"),
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        f"{CameraState.highest_confidence}%",
                        "0%"
                    ),
                    class_name=rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        get_confidence_color(),
                        "text-[#DC143C]"  # Crimson for 0% or no detection
                    )
                ),
                class_name="flex justify-between mt-2"
            ),
            class_name="bg-[#08264f]/80 p-3 rounded-xl border border-cyan-400/25"
        ),
        class_name="border border-cyan-400/30 bg-[#061b3d]/80 p-4 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] w-full backdrop-blur"
    )

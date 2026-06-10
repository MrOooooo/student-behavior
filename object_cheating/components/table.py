import reflex as rx
from typing import List, Dict

def create_data_row(data: Dict[str, str]):
    from object_cheating.states.camera_state import CameraState
    return rx.table.row(
        rx.table.cell(
            rx.text(
                data["no"],
                color="#dffbff",
                font_size="11px",
                weight="medium",
            ),
        ),
        rx.table.cell(
            rx.text(
                data["location_file"],
                color="#dffbff",
                font_size="11px",
                weight="medium",
            ),
        ),
        rx.table.cell(
            rx.badge(
                data["behaviour"],
                color_scheme=rx.match(
                    data["behaviour"],
                    ("cheating", "tomato"),
                    ("left", "orange"),
                    ("right", "orange"),
                    ("Look Around", "violet"),
                    ("Normal", "grass"),
                    ("normal", "grass"),
                    ("center", "green"),
                    ("Bend Over The Desk", "cyan"),
                    ("Hand Under Table", "indigo"),
                    ("Stand Up", "sky"),
                    ("Wave", "pink"),
                    ("sitting", "grass"),
                    ("writing", "blue"),
                    ("raising_hand", "tomato"),
                    ("standing", "amber"),
                    ("turned_around", "violet"),
                    ("lie_on_the_desk", "crimson"),
                    ("neutral", "grass"),
                    ("happy", "amber"),
                    ("sad", "blue"),
                    ("surprise", "yellow"),
                    ("anger", "tomato"),
                    "gray"
                ),
                size="1"
            ),
        ),
        rx.table.cell(
            rx.text(
                data["coordinate"],
                color="#dffbff",
                font_size="11px",
                weight="regular",
            ),
        ),
        align="center",
        white_space="nowrap",
    )

def tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.vstack(
        rx.scroll_area(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.foreach(
                            ["No", "Location File", "Behaviour", "Coordinate"],
                            lambda title: rx.table.column_header_cell(
                                rx.text(title, font_size="12px", weight="bold", color="#dffbff"),
                            ),
                        ),
                    ),
                    position="sticky",
                    top="0",
                    background_color="#08264f",
                    z_index="1",
                ),
                rx.table.body(
                    rx.foreach(
                        CameraState.table_data,
                        create_data_row
                    ),
                ),
                width="100%",
                variant="surface",
                size="2",
            ),
            type="always",
            scrollbars="vertical",
            style={
                "height": "267px",  # Tinggi untuk 5 baris
                "border": "1px solid rgba(34,211,238,0.35)",
                "border_radius": "12px",
                "background": "rgba(8,38,79,0.76)",
            },
        ),
        background="rgba(6,27,61,0.78)",
        padding="4",
        border_radius="16px",
        border="1px solid rgba(34,211,238,0.30)",
        box_shadow="0 0 24px rgba(34,211,238,0.14)",
        width="100%",
        max_width="100%",
    )
def _tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    ["No", "Location File", "Behaviour", "Coordinate"],
                    lambda title: rx.table.column_header_cell(
                            rx.text(title, font_size="12px", weight="bold", color="#dffbff"),
                    ),
                ),
            ),
        ),
        rx.table.body(
            rx.foreach(CameraState.table_data, create_data_row),
            style={"max_height": "200px", "overflow_y": "auto"},
        ),
        width="100%",
        variant="surface",
        size="2",
    )

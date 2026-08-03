import reflex as rx
from typing import Dict


def create_data_row(data: Dict[str, str]):
    cell_text_style = {
        "color": "#061a2e",
        "font_size": "12px",
        "weight": "bold",
    }
    return rx.table.row(
        rx.table.cell(rx.text(data["no"], **cell_text_style)),
        rx.table.cell(rx.text(data["person_id"], **cell_text_style)),
        rx.table.cell(rx.text(data["location_file"], **cell_text_style)),
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
                    ("hand_raising", "tomato"),
                    ("reading", "grass"),
                    ("using_phone", "amber"),
                    ("bowing_head", "violet"),
                    ("leaning_over_table", "cyan"),
                    "gray",
                ),
                size="1",
                variant="solid",
                high_contrast=True,
                style={"font_weight": "800", "letter_spacing": "0.02em"},
            )
        ),
        rx.table.cell(rx.text(data["coordinate"], **cell_text_style)),
        align="center",
        white_space="nowrap",
        background="linear-gradient(90deg, rgba(240,249,255,0.98), rgba(207,250,254,0.94))",
        border_bottom="1px solid rgba(8,47,73,0.18)",
        # _hover={"background": "#ffffff"},
    )


def tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.vstack(
        rx.scroll_area(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.foreach(
                            ["No", "Person", "Location File", "Behaviour", "Coordinate"],
                            lambda title: rx.table.column_header_cell(
                                rx.text(title, font_size="12px", weight="bold", color="#ffffff"),
                                color="#ffffff",
                                background="#031a32",
                            ),
                        ),
                    ),
                    position="sticky",
                    top="0",
                    background_color="#06244D",
                    z_index="1",
                ),
                rx.table.body(rx.foreach(CameraState.table_data, create_data_row)),
                width="100%",
                variant="surface",
                size="2",
            ),
            type="always",
            scrollbars="vertical",
            style={
                "height": "267px",
                "border": "1px solid rgba(34,211,238,0.65)",
                "border_radius": "12px",
                # "background": "rgba(219, 245, 255, 0.96)",
                "box_shadow": "inset 0 0 18px rgba(3, 26, 50, 0.18)",
            },
        ),
        background="linear-gradient(180deg, rgba(8,47,73,0.96), rgba(3,26,50,0.98))",
        padding="4",
        border_radius="16px",
        border="1px solid rgba(34,211,238,0.55)",
        box_shadow="0 0 28px rgba(34,211,238,0.25)",
        width="100%",
        max_width="100%",
    )


def _tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    ["No", "Person", "Location File", "Behaviour", "Coordinate"],
                    lambda title: rx.table.column_header_cell(rx.text(title, font_size="12px", weight="bold", color="#ffffff")),
                ),
            ),
        ),
        rx.table.body(rx.foreach(CameraState.table_data, create_data_row), style={"max_height": "200px", "overflow_y": "auto"}),
        width="100%",
        variant="surface",
        size="2",
    )

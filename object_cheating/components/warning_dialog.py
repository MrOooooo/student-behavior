import reflex as rx
from object_cheating.states.camera_state import CameraState


def warning_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("检测正在运行", class_name="text-red-600 font-bold"),
            rx.dialog.description(
                rx.text("请先关闭检测，再切换模型。这样可以确保当前模型正确释放，并初始化新的模型。", class_name="text-gray-700 my-4"),
            ),
            rx.hstack(
                rx.dialog.close(
                    rx.button("取消", class_name="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"),
                ),
                spacing="4",
                justify="end",
                padding_top="4",
            ),
        ),
        open=CameraState.show_warning_dialog,
        on_open_change=CameraState.close_warning_dialog,
    )

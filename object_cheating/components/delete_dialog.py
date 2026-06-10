import reflex as rx
from object_cheating.states.camera_state import CameraState

def delete_dialog() -> rx.Component:
    return rx.alert_dialog.root(
        rx.alert_dialog.content(
            rx.alert_dialog.title(
                "清除显示内容？",
                class_name="text-xl font-bold text-gray-900"
            ),
            rx.alert_dialog.description(
                rx.text(
                    "这将删除所有当前的媒体内容，包括图片, "
                    "视频和网络摄像头的画面。此操作无法撤销。",
                    class_name="text-gray-600 mt-2"
                ),
                margin_bottom="20px",
            ),
            rx.hstack(
                rx.alert_dialog.action(
                    rx.button(
                        "清除",
                        color_scheme="red",
                        on_click=CameraState.confirm_clear,
                        class_name="px-4 py-2"
                    ),
                ),
                rx.spacer(),
                rx.alert_dialog.cancel(
                    rx.button(
                        "取消",
                        class_name="px-4 py-2 bg-gray-200 text-gray-700 hover:bg-gray-300"
                    )
                ),
            ),
            class_name="p-6"
        ),
        open=CameraState.show_delete_dialog,
        on_open_change=CameraState.cancel_clear,
    )
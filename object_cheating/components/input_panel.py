import reflex as rx
from object_cheating.states.camera_state import CameraState


def input_panel() -> rx.Component:
    button_style = "border border-cyan-300/50 bg-cyan-400/10 text-cyan-50 px-6 py-2 w-full text-center rounded-lg shadow-[0_0_14px_rgba(34,211,238,0.15)] hover:bg-cyan-300/20"
    disabled_style = "border border-slate-600 bg-slate-800/80 text-slate-500 px-6 py-2 w-full text-center rounded-lg cursor-not-allowed"

    media_active = rx.cond(
        (CameraState.current_frame != "") | CameraState.video_playing,
        True,
        False,
    )

    return rx.el.div(
        rx.el.div(
            rx.el.h3("输入方式选择", class_name="text-lg font-semibold mb-2 text-cyan-50"),
            rx.el.div(
                rx.upload(
                    rx.button(
                        "图片",
                        class_name=rx.cond(
                            CameraState.camera_active | media_active,
                            disabled_style,
                            button_style,
                        ),
                    ),
                    multiple=False,
                    border="none",
                    padding="0",
                    margin="0",
                    id="image_upload",
                    style={"border": "none"},
                    disabled=CameraState.camera_active | media_active,
                ),
                rx.button(
                    "上传",
                    on_click=CameraState.handle_image_upload(rx.upload_files(upload_id="image_upload")),
                    class_name="hidden",
                    id="upload_button",
                ),
                rx.script("""
                    document.addEventListener('change', function(e) {
                        if (e.target && e.target.closest('#image_upload')) {
                            setTimeout(() => {
                                document.getElementById('upload_button').click();
                            }, 100);
                        }
                    });
                """),
                rx.upload(
                    rx.button(
                        "视频",
                        class_name=rx.cond(
                            CameraState.camera_active | media_active,
                            disabled_style,
                            button_style,
                        ),
                    ),
                    multiple=False,
                    border="none",
                    padding="0",
                    margin="0",
                    id="video_upload",
                    style={"border": "none"},
                    disabled=CameraState.camera_active | media_active,
                ),
                rx.button(
                    "上传",
                    on_click=CameraState.handle_video_upload(rx.upload_files(upload_id="video_upload")),
                    class_name="hidden",
                    id="upload_video_button",
                ),
                rx.script("""
                    document.addEventListener('change', function(e) {
                        if (e.target && e.target.closest('#video_upload')) {
                            setTimeout(() => {
                                document.getElementById('upload_video_button').click();
                            }, 100);
                        }
                    });
                """),
                class_name="grid grid-cols-2 gap-4 w-full",
            ),
            rx.el.div(
                rx.button(
                    "摄像头",
                    on_click=CameraState.toggle_camera,
                    class_name=rx.cond(CameraState.camera_active | media_active, disabled_style, button_style),
                    disabled=CameraState.camera_active | media_active,
                ),
                rx.button(
                    "保存当前帧",
                    on_click=CameraState.save_current_frame,
                    class_name=rx.cond(CameraState.current_frame != "", button_style, disabled_style),
                    disabled=CameraState.current_frame == "",
                ),
                class_name="grid grid-cols-2 gap-4 w-full mt-4",
            ),
            rx.el.div(
                rx.button(
                    "清除",
                    on_click=CameraState.try_clear_camera,
                    class_name=rx.cond(CameraState.current_frame != "", button_style, disabled_style),
                    disabled=CameraState.current_frame == "",
                ),
                class_name="flex justify-center w-full mt-4",
            ),
            class_name="w-full p-4 border border-cyan-400/30 bg-[#061b3d]/80 rounded-2xl shadow-[0_0_24px_rgba(34,211,238,0.16)] max-w-md mx-auto backdrop-blur",
        )
    )

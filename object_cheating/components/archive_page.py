import reflex as rx

from object_cheating.states.archive_state import ArchiveState

def _archive_preview_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.el.div(
                rx.el.div(
                    rx.dialog.title(
                        ArchiveState.preview_filename,
                        class_name="max-w-[900px] truncate text-xl font-bold text-cyan-50",
                    ),
                    rx.dialog.close(
                        rx.button(
                            "\u5173\u95ed",
                            on_click=ArchiveState.close_preview,
                            class_name="rounded-lg border border-cyan-300/40 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-50 hover:bg-cyan-300/20",
                        ),
                    ),
                    class_name="mb-4 flex items-center justify-between gap-4",
                ),
                rx.image(
                    src=ArchiveState.preview_image_src,
                    width="100%",
                    max_height="76vh",
                    object_fit="contain",
                    border_radius="16px",
                    background="rgba(2,8,23,0.92)",
                    border="1px solid rgba(34,211,238,0.35)",
                ),
                rx.hstack(
                    rx.badge(ArchiveState.preview_person, class_name="bg-cyan-300/15 text-cyan-50 border border-cyan-300/40"),
                    rx.badge(ArchiveState.preview_model, class_name="bg-blue-300/15 text-blue-50 border border-blue-300/40"),
                    spacing="2",
                    class_name="mt-4 flex-wrap",
                ),
                rx.text(ArchiveState.preview_path, class_name="mt-2 break-all text-xs text-cyan-100/60"),
            ),
            class_name="max-w-[92vw] border border-cyan-400/35 bg-[#031a32]/95 text-cyan-50 shadow-[0_0_38px_rgba(34,211,238,0.22)] backdrop-blur",
        ),
        open=ArchiveState.preview_open,
        on_open_change=ArchiveState.set_preview_open,
    )


def _archive_card(item):
    return rx.el.div(
        rx.image(
            src=item["image_src"],
            width="100%",
            height="220px",
            object_fit="contain",
            border_radius="12px",
            background="rgba(2,8,23,0.72)",
            border="1px solid rgba(34,211,238,0.35)",
            cursor="pointer",
            on_click=lambda: ArchiveState.open_preview(
                item["image_src"],
                item["person"],
                item["model"],
                item["filename"],
                item["path"],
            ),
        ),
        rx.el.div(
            rx.badge(item["group"], class_name="bg-cyan-300/15 text-cyan-50 border border-cyan-300/40"),
            rx.badge(item["model"], class_name="bg-blue-300/15 text-blue-50 border border-blue-300/40"),
            class_name="mt-3 flex flex-wrap gap-2",
        ),
        rx.text(item["person"], class_name="mt-2 text-sm font-bold text-cyan-50"),
        rx.text(item["filename"], class_name="mt-1 truncate text-xs text-cyan-100/75"),
        rx.text("\u70b9\u51fb\u56fe\u7247\u67e5\u770b\u5927\u56fe", class_name="mt-1 text-xs font-semibold text-cyan-300/80"),
        rx.text(item["path"], class_name="mt-1 truncate text-[10px] text-cyan-200/45"),
        class_name="rounded-2xl border border-cyan-400/25 bg-cyan-950/35 p-3 shadow-[0_0_22px_rgba(34,211,238,0.12)] backdrop-blur transition hover:border-cyan-300/60 hover:bg-cyan-900/35",
    )


def _archive_group_button(group):
    return rx.button(
        rx.hstack(
            rx.text(group["label"], class_name="font-bold"),
            rx.badge(group["count"], class_name="bg-white/15 text-current"),
            spacing="2",
            align="center",
        ),
        on_click=lambda: ArchiveState.set_archive_group(group["name"]),
        class_name=rx.cond(
            ArchiveState.archive_group == group["name"],
            "rounded-xl bg-cyan-300 px-4 py-2 text-sm font-bold text-slate-950 shadow-[0_0_18px_rgba(34,211,238,0.35)]",
            "rounded-xl border border-cyan-300/35 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 hover:bg-cyan-300/20",
        ),
    )


def archive_page() -> rx.Component:
    return rx.box(
        _archive_preview_dialog(),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.text("\u68c0\u6d4b\u7ed3\u679c\u5f52\u6863", class_name="text-xs tracking-[0.45em] text-cyan-300/80 uppercase"),
                    rx.heading("\u67e5\u770b\u884c\u4e3a\u7167\u7247", size="7", class_name="mt-2 text-cyan-50 drop-shadow-[0_0_14px_rgba(34,211,238,0.55)]"),
                    # rx.text("\u6309 Person \u67e5\u770b\u65f6\uff0c\u70b9\u51fb Person_001 \u53ef\u4ee5\u53ea\u770b\u8fd9\u4e2a\u5b66\u751f\u7684\u68c0\u6d4b\u7167\u7247\uff1b\u70b9\u51fb\u56fe\u7247\u53ef\u4ee5\u653e\u5927\u67e5\u770b\u5168\u56fe\u3002", class_name="mt-2 text-sm text-cyan-100/75"),
                ),
                rx.link(
                    rx.button("\u8fd4\u56de\u68c0\u6d4b\u9875\u9762", class_name="rounded-lg border border-cyan-300/50 bg-cyan-400/10 px-4 py-2 text-sm font-semibold text-cyan-100 hover:bg-cyan-300/20"),
                    href="/",
                ),
                class_name="mb-6 flex items-center justify-between gap-4 border-b border-cyan-400/30 pb-4",
            ),
            rx.el.div(
                rx.hstack(
                    rx.button(
                        "\u6309 Person \u67e5\u770b",
                        on_click=lambda: ArchiveState.set_archive_view("person"),
                        class_name=rx.cond(
                            ArchiveState.archive_view == "person",
                            "rounded-lg bg-cyan-300 px-4 py-2 text-sm font-bold text-slate-950",
                            "rounded-lg border border-cyan-300/40 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100",
                        ),
                    ),
                    rx.button(
                        "\u6309 Model \u67e5\u770b",
                        on_click=lambda: ArchiveState.set_archive_view("model"),
                        class_name=rx.cond(
                            ArchiveState.archive_view == "model",
                            "rounded-lg bg-cyan-300 px-4 py-2 text-sm font-bold text-slate-950",
                            "rounded-lg border border-cyan-300/40 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100",
                        ),
                    ),
                    rx.select(
                        ArchiveState.archive_dates,
                        value=ArchiveState.archive_date,
                        on_change=ArchiveState.set_archive_date,
                        placeholder="\u9009\u62e9\u65e5\u671f",
                        class_name="min-w-[160px] text-cyan-50",
                    ),
                    rx.button(
                        "\u5237\u65b0",
                        on_click=ArchiveState.refresh_archive,
                        class_name="rounded-lg border border-cyan-300/40 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100",
                    ),
                    spacing="3",
                    align="center",
                    class_name="flex flex-wrap rounded-2xl border border-cyan-400/25 bg-cyan-950/40 p-4",
                ),
                class_name="mb-4",
            ),
            rx.el.div(
                rx.text(
                    rx.cond(ArchiveState.archive_view == "person", "\u9009\u62e9\u5b66\u751f", "\u9009\u62e9\u6a21\u578b"),
                    class_name="mb-3 text-sm font-bold tracking-[0.2em] text-cyan-200/85",
                ),
                rx.cond(
                    ArchiveState.archive_groups.length() > 0,
                    rx.el.div(
                        rx.foreach(ArchiveState.archive_groups, _archive_group_button),
                        class_name="flex max-h-[170px] flex-wrap gap-3 overflow-y-auto rounded-2xl border border-cyan-400/25 bg-cyan-950/30 p-4",
                    ),
                    rx.el.div(
                        rx.text("\u6682\u65e0 Person / Model \u5206\u7c7b", class_name="text-sm text-cyan-100/70"),
                        class_name="rounded-2xl border border-cyan-400/25 bg-cyan-950/30 p-4",
                    ),
                ),
                class_name="mb-5",
            ),
            rx.text(ArchiveState.archive_message, class_name="mb-4 text-sm text-cyan-100/75"),
            rx.cond(
                ArchiveState.archive_items.length() > 0,
                rx.el.div(
                    rx.foreach(ArchiveState.archive_items, _archive_card),
                    class_name="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4",
                ),
                rx.el.div(
                    rx.text("\u6682\u65e0\u53ef\u663e\u793a\u56fe\u7247", class_name="text-lg font-bold text-cyan-50"),
                    rx.text("\u5148\u8fd4\u56de\u68c0\u6d4b\u9875\u9762\u8fd0\u884c\u68c0\u6d4b\uff0c\u4fdd\u5b58\u540e\u518d\u5237\u65b0\u8fd9\u91cc\u3002", class_name="mt-2 text-sm text-cyan-100/70"),
                    class_name="rounded-2xl border border-cyan-400/25 bg-cyan-950/40 p-8 text-center",
                ),
            ),
            class_name="mx-auto max-w-[1600px] px-5 py-6",
        ),
        class_name="min-h-screen bg-[#020817] text-cyan-50",
    )

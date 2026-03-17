"""Fusion-style UI shell for Fizix Mech."""

import dearpygui.dearpygui as dpg

from .config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    COLOR_ACCENT_PRIMARY,
    COLOR_ACCENT_SUCCESS,
    COLOR_BG_PRIMARY,
    COLOR_BG_SECONDARY,
    COLOR_BG_TERTIARY,
    COLOR_BORDER,
    COLOR_HOVER,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)
from .presets import ALL_PRESETS


def build_ui(app):
    _create_themes()

    with dpg.window(
        label="App Header",
        tag="topbar_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=True):
                dpg.add_text("FZXMCH", color=COLOR_ACCENT_PRIMARY)
                dpg.add_text("Fusion-style Workspace", color=COLOR_TEXT_MUTED)
            dpg.add_spacer(width=24)
            with dpg.group(horizontal=True):
                dpg.add_button(label="File", width=54, callback=lambda: app.show_command_palette())
                dpg.add_button(label="Help", width=54, callback=lambda: app.show_help())
            dpg.add_spacer(width=18)
            app.workspace_badge = dpg.add_text("BUILD WORKSPACE", color=COLOR_ACCENT_PRIMARY)
            dpg.add_text("|", color=COLOR_TEXT_MUTED)
            app.mode_hint = dpg.add_text("Author geometry, constraints, and forces.", color=COLOR_TEXT_SECONDARY)

    with dpg.window(
        label="Ribbon",
        tag="ribbon_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        with dpg.group():
            with dpg.group(horizontal=True):
                app.build_mode_btn = dpg.add_button(label="SOLID", width=84, callback=lambda: app.set_workspace_mode("build"))
                dpg.add_button(label="SURFACE", width=92, callback=lambda: app.show_add_menu())
                dpg.add_button(label="TOOLS", width=74, callback=lambda: app.show_command_palette())
                app.sim_mode_btn = dpg.add_button(label="SIM", width=74, callback=lambda: app.set_workspace_mode("sim"))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                with dpg.child_window(border=False, width=200, height=56):
                    dpg.add_text("Create", color=COLOR_TEXT_MUTED)
                    with dpg.group(horizontal=True):
                        app.add_menu_btn = dpg.add_button(label="Shift+A Add", width=104, callback=lambda: app.show_add_menu())
                        dpg.add_button(label="Sketch", width=72, callback=lambda: app.select_tool("surface"))
                with dpg.child_window(border=False, width=210, height=56):
                    dpg.add_text("Modify", color=COLOR_TEXT_MUTED)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Select", width=72, callback=lambda: app.select_tool("select"))
                        dpg.add_button(label="Move", width=72, callback=lambda: app.go_to_origin())
                        dpg.add_button(label="Snap", width=54, callback=lambda: app.on_snap_toggle(not app.snap))
                with dpg.child_window(border=False, width=220, height=56):
                    dpg.add_text("Simulation", color=COLOR_TEXT_MUTED)
                    with dpg.group(horizontal=True):
                        app.play_pause_btn = dpg.add_button(label="Run", width=66, callback=lambda: app.toggle_simulation())
                        dpg.add_button(label="Pause", width=66, callback=lambda: app.set_simulation_running(False))
                        dpg.add_button(label="Restore", width=72, callback=lambda: app.restore_startup_scene())
                with dpg.child_window(border=False, width=220, height=56):
                    dpg.add_text("View", color=COLOR_TEXT_MUTED)
                    with dpg.group(horizontal=True):
                        app.grid_toggle_btn = dpg.add_button(label="Grid On", width=68, callback=lambda: app.toggle_grid())
                        app.snap_toggle_btn = dpg.add_button(label="Snap On", width=72, callback=lambda: app.on_snap_toggle(not app.snap))
                        dpg.add_button(label="Origin", width=64, callback=lambda: app.go_to_origin())

    with dpg.window(
        label="Browser",
        tag="browser_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        dpg.add_text("BROWSER", color=COLOR_TEXT_MUTED)
        dpg.add_separator()
        with dpg.collapsing_header(label="Document", default_open=True):
            dpg.add_text("Functional_v1", color=COLOR_TEXT_PRIMARY)
            dpg.add_text("Origin, bodies, surfaces, forces", color=COLOR_TEXT_MUTED, wrap=240)
        with dpg.collapsing_header(label="Scene Graph", default_open=True):
            with dpg.child_window(tag="hier_panel", width=-1, height=340, border=True):
                dpg.add_group(tag="hier_tree")
        with dpg.collapsing_header(label="Presets", default_open=True):
            dpg.add_text("Quick rigs", color=COLOR_TEXT_SECONDARY)

            def _on_preset_select(sender, app_data):
                from .presets import apply_preset, get_preset_by_name

                preset = get_preset_by_name(app_data)
                if preset:
                    apply_preset(app, preset)

            dpg.add_combo(
                items=[p.name for p in ALL_PRESETS],
                width=-1,
                callback=_on_preset_select,
                tag="preset_combo",
            )
            dpg.add_text("Ball Drop, Ramp, Domino, Car, Rocket, more.", color=COLOR_TEXT_MUTED, wrap=240)

    with dpg.window(
        label="Viewport",
        tag="viewport_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("DESIGN", color=COLOR_TEXT_PRIMARY)
                dpg.add_spacer(width=10)
                app.viewport_subtitle = dpg.add_text("Perspective / world coordinates", color=COLOR_TEXT_MUTED)
            dpg.add_separator()
            with dpg.child_window(tag="viewport_canvas_host", autosize_x=True, autosize_y=True, border=False):
                dpg.add_drawlist(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, tag="main_canvas")

    with dpg.window(
        label="Inspector",
        tag="inspector_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        dpg.add_text("INSPECTOR", color=COLOR_TEXT_MUTED)
        dpg.add_separator()
        with dpg.collapsing_header(label="Workspace", default_open=True):
            app.run_status = dpg.add_text("Status: Paused", color=COLOR_TEXT_SECONDARY)
            app.stats = dpg.add_text("Stats...", color=COLOR_TEXT_SECONDARY, wrap=280)
            dpg.add_spacer(height=4)
            dpg.add_text("Time Scale", color=COLOR_TEXT_MUTED)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    default_value=1.0,
                    min_value=0.1,
                    max_value=3.0,
                    width=170,
                    format="%.1fx",
                    clamped=True,
                    tag="time_scale_slider",
                    callback=lambda s, a: setattr(app.physics, "time_scale", float(a)),
                )
                dpg.add_text("1.0x", tag="time_scale_value", color=COLOR_TEXT_SECONDARY)
            dpg.add_text("Gravity", color=COLOR_TEXT_MUTED)
            dpg.add_slider_float(
                default_value=-9.81,
                min_value=-20.0,
                max_value=20.0,
                width=260,
                format="%.2f",
                clamped=True,
                tag="gravity_slider",
                callback=lambda s, a: app.on_gravity_change(a),
            )
        with dpg.collapsing_header(label="Tool Setup", default_open=True):
            with dpg.child_window(tag="tool_panel", width=-1, height=260, border=True):
                dpg.add_text("Pick a tool.", color=COLOR_TEXT_SECONDARY)
        with dpg.collapsing_header(label="Selection", default_open=True):
            with dpg.child_window(tag="prop_panel", width=-1, height=300, border=True):
                dpg.add_text("Nothing selected.", color=COLOR_TEXT_SECONDARY)
        with dpg.collapsing_header(label="Forces", default_open=False):
            with dpg.group(tag="browser_forces_panel"):
                pass
            build_forces_panel(app, "browser_forces_panel")
        with dpg.collapsing_header(label="Precision", default_open=False):
            with dpg.table(
                header_row=False,
                resizable=False,
                policy=dpg.mvTable_SizingStretchProp,
                borders_innerH=True,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ):
                dpg.add_table_column(init_width_or_weight=1)
                dpg.add_table_column(init_width_or_weight=1)
                with dpg.table_row():
                    dpg.add_text("Step")
                    dpg.add_input_float(
                        default_value=1.0,
                        min_value=0.0,
                        max_value=1000.0,
                        step=0.1,
                        width=110,
                        tag="snap_input_bottom",
                        callback=lambda s, a: app.on_snap_change(a),
                    )
                with dpg.table_row():
                    dpg.add_text("Angle Snap")
                    dpg.add_checkbox(
                        default_value=getattr(app, "angle_snap_enabled", False),
                        callback=lambda s, a: (setattr(app, "angle_snap_enabled", bool(a)), app._save_settings()),
                        tag="angle_snap_toggle",
                    )
                with dpg.table_row():
                    dpg.add_text("Angle")
                    dpg.add_input_float(
                        default_value=15.0,
                        min_value=0.0,
                        max_value=180.0,
                        step=1.0,
                        width=110,
                        tag="angle_snap_input",
                        callback=lambda s, a: (setattr(app, "angle_snap_deg", float(a)), app._save_settings()),
                    )
                with dpg.table_row():
                    dpg.add_text("Mag Snap")
                    dpg.add_checkbox(
                        default_value=getattr(app, "mag_snap_enabled", False),
                        callback=lambda s, a: (setattr(app, "mag_snap_enabled", bool(a)), app._save_settings()),
                        tag="mag_snap_toggle",
                    )
                with dpg.table_row():
                    dpg.add_text("Magnitude")
                    dpg.add_input_float(
                        default_value=0.25,
                        min_value=0.0,
                        max_value=1000.0,
                        step=0.05,
                        width=110,
                        tag="mag_snap_input",
                        callback=lambda s, a: (setattr(app, "mag_snap_step", max(0.0, float(a))), app._save_settings()),
                    )

    with dpg.window(
        label="Canvas Menu",
        tag="canvas_context_menu",
        show=False,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
        no_collapse=True,
        no_scrollbar=True,
        width=240,
    ):
        dpg.add_group(tag="canvas_context_menu_items")

    with dpg.window(
        label="Add",
        tag="add_menu_win",
        show=False,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
        no_collapse=True,
        width=360,
        height=420,
    ):
        dpg.add_text("Add", color=COLOR_TEXT_PRIMARY)
        dpg.add_text("Shift+A opens this palette. Choose once, place once.", color=COLOR_TEXT_MUTED, wrap=320)
        dpg.add_separator()
        dpg.add_text("Next placement: (0.00, 0.00)", tag="add_menu_anchor", color=COLOR_TEXT_SECONDARY)
        dpg.add_input_text(
            tag="add_menu_search",
            width=-1,
            hint="Search primitives, tools, presets...",
            callback=lambda s, a: app._refresh_add_menu(a),
        )
        dpg.add_spacer(height=6)
        with dpg.child_window(tag="add_menu_results", width=-1, height=280, border=True):
            pass
        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Close", width=90, callback=lambda: app.hide_add_menu())

    with dpg.window(
        label="Timeline",
        tag="statusbar_win",
        no_close=True,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
    ):
        with dpg.group(horizontal=True):
            dpg.add_button(label="|<", width=36, callback=lambda: app.restore_startup_scene())
            dpg.add_button(label="<", width=32, callback=lambda: app.undo())
            dpg.add_button(label=">", width=32, callback=lambda: app.redo())
            dpg.add_button(label="Play", width=48, callback=lambda: app.toggle_simulation())
            dpg.add_spacer(width=10)
            app.footer_workspace = dpg.add_text("BUILD", color=COLOR_ACCENT_PRIMARY)
            dpg.add_text("|", color=COLOR_TEXT_MUTED)
            app.footer_tool = dpg.add_text("SELECT", color=COLOR_TEXT_SECONDARY)
            dpg.add_text("|", color=COLOR_TEXT_MUTED)
            app.footer_hint = dpg.add_text("Shift+A adds parts. Right-click opens canvas actions. Ctrl+Drag pans.", color=COLOR_TEXT_MUTED)

    with dpg.handler_registry():
        dpg.add_mouse_down_handler(callback=app.on_mouse_down)
        dpg.add_mouse_release_handler(callback=app.on_mouse_up)
        dpg.add_mouse_move_handler(callback=app.on_mouse_move)
        dpg.add_mouse_drag_handler(callback=app.on_mouse_drag)
        dpg.add_mouse_wheel_handler(callback=app.on_mouse_wheel)
        dpg.add_key_press_handler(dpg.mvKey_A, callback=app.on_hotkey_a)

    for tag in (
        "topbar_win",
        "ribbon_win",
        "browser_win",
        "viewport_win",
        "inspector_win",
        "statusbar_win",
        "canvas_context_menu",
        "add_menu_win",
    ):
        dpg.bind_item_theme(tag, "cad_panel_theme")
    dpg.bind_item_theme("main_canvas", "cad_canvas_theme")


def build_forces_panel(app, parent_tag: str):
    try:
        if dpg.does_item_exist("lf_list"):
            return
        dpg.add_text("Attached local forces", parent=parent_tag, color=COLOR_TEXT_SECONDARY)
        dpg.add_text("In local scope, click a body and drag to define force.", parent=parent_tag, color=COLOR_TEXT_MUTED, wrap=260)
        dpg.add_child_window(tag="lf_list", parent=parent_tag, width=-1, height=150, border=True)
        _ui_refresh_local_forces(app)
    except Exception:
        pass


def _create_themes():
    if not dpg.does_item_exist("cad_panel_theme"):
        with dpg.theme(tag="cad_panel_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLOR_BG_SECONDARY)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLOR_BG_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, COLOR_BG_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, COLOR_BG_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Button, COLOR_BG_TERTIARY)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLOR_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLOR_ACCENT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLOR_BG_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, COLOR_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, COLOR_BG_TERTIARY)
                dpg.add_theme_color(dpg.mvThemeCol_Header, COLOR_BG_TERTIARY)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, COLOR_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, COLOR_BG_TERTIARY)
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Separator, COLOR_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_TEXT_PRIMARY)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, 0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0, 0)

    if not dpg.does_item_exist("cad_canvas_theme"):
        with dpg.theme(tag="cad_canvas_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (23, 28, 36, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (52, 61, 72, 255))

    _create_button_theme("mode_build_idle_theme", COLOR_BG_TERTIARY, COLOR_HOVER, COLOR_ACCENT_PRIMARY)
    _create_button_theme("mode_build_active_theme", COLOR_ACCENT_PRIMARY, COLOR_ACCENT_PRIMARY, COLOR_ACCENT_PRIMARY)
    _create_button_theme("mode_sim_idle_theme", COLOR_BG_TERTIARY, COLOR_HOVER, COLOR_ACCENT_SUCCESS)
    _create_button_theme("mode_sim_active_theme", COLOR_ACCENT_SUCCESS, COLOR_ACCENT_SUCCESS, COLOR_ACCENT_SUCCESS)
    _create_button_theme("tool_active_theme", (51, 83, 118, 255), (61, 101, 146, 255), COLOR_ACCENT_PRIMARY)
    _create_button_theme("tool_idle_theme", COLOR_BG_TERTIARY, COLOR_HOVER, COLOR_BORDER)
    _create_button_theme("danger_button_theme", (92, 42, 42, 255), (118, 52, 52, 255), (150, 70, 70, 255))
    _create_button_theme("success_button_theme", (39, 88, 62, 255), (48, 104, 74, 255), COLOR_ACCENT_SUCCESS)


def _create_button_theme(tag: str, base_color, hover_color, active_color):
    if dpg.does_item_exist(tag):
        return
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, base_color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover_color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active_color)
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_TEXT_PRIMARY)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2, 0)


def _ui_refresh_local_forces(app):
    try:
        panel = "lf_list"
        if not dpg.does_item_exist(panel):
            return
        dpg.delete_item(panel, children_only=True)
        for i, lf in enumerate(getattr(app.physics, "local_forces", []), start=1):
            with dpg.group(parent=panel):
                dpg.add_text(f"{lf.label}  {lf.magnitude:.2f} N  @  {lf.angle_deg:.1f} deg  ({lf.mode})")
                dpg.add_button(label="Remove", width=86, callback=_mk_del(app, i - 1))
                dpg.add_separator()
    except Exception:
        pass


def _mk_del(app, idx):
    return lambda: _ui_del_local_force(app, idx)


def _ui_del_local_force(app, idx):
    try:
        local_forces = getattr(app.physics, "local_forces", [])
        if 0 <= idx < len(local_forces):
            local_forces.pop(idx)
        _ui_refresh_local_forces(app)
    except Exception:
        pass


def _ui_calc_eval():
    try:
        import math as _math

        expr = dpg.get_value("calc_expr") or ""
        ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
        ns.update({"pi": _math.pi, "e": _math.e})
        val = eval(expr, {"__builtins__": {}}, ns)
        dpg.set_value("calc_result", f"Result: {val}")
    except Exception as exc:
        dpg.set_value("calc_result", f"Error: {exc}")

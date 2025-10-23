#UI Module

import dearpygui.dearpygui as dpg


def build_ui(app):
    # ---------- Left panel (decluttered with collapsible sections) ----------
    with dpg.window(label="Controls", pos=(10, 10), width=320, height=850,
                    no_close=True, no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True,
                    tag="controls_win"):
        wrap_w = 300
        dpg.add_text("FIJIXVALA", color=(59, 130, 246, 255), wrap=wrap_w)

        with dpg.collapsing_header(label="Overview", default_open=True):
            app.stats = dpg.add_text("Stats...", wrap=wrap_w)
            app.run_status = dpg.add_text(
                f"Status: {'Running' if app.physics.is_running else 'Paused'}",
                wrap=wrap_w
            )

        with dpg.collapsing_header(label="Simulation", default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=lambda: setattr(app.physics, "is_running", True))
                dpg.add_button(label="Pause", callback=lambda: setattr(app.physics, "is_running", False))
                dpg.add_button(label="Restart", callback=lambda: app.restart_sim())
                dpg.add_button(label="Go to Origin", callback=lambda: app.go_to_origin())
            dpg.add_text("Gravity (m/s^2)")
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

        with dpg.collapsing_header(label="Mode", default_open=True):
            dpg.add_radio_button(
                ["select", "fling", "surface", "force", "body"],
                default_value="select",
                callback=lambda s, a: app.on_mode_change(a or "select"),
            )
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text(
                    "Select - pick existing objects\n"
                    "Fling - click a body, drag, release to apply impulse at click point (or center if snapped)\n"
                    "Surface - draw static surface segment\n"
                    "Force - draw global force vector\n"
                    "Body - place a box or circle\n"
                    "Hold Ctrl and drag to pan the view. Hold Shift to select."
                )

        with dpg.collapsing_header(label="Snapping", default_open=False):
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp,
                           borders_innerH=True, borders_innerV=False, borders_outerH=False, borders_outerV=False):
                dpg.add_table_column(init_width_or_weight=1)
                dpg.add_table_column(init_width_or_weight=1)
                with dpg.table_row():
                    dpg.add_text("Snap to grid")
                    dpg.add_checkbox(
                        default_value=getattr(app, "snap", True),
                        callback=lambda s, a: setattr(app, "snap", a),
                        tag="snap_toggle",
                    )
                with dpg.table_row():
                    dpg.add_text("Grid (m)")
                    dpg.add_input_float(
                        default_value=1.0,
                        min_value=0.0,
                        max_value=1000.0,
                        step=0.1,
                        width=140,
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
                    dpg.add_text("Angle (deg)")
                    dpg.add_input_float(
                        default_value=15.0,
                        min_value=0.0,
                        max_value=180.0,
                        step=1.0,
                        width=140,
                        tag="angle_snap_input",
                        callback=lambda s, a: (setattr(app, "angle_snap_deg", float(a)), app._save_settings()),
                    )
                with dpg.table_row():
                    dpg.add_text("Magnitude Snap")
                    dpg.add_checkbox(
                        default_value=getattr(app, "mag_snap_enabled", False),
                        callback=lambda s, a: (setattr(app, "mag_snap_enabled", bool(a)), app._save_settings()),
                        tag="mag_snap_toggle",
                    )
                with dpg.table_row():
                    dpg.add_text("Magnitude (m)")
                    dpg.add_input_float(
                        default_value=0.25,
                        min_value=0.0,
                        max_value=1000.0,
                        step=0.05,
                        width=140,
                        tag="mag_snap_input",
                        callback=lambda s, a: (setattr(app, "mag_snap_step", max(0.0, float(a))), app._save_settings()),
                    )

        # Forces controls are embedded into Tool Properties via build_forces_panel

        with dpg.collapsing_header(label="Edit", default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Undo", callback=lambda: app.undo())
                dpg.add_button(label="Redo", callback=lambda: app.redo())
                dpg.add_button(label="Delete", callback=lambda: app.delete_selected())
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Deletes currently selected objects.")

        with dpg.collapsing_header(label="Tool Properties", default_open=False):
            with dpg.child_window(tag="tool_panel", width=wrap_w, height=260, border=True):
                dpg.add_text("Select a tool to edit its defaults.")

        with dpg.collapsing_header(label="Selection Properties", default_open=True):
            with dpg.child_window(tag="prop_panel", width=wrap_w, height=300, border=True):
                dpg.add_text("Select an item to edit its properties.")

        with dpg.collapsing_header(label="Calculator", default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="calc_expr", width=200, hint="e.g. 2*sin(pi/3)")
                dpg.add_button(label="=", callback=lambda: _ui_calc_eval())
            dpg.add_text("Result: ", tag="calc_result")

    # ---------- Main viewport window ----------
    with dpg.window(label="Viewport", pos=(360, 10), width=1040, height=880,
                    no_close=True, no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True,
                    tag="viewport_win"):
        with dpg.group(horizontal=True, tag="vp_row"):
            with dpg.group(tag="vp_left"):
                # Base drawlist for physics rendering (full height)
                main_canvas = dpg.add_drawlist(width=900, height=850, tag="main_canvas")
                # Attach per-canvas handlers so click/release don't misfire off-canvas
                with dpg.item_handler_registry(tag="canvas_handlers"):
                    dpg.add_item_clicked_handler(callback=app.on_mouse_down)
                    dpg.add_item_deactivated_handler(callback=app.on_mouse_up)
                dpg.bind_item_handler_registry("main_canvas", "canvas_handlers")
            with dpg.child_window(tag="hier_panel", width=260, height=850, border=True):
                dpg.add_group(tag="hier_tree")

        # Global handlers for move/drag, wheel (down/up handled per-canvas)
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=app.on_mouse_move)
            dpg.add_mouse_drag_handler(callback=app.on_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=app.on_mouse_wheel)

    # ---------- Theme ----------
    with dpg.theme(tag="viewport_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)
    dpg.bind_item_theme("viewport_win", "viewport_theme")

def build_forces_panel(app, parent_tag: str):
    try:
        with dpg.group(parent=parent_tag):
            dpg.add_text("Forces", color=(200, 200, 220, 255))
            dpg.add_separator()
            dpg.add_text("Local Forces")
            dpg.add_text("Click a body in Local scope, then drag to set force.")
            dpg.add_separator()
            dpg.add_child_window(tag="lf_list", width=280, height=180, border=True)
            _ui_refresh_local_forces(app)
    except Exception:
        pass

# ---------- Local helpers for UI callbacks ----------
def _ui_add_local_force(app):
    # Deprecated: Local forces are now drawn directly in canvas
    return

def _ui_refresh_local_forces(app):
    try:
        panel = 'lf_list'
        if not dpg.does_item_exist(panel):
            return
        dpg.delete_item(panel, children_only=True)
        for i, lf in enumerate(getattr(app.physics, 'local_forces', []), start=1):
            with dpg.group(parent=panel, horizontal=True):
                dpg.add_text(f"{lf.label}: mag={lf.magnitude:.2f}N, ang={lf.angle_deg:.1f}°, mode={lf.mode}")
                def _mk_del(idx=i-1):
                    return lambda: _ui_del_local_force(app, idx)
                dpg.add_button(label="Remove", callback=_mk_del())
    except Exception:
        pass

def _ui_del_local_force(app, idx):
    try:
        l = getattr(app.physics, 'local_forces', [])
        if 0 <= idx < len(l):
            l.pop(idx)
        _ui_refresh_local_forces(app)
    except Exception:
        pass

def _ui_calc_eval():
    try:
        import math as _m
        expr = dpg.get_value('calc_expr') or ''
        # Tiny safe eval: only math namespace, no builtins
        ns = {k: getattr(_m, k) for k in dir(_m) if not k.startswith('_')}
        ns.update({'pi': _m.pi, 'e': _m.e})
        val = eval(expr, {"__builtins__": {}}, ns)
        dpg.set_value('calc_result', f"Result: {val}")
    except Exception as e:
        dpg.set_value('calc_result', f"Error: {e}")

#FiZiXMeCH

import time
import math
import dearpygui.dearpygui as dpg
import json
import os

from physics import Physics, Vector
from renderer import Renderer
from ui import build_ui, build_forces_panel

# ---------- CONFIG ----------
WIN_W, WIN_H = 1400, 900
CANVAS_W, CANVAS_H = 1000, 850
PANEL_W = 360
ZOOM_FACTOR = 0.5
ZOOM_MIN, ZOOM_MAX = 20.0, 400.0
DEFAULT_ZOOM = 80.0
CLICK_THRESHOLD = 20
MIN_MAGNITUDE = 0.01
ZOOM_HALFLIFE = 0.12


class App:
    def __init__(self) -> None:
        try:
            dpg.destroy_context()
        except Exception:
            pass

        self.physics = Physics()
        self.R: Renderer = None

        # Interaction / state
        self.mode = "select"
        self.snap = True
        self.dragging = False
        self.drag_start = (0.0, 0.0)
        self.drag_curr = (0.0, 0.0)
        self.drag_start_raw = (0.0, 0.0)
        self.drag_curr_raw = (0.0, 0.0)
        self.surface_start_local = None
        self.surface_curr_local = None
        self.surface_preview_id = None

        # Vector drawing
        self.is_vector_drawing = False
        self.preview_vector = None
        self.resultant = None
        self.vector_counter = 0
        # Force tool scope: 'global' or 'local'
        self.force_scope = "global"
        self._lf_body_target_id = None

        # Fling tool state/settings
        self.fling_scale = 1.0  # impulse per world unit drag (N·s/m)
        self.fling_limit_speed = True
        self.fling_max_speed = 50.0
        self.fling_snap_to_center = False
        self._fling_selected_body = None
        self._fling_apply_point = None
        self._last_fling_time = 0.0
        self._fling_cooldown = 0.10  # seconds between flings

        # Angle/Magnitude snapping
        self.angle_snap_enabled = False
        self.angle_snap_deg = 15.0
        self.mag_snap_enabled = False
        self.mag_snap_step = 0.25

        # UI references
        self.stats = None

        # Panning
        self.is_panning = False
        self.pan_start_local = (0.0, 0.0)
        self.pan_cam_start = (0.0, 0.0)

        # Snapping (world units)
        self.snap_step = 1.0


        # Selection/editor state
        self.selected_type = None  # 'circle' | 'box' | 'surface' | 'vector'
        self.selected_ref = None
        self.hier_dirty = True

        # Tool defaults
        self.body_tool_type = "box"  # 'box' or 'circle'
        self.body_default_mass = 1.0
        self.body_box_w = 1.0
        self.body_box_h = 1.0
        self.body_circle_r = 0.5
        # Surface defaults (per-surface adjustable)
        self.surface_radius = 0.05
        self.surface_static_mu = 0.8
        self.surface_dynamic_mu = 0.6
        self.surface_elasticity = 0.3

        # DearPyGui setup
        dpg.create_context()
        dpg.create_viewport(title="FZXMCH", width=WIN_W, height=WIN_H)
        try:
            # Windowed mode with decorations
            dpg.configure_viewport(decorated=True)
        except Exception:
            pass

        build_ui(self)
        self.R = Renderer("main_canvas", CANVAS_W, CANVAS_H)
        self.R.zoom = DEFAULT_ZOOM

        # Load persisted settings
        self._load_settings()
        try:
            if dpg.does_item_exist("snap_input"):
                dpg.set_value("snap_input", float(self.snap_step))
            if dpg.does_item_exist("snap_input_bottom"):
                dpg.set_value("snap_input_bottom", float(self.snap_step))
            if dpg.does_item_exist("snap_toggle"):
                dpg.set_value("snap_toggle", bool(self.snap))
            if dpg.does_item_exist("gravity_slider"):
                dpg.set_value("gravity_slider", float(self.physics.space.gravity[1]))
            self._update_tool_panel()
        except Exception:
            pass

        # Smooth zoom state
        self.zoom_target = DEFAULT_ZOOM
        self.zoom_focus_local = (CANVAS_W * 0.5, CANVAS_H * 0.5)
        self.zoom_focus_world = (0.0, 0.0)
        self.zoom_focus_ttl = 0.0
        self.zoom_focus_active = False

        self._last = time.time()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # History (undo/redo) and initial snapshot
        self._undo_stack = []
        self._redo_stack = []
        self._startup_snapshot = self._snapshot_state()

    # ---------- Mouse utilities ----------
    def _local_mouse(self):
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        try:
            rect_min = dpg.get_item_rect_min("main_canvas")
        except Exception:
            rect_min = dpg.get_item_pos("main_canvas")
        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]
        return local_x, local_y

    def _adjust_zoom(self, delta):
        self.zoom_target = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_target + delta))

    def _set_zoom_focus(self, local_x, local_y):
        wx, wy = self.R.to_world(local_x, local_y)
        self.zoom_focus_local = (local_x, local_y)
        self.zoom_focus_world = (wx, wy)
        self.zoom_focus_ttl = 0.4
        self.zoom_focus_active = True

    # ----- Settings callbacks -----
    def on_snap_change(self, value):
        try:
            v = float(value)
        except Exception:
            v = self.snap_step
        if v <= 0.0:
            self.snap = False
        else:
            self.snap = True
            self.snap_step = max(1e-6, v)
        self._save_settings()

    def on_gravity_change(self, value):
        try:
            gv = float(value)
        except Exception:
            return
        try:
            self.physics.space.gravity = (0, gv)
        except Exception:
            pass
        self._save_settings()

    def go_to_origin(self):
        try:
            self.R.cam[0] = 0.0
            self.R.cam[1] = 0.0
        except Exception:
            pass

    def _snap_world(self, wx: float, wy: float):
        if not getattr(self, 'snap', False):
            return wx, wy
        try:
            step = max(1e-9, float(self.snap_step))
        except Exception:
            step = 1.0
        return round(wx / step) * step, round(wy / step) * step

    def _snap_angle_mag(self, p0: tuple[float, float], p1: tuple[float, float]):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = (x1 - x0), (y1 - y0)
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) if r > 1e-12 else 0.0
        # angle snap
        if getattr(self, 'angle_snap_enabled', False):
            try:
                step_deg = float(self.angle_snap_deg)
            except Exception:
                step_deg = 15.0
            step_rad = max(1e-9, math.radians(step_deg))
            theta = round(theta / step_rad) * step_rad
        # magnitude snap
        if getattr(self, 'mag_snap_enabled', False):
            try:
                mstep = float(self.mag_snap_step)
            except Exception:
                mstep = 0.25
            mstep = max(1e-9, mstep)
            r = round(r / mstep) * mstep
        return (x0 + r * math.cos(theta), y0 + r * math.sin(theta))

    def _animate_zoom(self, dt):
        if dt <= 0:
            return
        alpha = 1.0 - (0.5 ** (dt / max(1e-6, ZOOM_HALFLIFE)))
        old_zoom = self.R.zoom
        target = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_target))
        new_zoom = old_zoom + (target - old_zoom) * alpha
        if abs(new_zoom - old_zoom) >= 1e-6:
            self.R.zoom = new_zoom
            if self.zoom_focus_active or abs(target - new_zoom) > 1e-4:
                sx, sy = self.zoom_focus_local
                wx, wy = self.zoom_focus_world
                cx, cy = self.R._mid()
                self.R.cam[0] = (sx - cx) / self.R.zoom - wx
                self.R.cam[1] = -(sy - cy) / self.R.zoom - wy
        else:
            self.R.zoom = new_zoom
        if self.zoom_focus_active:
            self.zoom_focus_ttl -= dt
            if self.zoom_focus_ttl <= 0:
                self.zoom_focus_active = False

    def _update_surface_preview(self):
        if self.mode != "surface" or not self.dragging:
            return
        if not self.surface_start_local or not self.surface_curr_local:
            return
        self._clear_surface_preview()
        try:
            sx, sy = self.surface_start_local
            ex, ey = self.surface_curr_local
            w0 = self.R.to_world(sx, sy)
            w1 = self.R.to_world(ex, ey)
            if self.snap:
                w0 = self._snap_world(*w0)
                w1 = self._snap_world(*w1)
            p0 = self.R.to_screen(*w0)
            p1 = self.R.to_screen(*w1)
            dpg.draw_line(
                p0,
                p1,
                color=(120, 255, 255, 200),
                thickness=2,
                parent="overlay_drawlist",
                tag="surface_preview",
            )
        except Exception:
            pass

    def _clear_surface_preview(self):
        if dpg.does_item_exist("surface_preview"):
            dpg.delete_item("surface_preview")  

    def on_mode_change(self, mode: str):
        self.mode = mode
        if mode != "surface":
            self.surface_start_local = None
            self.surface_curr_local = None
            self._clear_surface_preview()
        if mode != "fling":
            self._fling_selected_body = None
            self._fling_apply_point = None
        if mode != "force":
            self.is_vector_drawing = False
            self.preview_vector = None
            self._lf_body_target_id = None
        try:
            self._update_tool_panel()
        except Exception:
            pass

    def on_force_scope_change(self, scope: str):
        scope = (scope or 'global').lower()
        if scope not in ('global', 'local'):
            scope = 'global'
        self.force_scope = scope
        # Reset any in-progress preview when switching scope
        self.is_vector_drawing = False
        self.preview_vector = None
        self._lf_body_target_id = None
        try:
            self._update_tool_panel()
        except Exception:
            pass

    def on_key_down(self, sender, app_data):
        key = app_data
        k_space = getattr(dpg, "mvKey_Space", None)
        k_plus = getattr(dpg, "mvKey_Equal", None)
        k_minus = getattr(dpg, "mvKey_Minus", None)
        k_ctrl = getattr(dpg, "mvKey_Control", None)
        k_lctrl = getattr(dpg, "mvKey_LControl", None)
        k_rctrl = getattr(dpg, "mvKey_RControl", None)
        k_z = getattr(dpg, "mvKey_Z", None)
        k_y = getattr(dpg, "mvKey_Y", None)
        k_del = getattr(dpg, "mvKey_Delete", None)
        k_r = getattr(dpg, "mvKey_R", None)

        if key == k_space:
            self.physics.is_running = not self.physics.is_running
            return
        if key == k_plus:
            self.zoom_target = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_target * (1.0 + ZOOM_FACTOR)))
            return
        if key == k_minus:
            self.zoom_target = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_target / (1.0 + ZOOM_FACTOR)))
            return

        def _is_down(code):
            return (code is not None) and bool(dpg.is_key_down(code))
        ctrl_down = _is_down(k_ctrl) or _is_down(k_lctrl) or _is_down(k_rctrl)
        if ctrl_down and key == k_z:
            self.undo()
            return
        if ctrl_down and key == k_y:
            self.redo()
            return
        if key == k_del:
            self.delete_selected()
            return
        if ctrl_down and key == k_r:
            self.restart_sim()
            return

    # ---- Hotkey helpers for explicit key handlers ----
    def _ctrl_down(self) -> bool:
        keys = [
            getattr(dpg, "mvKey_Control", None),
            getattr(dpg, "mvKey_LControl", None),
            getattr(dpg, "mvKey_RControl", None),
        ]
        for code in keys:
            try:
                if code is not None and dpg.is_key_down(code):
                    return True
            except Exception:
                pass
        return False

    def on_hotkey_z(self, sender, app_data):
        if self._ctrl_down():
            self.undo()

    def on_hotkey_y(self, sender, app_data):
        if self._ctrl_down():
            self.redo()

    def on_hotkey_delete(self, sender, app_data):
        self.delete_selected()

    # no radio sync needed; value drives mode

    # ---------- Mouse handlers ----------
    def on_mouse_down(self, sender, app_data):
        # Ensure clicks are on the canvas for canvas interactions
        if not dpg.is_item_hovered("main_canvas"):
            return
        local_x, local_y = self._local_mouse()
        # Ctrl+Drag panning (replaces right-click pan)
        try:
            ctrl_down = self._ctrl_down()
        except Exception:
            ctrl_down = False
        if ctrl_down:
            self.is_panning = True
            self.pan_start_local = (local_x, local_y)
            self.pan_cam_start = (self.R.cam[0], self.R.cam[1])
            return

        wx_unsnap, wy_unsnap = self.R.to_world(local_x, local_y)
        wx, wy = wx_unsnap, wy_unsnap
        if self.snap:
            wx, wy = self._snap_world(wx, wy)
        self.dragging = True
        self.drag_start_raw = (wx, wy)
        self.drag_curr_raw = (wx, wy)
        self.drag_start = (wx, wy)
        self.drag_curr = (wx, wy)

        # Shift acts as a temporary Select tool
        try:
            shift_down = bool(dpg.is_key_down(getattr(dpg, 'mvKey_Shift', None))) or \
                         bool(dpg.is_key_down(getattr(dpg, 'mvKey_LShift', None))) or \
                         bool(dpg.is_key_down(getattr(dpg, 'mvKey_RShift', None)))
        except Exception:
            shift_down = False

        if self.mode == "surface" and not shift_down:
            self._clear_surface_preview()
            self.surface_start_local = (local_x, local_y)
            self.surface_curr_local = (local_x, local_y)
            # Draw initial preview so the ghost appears immediately on press
            self._update_surface_preview()

        if self.mode == "select" or shift_down:
            import pymunk
            p = pymunk.Vec2d(*self.drag_start)
            body = self.physics.pick_body(p)
            if body is not None:
                shape = None
                for sh in self.physics.dynamic:
                    if sh.body is body:
                        shape = sh
                        break
                if shape is not None:
                    if hasattr(shape, 'radius'):
                        self.selected_type, self.selected_ref = 'circle', shape
                    elif isinstance(shape, pymunk.Poly):
                        self.selected_type, self.selected_ref = 'box', shape
                    else:
                        self.selected_type, self.selected_ref = 'body', shape
                    self._update_properties_panel()
                    self.dragging = False
                    return
            seg = self.physics.pick_surface(p)
            if seg is not None:
                self.selected_type, self.selected_ref = 'surface', seg
                self._update_properties_panel()
                self.dragging = False
                return
            V = self.physics.pick_vector(p)
            if V is not None:
                self.selected_type, self.selected_ref = 'vector', V
                self._update_properties_panel()
                self.dragging = False
                return

        if self.mode == "force":
            if getattr(self, 'force_scope', 'global') == 'global':
                self.is_vector_drawing = True
                self.preview_vector = Vector("GHOST", 0, 0, wx, wy)
            else:
                # Local force: require clicking a body; anchor at body COM
                try:
                    import pymunk
                    p = pymunk.Vec2d(wx, wy)
                except Exception:
                    p = type("P", (), {"x": wx, "y": wy})
                body = self.physics.pick_body(p)
                if body is None:
                    # No body under cursor; don't start drawing
                    self.dragging = False
                    return
                # Select corresponding shape for visual feedback
                try:
                    for sh in self.physics.dynamic:
                        if sh.body is body:
                            self.selected_type = 'circle' if hasattr(sh, 'radius') else 'box'
                            self.selected_ref = sh
                            break
                except Exception:
                    pass
                self._lf_body_target_id = id(body)
                sx, sy = float(body.position.x), float(body.position.y)
                self.is_vector_drawing = True
                self.preview_vector = Vector("GHOST", sx, sy, wx, wy)
        elif self.mode == "fling":
            try:
                import pymunk
                p = pymunk.Vec2d(wx, wy)
            except Exception:
                p = type("P", (), {"x": wx, "y": wy})
            body = self.physics.pick_body(p)
            self._fling_selected_body = body
            if body is not None:
                if self.fling_snap_to_center:
                    self._fling_apply_point = (float(body.position.x), float(body.position.y))
                else:
                    self._fling_apply_point = (wx, wy)
            else:
                self._fling_apply_point = None

    def on_mouse_drag(self, sender, app_data):
        local_x, local_y = self._local_mouse()
        wx_unsnap, wy_unsnap = self.R.to_world(local_x, local_y)
        if self.mode == "surface":
            self.surface_curr_local = (local_x, local_y)
            self._update_surface_preview()
        wx, wy = wx_unsnap, wy_unsnap
        if self.snap:
            wx, wy = self._snap_world(wx, wy)
        if self.is_panning:
            dxs = local_x - self.pan_start_local[0]
            dys = local_y - self.pan_start_local[1]
            self.R.cam[0] = self.pan_cam_start[0] + dxs / self.R.zoom
            self.R.cam[1] = self.pan_cam_start[1] - dys / self.R.zoom
            return
        if not self.dragging:
            return
        self.drag_curr_raw = (wx, wy)
        self.drag_curr = (wx, wy)
        if self.mode == "force" and self.is_vector_drawing:
            # Keep origin at preview's anchor (0,0 for global; body COM for local)
            try:
                sx, sy = self.preview_vector.x1, self.preview_vector.y1
            except Exception:
                sx, sy = 0.0, 0.0
            self.preview_vector = Vector("GHOST", sx, sy, wx, wy)

    def on_mouse_move(self, sender, app_data):
        local_x, local_y = self._local_mouse()
        if self.mode == "surface" and self.dragging:
            self.surface_curr_local = (local_x, local_y)
            self._update_surface_preview()
        wx, wy = self.R.to_world(local_x, local_y)
        if self.snap:
            wx, wy = self._snap_world(wx, wy)
        self.drag_curr_raw = (wx, wy)
        self.drag_curr = (wx, wy)

    def on_mouse_up(self, sender, app_data):
        if self.is_panning:
            self.is_panning = False
            return
        if not self.dragging:
            return
        self.dragging = False

        local_x, local_y = self._local_mouse()
        wx_unsnap, wy_unsnap = self.R.to_world(local_x, local_y)
        if self.mode == "surface":
            self.surface_curr_local = (local_x, local_y)
        wx, wy = wx_unsnap, wy_unsnap
        if self.snap:
            wx, wy = self._snap_world(wx, wy)
        self.drag_curr_raw = (wx, wy)

        if self.mode == "surface":
            start_local = self.surface_start_local
            end_local = self.surface_curr_local or start_local
            self._clear_surface_preview()
            if start_local is None or end_local is None:
                self.surface_start_local = None
                self.surface_curr_local = None
                return
            start_world = self.R.to_world(*start_local)
            end_world = self.R.to_world(*end_local)
            if math.dist(start_world, end_world) > 0.01:
                seg_start = start_world
                if self.snap:
                    seg_start = self._snap_world(*seg_start)
                # apply dir/mag snap
                seg_end = self._snap_angle_mag(seg_start, end_world)
                self._push_undo()
                s = self.physics.add_surface(
                    seg_start,
                    seg_end,
                    radius=float(self.surface_radius),
                    static_mu=float(self.surface_static_mu),
                    dynamic_mu=float(self.surface_dynamic_mu),
                    elasticity=float(self.surface_elasticity),
                )
                try:
                    import pymunk as _pm
                    if isinstance(s, _pm.Segment):
                        idx = sum(1 for sh in self.physics.static if isinstance(sh, _pm.Segment))
                        setattr(s, 'name', getattr(s, 'name', None) or f"Surface {idx}")
                except Exception:
                    pass
                self.hier_dirty = True
            self.surface_start_local = None
            self.surface_curr_local = None
            return

        if self.mode == "body":
            if math.dist(self.drag_curr_raw, self.drag_start_raw) < CLICK_THRESHOLD:
                pos = self.drag_start
                if self.body_tool_type == "circle":
                    self._push_undo()
                    r = float(self.body_circle_r)
                    s = self.physics.add_circle(r=r, pos=pos, m=float(self.body_default_mass))
                    try:
                        setattr(s, 'name', f"Circle {len(self.physics.dynamic)}")
                    except Exception:
                        pass
                else:
                    self._push_undo()
                    w = float(self.body_box_w)
                    h = float(self.body_box_h)
                    s = self.physics.add_box(w=w, h=h, pos=pos, m=float(self.body_default_mass))
                    try:
                        setattr(s, 'name', f"Box {len(self.physics.dynamic)}")
                    except Exception:
                        pass
                self.hier_dirty = True

        elif self.mode == "force" and self.is_vector_drawing:
            # Finalize global or local force drawing
            try:
                sx, sy = self.preview_vector.x1, self.preview_vector.y1
            except Exception:
                sx, sy = 0.0, 0.0
            ex, ey = self._snap_angle_mag((sx, sy), (wx, wy))
            dx, dy = (ex - sx), (ey - sy)
            if math.hypot(dx, dy) >= MIN_MAGNITUDE:
                scope = getattr(self, 'force_scope', 'global')
                if scope == 'global':
                    self._push_undo()
                    self.vector_counter += 1
                    V = Vector(f"G{self.vector_counter}", sx, sy, ex, ey)
                    self.physics.vectors.append(V)
                    self.hier_dirty = True
                else:
                    # Local force: attach to selected body id
                    try:
                        from physics import LocalForce
                        ang = math.degrees(math.atan2(dy, dx))
                        mag = math.hypot(dx, dy)
                        bid = self._lf_body_target_id
                        if bid is not None:
                            idx = len(getattr(self.physics, 'local_forces', [])) + 1
                            lf = LocalForce(label=f"L{idx}", body_id=bid, magnitude=float(mag), angle_deg=float(ang), mode='world')
                            self._push_undo()
                            self.physics.local_forces.append(lf)
                            self.hier_dirty = True
                    except Exception:
                        pass
            self.preview_vector = None
            self.is_vector_drawing = False
            self._lf_body_target_id = None
            try:
                self._update_tool_panel()
            except Exception:
                pass
        elif self.mode == "fling":
            now = time.time()
            if (now - self._last_fling_time) >= self._fling_cooldown:
                body = self._fling_selected_body
                apply_pt = self._fling_apply_point
                if body is not None and apply_pt is not None:
                    sx, sy = apply_pt
                    ex, ey = self._snap_angle_mag(apply_pt, (wx, wy))
                    dx = ex - sx
                    dy = ey - sy
                    if math.hypot(dx, dy) >= MIN_MAGNITUDE:
                        Jx = float(self.fling_scale) * dx
                        Jy = float(self.fling_scale) * dy
                        try:
                            self._push_undo()
                        except Exception:
                            pass
                        self.physics.apply_impulse(body, (Jx, Jy), apply_pt)
                        if self.fling_limit_speed:
                            self.physics.clamp_body_speed(body, float(self.fling_max_speed))
                        self._last_fling_time = now
            self._fling_selected_body = None
            self._fling_apply_point = None

        self._update_properties_panel()

    def on_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("main_canvas"):
            local_x, local_y = self._local_mouse()
            self._set_zoom_focus(local_x, local_y)
            try:
                steps = int(app_data)
            except Exception:
                steps = 1 if app_data > 0 else -1
            if steps != 0:
                factor = (1.0 + ZOOM_FACTOR) ** steps
                self.zoom_target = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_target * factor))

    # ---------- Rendering ----------
    def _render(self):
        R = self.R
        try:
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
            margin = 10
            left_w = max(300, int(vw * 0.26))
            if dpg.does_item_exist("controls_win"):
                dpg.configure_item("controls_win", pos=(margin, margin), width=left_w, height=vh - margin * 2)
            vx = margin + left_w + margin
            vw_w = max(300, vw - vx - margin)
            vw_h = vh - margin * 2
            if dpg.does_item_exist("viewport_win"):
                dpg.configure_item("viewport_win", pos=(vx, margin), width=vw_w, height=vw_h)
            right_w = 260
            # If a bottom world panel exists, reserve its height; otherwise, use full height
            bottom_h = 0
            if dpg.does_item_exist("world_panel"):
                # legacy support if present
                bottom_h = 140
            if dpg.does_item_exist("hier_panel"):
                dpg.configure_item("hier_panel", width=right_w, height=max(50, vw_h - 20))
            left_area_w = max(200, vw_w - right_w - 20)
            if dpg.does_item_exist("main_canvas"):
                # Fill available height minus small padding
                canvas_h = max(200, vw_h - bottom_h - 20)
                dpg.configure_item("main_canvas", width=left_area_w, height=canvas_h)
            if dpg.does_item_exist("overlay_drawlist"):
                dpg.configure_item("overlay_drawlist", width=left_area_w, height=canvas_h)
        except Exception:
            pass

        if self.hier_dirty:
            try:
                self._rebuild_hierarchy()
            except Exception as e:
                print("[WARN] Hierarchy rebuild failed:", e)
            self.hier_dirty = False

        try:
            size_w, size_h = dpg.get_item_rect_size("main_canvas")
            if size_w and size_h:
                R.w, R.h = int(size_w), int(size_h)
        except Exception:
            pass

        R.clear()
        R.draw_grid(spacing=1.0)
        try:
            R.draw_bounds(self.physics.bounds_limit)
        except Exception:
            pass

        if self.dragging and self.mode == "surface":
            try:
                if self.surface_start_local and self.surface_curr_local:
                    sx, sy = self.surface_start_local
                    ex, ey = self.surface_curr_local
                    w0 = self.R.to_world(sx, sy)
                    w1 = self.R.to_world(ex, ey)
                    # grid snap anchor
                    if self.snap:
                        w0 = self._snap_world(*w0)
                    # apply angle/magnitude snap on delta
                    w1 = self._snap_angle_mag(w0, w1)
                    p0 = self.R.to_screen(*w0)
                    p1 = self.R.to_screen(*w1)
                    dpg.draw_line(p0, p1, color=(120, 255, 255, 200), thickness=2, parent=self.R.tag)
            except Exception:
                pass
        elif self.dragging and self.mode == "fling" and self._fling_apply_point is not None:
            try:
                p0 = self._fling_apply_point
                p1 = self._snap_angle_mag(p0, self.drag_curr)
                self.R.draw_dotted_world(p0, p1, color=(59,130,246,220), dash=8, gap=6)
                p1s = self.R.to_screen(*p1)
                mag = math.dist(p0, p1) * float(self.fling_scale)
                dpg.draw_text((int(p1s[0]) + 8, int(p1s[1]) - 14), f"J={mag:.2f}", color=(59,130,246,220), parent=self.R.tag, size=14)
            except Exception:
                pass
        elif self.mode == "force" and self.preview_vector:
            # draw snapped preview
            try:
                sx, sy = self.preview_vector.x1, self.preview_vector.y1
                ex, ey = self._snap_angle_mag((sx, sy), (self.drag_curr[0], self.drag_curr[1]))
                Vg = Vector(self.preview_vector.label, sx, sy, ex, ey)
            except Exception:
                Vg = self.preview_vector
            R.draw_vector(Vg, color=(180, 180, 180, 220))
        elif self.mode == "body":
            cx, cy = self.drag_curr
            if self.body_tool_type == "circle":
                R.ghost_circle((cx, cy), float(self.body_circle_r))
            else:
                R.ghost_box((cx, cy), float(self.body_box_w), float(self.body_box_h))

        for s in self.physics.static:
            R.draw_shape(s)
            try:
                R.draw_shape_label(s)
            except Exception:
                pass
        for s in self.physics.dynamic:
            R.draw_shape(s)
            try:
                R.draw_shape_label(s)
            except Exception:
                pass
        # Draw local forces as small arrows at body COM
        try:
            for lf in getattr(self.physics, 'local_forces', []):
                # find body by id
                body = None
                for sh in self.physics.dynamic:
                    if id(sh.body) == lf.body_id:
                        body = sh.body
                        break
                if body is None:
                    continue
                base = (float(body.position.x), float(body.position.y))
                # Visual scale for arrow (non-physical; aids visualization)
                ang = math.radians(float(lf.angle_deg))
                if lf.mode == 'body':
                    ang = float(body.angle) + ang
                length = max(0.4, min(2.0, float(lf.magnitude) * 0.05))
                tip = (base[0] + length * math.cos(ang), base[1] + length * math.sin(ang))
                tmp = Vector(lf.label or 'LF', base[0], base[1], tip[0], tip[1])
                R.draw_vector(tmp, color=(59, 130, 246, 220))
        except Exception:
            pass
        # Live property updates (COM, angles)
        try:
            if self.selected_type in ("circle", "box") and self.selected_ref is not None:
                sh = self.selected_ref
                comx, comy = float(sh.body.position.x), float(sh.body.position.y)
                if dpg.does_item_exist("prop_com"):
                    dpg.set_value("prop_com", f"COM: ({comx:.3f}, {comy:.3f})")
                ang_deg = math.degrees(float(getattr(sh.body, 'angle', 0.0)))
                if dpg.does_item_exist("prop_angle"):
                    dpg.set_value("prop_angle", float(ang_deg))
                if dpg.does_item_exist("prop_angle_ro"):
                    dpg.set_value("prop_angle_ro", f"{ang_deg:.2f}")
        except Exception:
            pass
            R.draw_velocity(s)
        for V in self.physics.vectors:
            R.draw_vector(V)

        if self.physics.vectors:
            sx = sum(V.Fx for V in self.physics.vectors)
            sy = sum(V.Fy for V in self.physics.vectors)
            if abs(sx) + abs(sy) >= MIN_MAGNITUDE:
                self.resultant = Vector("R", 0, 0, sx, sy)
                R.draw_vector(self.resultant, color=(239, 68, 68, 255))
                R.draw_dotted_world((sx, sy), (sx, 0), color=(239, 68, 68, 180))
                R.draw_dotted_world((sx, sy), (0, sy), color=(239, 68, 68, 180))

        # Highlight selection
        try:
            if self.selected_ref and self.selected_type in ("circle", "box", "surface"):
                R.highlight_shape(self.selected_ref)
            elif self.selected_ref and self.selected_type == 'vector':
                R.draw_vector(self.selected_ref, color=(250, 204, 21, 220))
        except Exception:
            pass

        if hasattr(self, "_prop_update_timer"):
            self._prop_update_timer += 1
        else:
            self._prop_update_timer = 0
        if self._prop_update_timer % 10 == 0:
            self._update_properties_panel()

    # ---------- Hierarchy ----------
    def _on_hier_select(self, sender, app_data, user_data):
        try:
            typ, ref = user_data
        except Exception:
            return
        self.selected_type = typ
        self.selected_ref = ref
        self._update_properties_panel()

    def _rebuild_hierarchy(self):
        if not dpg.does_item_exist("hier_tree"):
            return
        try:
            dpg.delete_item("hier_tree", children_only=True)
        except Exception:
            pass
        bodies_node = dpg.add_tree_node(label="Bodies", default_open=True, parent="hier_tree")
        for i, sh in enumerate(self.physics.dynamic):
            label = "Circle" if hasattr(sh, 'radius') else "Box"
            base = f"{label} {i+1}"
            if not getattr(sh, 'name', None):
                try:
                    setattr(sh, 'name', base)
                except Exception:
                    pass
            name = getattr(sh, 'name', base)
            typ = 'circle' if label == 'Circle' else 'box'
            dpg.add_selectable(label=name, parent=bodies_node, callback=self._on_hier_select, user_data=(typ, sh))
        import pymunk
        surfaces_node = dpg.add_tree_node(label="Surfaces", default_open=True, parent="hier_tree")
        idx = 0
        for sh in self.physics.static:
            if isinstance(sh, pymunk.Segment):
                idx += 1
                if not getattr(sh, 'name', None):
                    try:
                        setattr(sh, 'name', f"Surface {idx}")
                    except Exception:
                        pass
                sname = getattr(sh, 'name', f"Surface {idx}")
                dpg.add_selectable(label=sname, parent=surfaces_node, callback=self._on_hier_select, user_data=('surface', sh))
        forces_node = dpg.add_tree_node(label="Forces", default_open=True, parent="hier_tree")
        for V in self.physics.vectors:
            label = V.label or "Force"
            dpg.add_selectable(label=label, parent=forces_node, callback=self._on_hier_select, user_data=('vector', V))

    # ---------- Properties Panel ----------
    def _update_properties_panel(self):
        panel = "prop_panel"
        if not dpg.does_item_exist(panel):
            return
        if not self.selected_ref or not self.selected_type:
            dpg.configure_item(panel, label="Properties: None")
            dpg.delete_item(panel, children_only=True)
            dpg.add_text("Select an item to edit its properties.", tag="prop_placeholder", parent=panel)
            return
        import math as _m
        typ = self.selected_type
        ref = self.selected_ref
        display_name = (ref.label if typ == "vector" else getattr(ref, "name", "")) or typ.title()
        dpg.configure_item(panel, label=f"Properties: {typ.title()}: {display_name}")
        if dpg.does_item_exist("prop_placeholder"):
            dpg.delete_item("prop_placeholder")

        if typ in ("circle", "box"):
            shape = ref

            def on_name(s, a, u):
                setattr(shape, "name", a or "")
                self.hier_dirty = True
                dpg.configure_item(panel, label=f"Properties: {typ.title()}: {getattr(shape, 'name', '')}")

            def on_mass(s, a, u):
                try:
                    shape.body.mass = float(a)
                except Exception:
                    pass

            def on_angle_deg(s, a, u):
                try:
                    deg = float(a)
                    if getattr(self, 'angle_snap_enabled', False):
                        step = max(1e-9, float(self.angle_snap_deg))
                        deg = round(deg / step) * step
                    shape.body.angle = math.radians(deg)
                except Exception:
                    pass

            if not dpg.does_item_exist("prop_name"):
                dpg.add_input_text(label="Name", tag="prop_name", parent=panel, callback=on_name)
            if not dpg.does_item_exist("prop_mass"):
                dpg.add_input_float(label="Mass (kg)", tag="prop_mass", parent=panel, width=150, callback=on_mass)

            dpg.set_value("prop_name", getattr(shape, "name", ""))
            dpg.set_value("prop_mass", float(shape.body.mass))

            try:
                import pymunk as _pm
                if isinstance(shape, _pm.Circle):
                    if not dpg.does_item_exist("prop_radius"):
                        dpg.add_input_float(label="Radius (m)", tag="prop_radius", parent=panel, width=150,
                                            callback=lambda s,a,u: self._on_sel_radius(shape, a))
                    dpg.set_value("prop_radius", float(getattr(shape, 'radius', 0.5)))
                    # Angle readout for circle (read-only)
                    if not dpg.does_item_exist("prop_angle_ro"):
                        dpg.add_text("Angle (deg):", tag="prop_angle_ro_label", parent=panel)
                        dpg.add_text("0.0", tag="prop_angle_ro", parent=panel)
                elif isinstance(shape, _pm.Poly):
                    if not dpg.does_item_exist("prop_width"):
                        dpg.add_input_float(label="Width (m)", tag="prop_width", parent=panel, width=150,
                                            callback=lambda s,a,u: self._on_sel_box_w(shape, a))
                    if not dpg.does_item_exist("prop_height"):
                        dpg.add_input_float(label="Height (m)", tag="prop_height", parent=panel, width=150,
                                            callback=lambda s,a,u: self._on_sel_box_h(shape, a))
                    if not dpg.does_item_exist("prop_angle"):
                        dpg.add_input_float(label="Rotation (deg)", tag="prop_angle", parent=panel, width=150, callback=on_angle_deg)
                    bb = getattr(shape, 'bb', None)
                    if bb is not None:
                        dpg.set_value("prop_width", float(max(1e-6, bb.right - bb.left)))
                        dpg.set_value("prop_height", float(max(1e-6, bb.top - bb.bottom)))
            except Exception:
                pass

            # COM readout (live)
            if not dpg.does_item_exist("prop_com"):
                dpg.add_text("COM: (0,0)", tag="prop_com", parent=panel)

        elif typ == "surface":
            seg = ref

            def on_name(s, a, u):
                setattr(seg, "name", a or "")
                self.hier_dirty = True
                dpg.configure_item(panel, label=f"Properties: Surface: {getattr(seg, 'name', '')}")

            def on_sfr(s, a, u):
                try:
                    v = max(0.0, float(a))
                    setattr(seg, "static_friction", v)
                except Exception:
                    pass

            def on_dfr(s, a, u):
                try:
                    v = max(0.0, float(a))
                    seg.friction = v
                    setattr(seg, "dynamic_friction", v)
                except Exception:
                    pass

            if not dpg.does_item_exist("prop_s_name"):
                dpg.add_input_text(label="Name", tag="prop_s_name", parent=panel, callback=on_name)
            if not dpg.does_item_exist("prop_sfr"):
                dpg.add_input_float(label="Static Friction", tag="prop_sfr", parent=panel, width=150, callback=on_sfr)
            if not dpg.does_item_exist("prop_dfr"):
                dpg.add_input_float(label="Dynamic Friction", tag="prop_dfr", parent=panel, width=150, callback=on_dfr)
            if not dpg.does_item_exist("prop_selast"):
                dpg.add_input_float(label="Elasticity", tag="prop_selast", parent=panel, width=150, callback=lambda s,a,u: setattr(seg, 'elasticity', min(1.0, max(0.0, float(a)))) if a is not None else None)

            dpg.set_value("prop_s_name", getattr(seg, "name", ""))
            dpg.set_value("prop_sfr", float(getattr(seg, "static_friction", getattr(seg, "friction", 0.6))))
            dpg.set_value("prop_dfr", float(getattr(seg, "dynamic_friction", getattr(seg, "friction", 0.6))))
            dpg.set_value("prop_selast", float(getattr(seg, "elasticity", 0.3)))

        elif typ == "vector":
            V = ref

            def on_name(s, a, u):
                V.label = a or ""
                self.hier_dirty = True
                dpg.configure_item(panel, label=f"Properties: Vector: {V.label}")

            def on_mag(s, a, u):
                try:
                    mag = float(a)
                    ang = _m.degrees(_m.atan2(V.Fy, V.Fx))
                    V.x2 = V.x1 + mag * _m.cos(_m.radians(ang))
                    V.y2 = V.y1 + mag * _m.sin(_m.radians(ang))
                except Exception:
                    pass

            def on_ang(s, a, u):
                try:
                    ang = float(a)
                    mag = _m.hypot(V.Fx, V.Fy)
                    V.x2 = V.x1 + mag * _m.cos(_m.radians(ang))
                    V.y2 = V.y1 + mag * _m.sin(_m.radians(ang))
                except Exception:
                    pass

            if not dpg.does_item_exist("prop_v_name"):
                dpg.add_input_text(label="Name", tag="prop_v_name", parent=panel, callback=on_name)
            if not dpg.does_item_exist("prop_vec_mag"):
                dpg.add_input_float(label="Magnitude (N)", tag="prop_vec_mag", parent=panel, width=150, callback=on_mag)
            if not dpg.does_item_exist("prop_vec_ang"):
                dpg.add_input_float(label="Angle (deg)", tag="prop_vec_ang", parent=panel, width=150, callback=on_ang)

            dpg.set_value("prop_v_name", V.label or "")
            dpg.set_value("prop_vec_mag", float(_m.hypot(V.Fx, V.Fy)))
            dpg.set_value("prop_vec_ang", float(_m.degrees(_m.atan2(V.Fy, V.Fx))))

    # ---- Selection size update helpers ----
    def _on_sel_radius(self, shape, value):
        try:
            r = max(1e-6, float(value))
            new_sh = self.physics.update_circle(shape, float(shape.body.mass), r)
            if new_sh is not None:
                self.selected_ref = new_sh
        except Exception:
            pass

    def _on_sel_box_w(self, shape, value):
        try:
            w = max(1e-6, float(value))
            h = max(1e-6, float(dpg.get_value("prop_height")))
            new_sh = self.physics.update_box(shape, float(shape.body.mass), w, h)
            if new_sh is not None:
                self.selected_ref = new_sh
        except Exception:
            pass

    def _on_sel_box_h(self, shape, value):
        try:
            h = max(1e-6, float(value))
            w = max(1e-6, float(dpg.get_value("prop_width")))
            new_sh = self.physics.update_box(shape, float(shape.body.mass), w, h)
            if new_sh is not None:
                self.selected_ref = new_sh
        except Exception:
            pass

    # ---------- Undo/Redo & Deletion ----------
    def _snapshot_state(self):
        state = {
            'running': bool(self.physics.is_running),
            'dynamic': [],
            'static': [],
            'vectors': [],
            'counters': {'vector_counter': self.vector_counter},
        }
        import pymunk as _pm
        for sh in self.physics.dynamic:
            entry = {
                'name': getattr(sh, 'name', ''),
                'mass': float(sh.body.mass),
                'pos': (float(sh.body.position.x), float(sh.body.position.y)),
                'angle': float(sh.body.angle),
                'vel': (float(sh.body.velocity.x), float(sh.body.velocity.y)),
                'ang_vel': float(sh.body.angular_velocity),
            }
            if isinstance(sh, _pm.Circle):
                entry['type'] = 'circle'
                entry['radius'] = float(getattr(sh, 'radius', 0.5))
            else:
                entry['type'] = 'box'
                vs = getattr(sh, 'get_vertices', lambda: [])()
                if vs:
                    xs = [v.x for v in vs]
                    ys = [v.y for v in vs]
                    entry['width'] = float(max(xs) - min(xs))
                    entry['height'] = float(max(ys) - min(ys))
                else:
                    entry['width'] = 1.0
                    entry['height'] = 1.0
            state['dynamic'].append(entry)
        for sh in self.physics.static:
            if not hasattr(sh, 'a'):
                continue
            a_w = sh.body.local_to_world(sh.a)
            b_w = sh.body.local_to_world(sh.b)
            entry = {
                'name': getattr(sh, 'name', ''),
                'a': (float(a_w.x), float(a_w.y)),
                'b': (float(b_w.x), float(b_w.y)),
                'radius': float(getattr(sh, 'radius', 0.05)),
                'static_friction': float(getattr(sh, 'static_friction', getattr(sh, 'friction', 0.6))),
                'dynamic_friction': float(getattr(sh, 'dynamic_friction', getattr(sh, 'friction', 0.6))),
                'elasticity': float(getattr(sh, 'elasticity', 0.3)),
            }
            state['static'].append(entry)
        for V in self.physics.vectors:
            state['vectors'].append({
                'label': V.label,
                'x1': float(V.x1), 'y1': float(V.y1), 'x2': float(V.x2), 'y2': float(V.y2)
            })
        return state

    def _restore_state(self, state):
        try:
            for sh in list(self.physics.dynamic):
                self.physics.space.remove(sh, sh.body)
        except Exception:
            pass
        self.physics.dynamic = []
        try:
            for sh in list(self.physics.static):
                self.physics.space.remove(sh)
        except Exception:
            pass
        self.physics.static = []
        self.physics.vectors = []

        for d in state.get('dynamic', []):
            if d.get('type') == 'circle':
                s = self.physics.add_circle(r=d.get('radius', 0.5), pos=d.get('pos', (0, 0)), m=d.get('mass', 1.0))
            else:
                s = self.physics.add_box(w=d.get('width', 1.0), h=d.get('height', 1.0), pos=d.get('pos', (0, 0)), m=d.get('mass', 1.0))
            try:
                setattr(s, 'name', d.get('name', ''))
                s.body.angle = d.get('angle', 0.0)
                vx, vy = d.get('vel', (0.0, 0.0))
                s.body.velocity = (vx, vy)
                s.body.angular_velocity = d.get('ang_vel', 0.0)
            except Exception:
                pass
        for st in state.get('static', []):
            s = self.physics.add_surface(st.get('a', (0, 0)), st.get('b', (0, 0)), radius=st.get('radius', 0.05),
                                         static_mu=st.get('static_friction', 0.8), dynamic_mu=st.get('dynamic_friction', 0.6),
                                         elasticity=st.get('elasticity', 0.3))
            try:
                setattr(s, 'name', st.get('name', ''))
            except Exception:
                pass
        self.physics.vectors = [Vector(v['label'], v['x1'], v['y1'], v['x2'], v['y2']) for v in state.get('vectors', [])]
        self.vector_counter = max(self.vector_counter, state.get('counters', {}).get('vector_counter', self.vector_counter))
        self.physics.is_running = bool(state.get('running', True))
        self.selected_ref = None
        self.selected_type = None
        self.hier_dirty = True

    def _push_undo(self):
        try:
            self._undo_stack.append(self._snapshot_state())
            if len(self._undo_stack) > 100:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
        except Exception:
            pass

    def undo(self):
        if not self._undo_stack:
            return
        try:
            snap = self._undo_stack.pop()
            self._redo_stack.append(self._snapshot_state())
            self._restore_state(snap)
        except Exception:
            pass

    def redo(self):
        if not self._redo_stack:
            return
        try:
            snap = self._redo_stack.pop()
            self._undo_stack.append(self._snapshot_state())
            self._restore_state(snap)
        except Exception:
            pass

    def delete_selected(self):
        if not self.selected_ref or not self.selected_type:
            return
        self._push_undo()
        try:
            if self.selected_type in ('circle', 'box'):
                sh = self.selected_ref
                try:
                    self.physics.space.remove(sh, sh.body)
                except Exception:
                    pass
                self.physics.dynamic = [s for s in self.physics.dynamic if s is not sh]
            elif self.selected_type == 'surface':
                sh = self.selected_ref
                try:
                    self.physics.space.remove(sh)
                except Exception:
                    pass
                self.physics.static = [s for s in self.physics.static if s is not sh]
            elif self.selected_type == 'vector':
                V = self.selected_ref
                self.physics.vectors = [v for v in self.physics.vectors if v is not V]
        except Exception:
            pass
        self.selected_ref = None
        self.selected_type = None
        self.hier_dirty = True

    def restart_sim(self):
        try:
            self._restore_state(self._startup_snapshot)
        except Exception:
            pass

    # ---------- Tool Properties Panel ----------
    def _update_tool_panel(self):
        panel = "tool_panel"
        if not dpg.does_item_exist(panel):
            return
        try:
            dpg.delete_item(panel, children_only=True)
        except Exception:
            pass
        mode = self.mode
        if mode == "body":
            dpg.add_text("Body Tool", parent=panel)
            def on_body_type(s, a, u):
                self.body_tool_type = a
                self._save_settings()
            def on_body_mass(s, a, u):
                try:
                    self.body_default_mass = max(1e-6, float(a))
                except Exception:
                    pass
                self._save_settings()
            def on_box_w(s, a, u):
                try:
                    self.body_box_w = max(1e-6, float(a))
                except Exception:
                    pass
                self._save_settings()
            def on_box_h(s, a, u):
                try:
                    self.body_box_h = max(1e-6, float(a))
                except Exception:
                    pass
                self._save_settings()
            def on_circle_r(s, a, u):
                try:
                    self.body_circle_r = max(1e-6, float(a))
                except Exception:
                    pass
                self._save_settings()
            dpg.add_radio_button(["box", "circle"], default_value=self.body_tool_type, callback=on_body_type, parent=panel)
            dpg.add_input_float(label="Mass (kg)", default_value=float(self.body_default_mass), width=150, callback=on_body_mass, parent=panel)
            if self.body_tool_type == "circle":
                dpg.add_input_float(label="Radius (m)", default_value=float(self.body_circle_r), width=150, callback=on_circle_r, parent=panel)
            else:
                dpg.add_input_float(label="Width (m)", default_value=float(self.body_box_w), width=150, callback=on_box_w, parent=panel)
                dpg.add_input_float(label="Height (m)", default_value=float(self.body_box_h), width=150, callback=on_box_h, parent=panel)
        elif mode == "surface":
            dpg.add_text("Surface Tool", parent=panel)
            def on_radius(s, a, u):
                try:
                    self.surface_radius = max(0.0, float(a))
                except Exception:
                    pass
                self._save_settings()
            def on_sfr(s, a, u):
                try:
                    self.surface_static_mu = float(a)
                except Exception:
                    pass
                self._save_settings()
            def on_dfr(s, a, u):
                try:
                    self.surface_dynamic_mu = float(a)
                except Exception:
                    pass
                self._save_settings()
            def on_elast(s, a, u):
                try:
                    self.surface_elasticity = float(a)
                except Exception:
                    pass
                self._save_settings()
            dpg.add_input_float(label="Radius (m)", default_value=float(self.surface_radius), width=150, callback=on_radius, parent=panel)
            dpg.add_input_float(label="Static Friction", default_value=float(self.surface_static_mu), width=150, callback=on_sfr, parent=panel)
            dpg.add_input_float(label="Dynamic Friction", default_value=float(self.surface_dynamic_mu), width=150, callback=on_dfr, parent=panel)
            dpg.add_input_float(label="Elasticity", default_value=float(self.surface_elasticity), width=150, callback=on_elast, parent=panel)
        elif mode == "force":
            dpg.add_text("Force Tool", parent=panel)
            # Scope toggle between Global and Local drawing
            def _on_scope(s, a, u):
                self.on_force_scope_change(a)
            dpg.add_radio_button(["global", "local"], default_value=getattr(self, 'force_scope', 'global'), callback=_on_scope, parent=panel)
            if getattr(self, 'force_scope', 'global') == 'global':
                dpg.add_text("Global: draw from origin; applies to all bodies.", parent=panel)
            else:
                dpg.add_text("Local: click a body, then drag to set force.", parent=panel)
        elif mode == "fling":
            dpg.add_text("Fling Tool", parent=panel)
            def on_scale(s, a, u):
                try:
                    self.fling_scale = float(a)
                except Exception:
                    pass
                self._save_settings()
            def on_limit_toggle(s, a, u):
                self.fling_limit_speed = bool(a)
                self._save_settings()
            def on_max_speed(s, a, u):
                try:
                    self.fling_max_speed = max(0.0, float(a))
                except Exception:
                    pass
                self._save_settings()
            def on_snap_center(s, a, u):
                self.fling_snap_to_center = bool(a)
                self._save_settings()
            dpg.add_input_float(label="Strength Scale (J per m)", default_value=float(self.fling_scale), width=200, callback=on_scale, parent=panel)
            dpg.add_checkbox(label="Limit Max Speed", default_value=bool(self.fling_limit_speed), callback=on_limit_toggle, parent=panel)
            dpg.add_input_float(label="Max Speed (m/s)", default_value=float(self.fling_max_speed), width=150, callback=on_max_speed, parent=panel)
            dpg.add_checkbox(label="Snap to Center", default_value=bool(self.fling_snap_to_center), callback=on_snap_center, parent=panel)
        else:
            dpg.add_text("Select a tool to edit its defaults.", parent=panel)

        # Append Forces management only when in Force mode
        if mode == "force":
            try:
                build_forces_panel(self, panel)
            except Exception:
                pass

    # ---------- Settings ----------
    def _settings_path(self):
        return os.path.join(os.getcwd(), "settings.json")

    def _load_settings(self):
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.snap_step = float(data.get("snap_step", self.snap_step))
            gy = float(data.get("gravity_y", self.physics.space.gravity[1]))
            self.physics.space.gravity = (0, gy)
            self.body_tool_type = data.get("body_tool_type", self.body_tool_type)
            self.body_default_mass = float(data.get("body_default_mass", self.body_default_mass))
            self.body_box_w = float(data.get("body_box_w", self.body_box_w))
            self.body_box_h = float(data.get("body_box_h", self.body_box_h))
            self.body_circle_r = float(data.get("body_circle_r", self.body_circle_r))
            self.surface_radius = float(data.get("surface_radius", self.surface_radius))
            self.surface_static_mu = float(data.get("surface_static_mu", self.surface_static_mu))
            self.surface_dynamic_mu = float(data.get("surface_dynamic_mu", self.surface_dynamic_mu))
            self.surface_elasticity = float(data.get("surface_elasticity", self.surface_elasticity))
            # Fling tool
            self.fling_scale = float(data.get("fling_scale", self.fling_scale))
            self.fling_limit_speed = bool(data.get("fling_limit_speed", self.fling_limit_speed))
            self.fling_max_speed = float(data.get("fling_max_speed", self.fling_max_speed))
            self.fling_snap_to_center = bool(data.get("fling_snap_to_center", self.fling_snap_to_center))
            # Angle/Magnitude snap
            self.angle_snap_enabled = bool(data.get("angle_snap_enabled", self.angle_snap_enabled))
            self.angle_snap_deg = float(data.get("angle_snap_deg", self.angle_snap_deg))
            self.mag_snap_enabled = bool(data.get("mag_snap_enabled", self.mag_snap_enabled))
            self.mag_snap_step = float(data.get("mag_snap_step", self.mag_snap_step))
            # Clamp bounds
            if self.snap_step <= 0.0:
                self.snap = False
            else:
                self.snap = True
                self.snap_step = max(1e-6, self.snap_step)
            self.surface_radius = max(0.0, self.surface_radius)
            self.surface_static_mu = max(0.0, self.surface_static_mu)
            self.surface_dynamic_mu = max(0.0, self.surface_dynamic_mu)
            self.surface_elasticity = min(1.0, max(0.0, self.surface_elasticity))
        except Exception:
            pass

    def _save_settings(self):
        try:
            data = {
                "snap_step": float(self.snap_step),
                "gravity_y": float(self.physics.space.gravity[1]),
                "body_tool_type": self.body_tool_type,
                "body_default_mass": float(self.body_default_mass),
                "body_box_w": float(self.body_box_w),
                "body_box_h": float(self.body_box_h),
                "body_circle_r": float(self.body_circle_r),
                "surface_radius": float(self.surface_radius),
                "surface_static_mu": float(self.surface_static_mu),
                "surface_dynamic_mu": float(self.surface_dynamic_mu),
                "surface_elasticity": float(self.surface_elasticity),
                # Fling tool
                "fling_scale": float(self.fling_scale),
                "fling_limit_speed": bool(self.fling_limit_speed),
                "fling_max_speed": float(self.fling_max_speed),
                "fling_snap_to_center": bool(self.fling_snap_to_center),
                # Angle/Magnitude snap
                "angle_snap_enabled": bool(self.angle_snap_enabled),
                "angle_snap_deg": float(self.angle_snap_deg),
                "mag_snap_enabled": bool(self.mag_snap_enabled),
                "mag_snap_step": float(self.mag_snap_step),
            }
            with open(self._settings_path(), "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    # ---------- Main loop ----------
    def run(self):
        while dpg.is_dearpygui_running():
            now = time.time()
            dt = now - self._last
            self._last = now
            _dyn_before = len(self.physics.dynamic)
            self.physics.step(dt)
            if getattr(self.physics, 'last_pruned', 0) > 0 or len(self.physics.dynamic) != _dyn_before:
                self.hier_dirty = True
            self._animate_zoom(dt)
            self._render()
            try:
                d, st, ke = self.physics.stats()
                vecs = len(getattr(self.physics, 'vectors', []))
                lvecs = len(getattr(self.physics, 'local_forces', []))
                text = (
                    f"Bodies: {d}  |  Surfaces: {st}  |  Forces: {vecs}G/{lvecs}L  |  KE: {ke:.1f}"
                )
                if dpg.does_item_exist(self.stats):
                    dpg.set_value(self.stats, text)
                # Update run status indicator
                status_txt = f"Status: {'Running' if self.physics.is_running else 'Paused'}"
                if hasattr(self, 'run_status') and dpg.does_item_exist(self.run_status):
                    dpg.set_value(self.run_status, status_txt)
                    # Optional subtle color cue
                    try:
                        dpg.configure_item(self.run_status, color=(34,197,94,255) if self.physics.is_running else (239,68,68,255))
                    except Exception:
                        pass
            except Exception as e:
                print("[WARN] Stats update failed:", e)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


# ---------- MAIN ----------
if __name__ == "__main__":
    app = App()
    app.run()

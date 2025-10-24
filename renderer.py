# Base renderer

import math
import dearpygui.dearpygui as dpg
import pymunk

# ---------- Colors ----------
C_GRID = (200, 200, 210, 110)
C_GRID_MINOR = (210, 210, 220, 70)
C_AXIS = (40, 40, 48, 255)
C_CIRC = (59, 130, 246, 255)
C_BOX = (71, 85, 105, 255)
C_SURF = (110, 120, 140, 255)
C_VEL = (239, 68, 68, 255)
C_FORCE = (59, 130, 246, 255)
C_GHOST = (180, 180, 180, 220)
C_HILITE = (250, 204, 21, 220)
C_BOUNDS = (160, 160, 160, 120)
C_LABEL = (235, 235, 240, 255)


class Renderer:
    def __init__(self, tag: str, width: int, height: int):
        self.tag = tag
        self.w = width
        self.h = height
        self.cam = [0.0, 0.0]
        self.zoom = 80.0
        # World units per displayed user unit (meters)
        self.unit_scale = 1.0

    # ---------- Coordinate helpers ----------
    def _mid(self):
        return self.w // 2, self.h // 2

    def to_screen(self, x: float, y: float):
        cx, cy = self._mid()
        return cx + (x + self.cam[0]) * self.zoom, cy - (y + self.cam[1]) * self.zoom

    def to_world(self, sx: float, sy: float):
        cx, cy = self._mid()
        return (sx - cx) / self.zoom - self.cam[0], -(sy - cy) / self.zoom - self.cam[1]

    # ---------- Drawing ----------
    def clear(self):
        try:
            dpg.delete_item(self.tag, children_only=True)
        except Exception:
            pass

    def draw_grid(self, spacing: float = 1.0):
        """Draw a world-unit aligned grid.
        spacing: major grid spacing in world meters (default 1.0).
        Minor grid drawn at spacing/10 when visible.
        """
        # Visible world rect
        x0, y0 = self.to_world(0, self.h)
        x1, y1 = self.to_world(self.w, 0)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        px_per_world = max(1e-9, self.zoom)
        major = max(1e-9, float(spacing))
        minor = major / 10.0

        # Minor grid (0.1 m) when pixel spacing is adequate
        minor_px = minor * px_per_world
        if minor_px >= 6:
            kx0m = math.floor(x0 / minor)
            kx1m = math.ceil(x1 / minor)
            ky0m = math.floor(y0 / minor)
            ky1m = math.ceil(y1 / minor)
            for k in range(kx0m, kx1m + 1):
                # skip where coincides with major
                if (k % 10) == 0:
                    continue
                xw = k * minor
                sx, _ = self.to_screen(xw, 0)
                dpg.draw_line((sx, 0), (sx, self.h), color=C_GRID_MINOR, parent=self.tag)
            for k in range(ky0m, ky1m + 1):
                if (k % 10) == 0:
                    continue
                yw = k * minor
                _, sy = self.to_screen(0, yw)
                dpg.draw_line((0, sy), (self.w, sy), color=C_GRID_MINOR, parent=self.tag)

        # Major grid (1 m or spacing)
        kx0 = math.floor(x0 / major)
        kx1 = math.ceil(x1 / major)
        ky0 = math.floor(y0 / major)
        ky1 = math.ceil(y1 / major)
        t_major = 1 if (major * px_per_world) < 80 else 2
        for k in range(kx0, kx1 + 1):
            xw = k * major
            sx, _ = self.to_screen(xw, 0)
            dpg.draw_line((sx, 0), (sx, self.h), color=C_GRID, thickness=t_major, parent=self.tag)
        for k in range(ky0, ky1 + 1):
            yw = k * major
            _, sy = self.to_screen(0, yw)
            dpg.draw_line((0, sy), (self.w, sy), color=C_GRID, thickness=t_major, parent=self.tag)

        # Axes
        ax, ay = self.to_screen(0, 0)
        dpg.draw_line((0, ay), (self.w, ay), color=C_AXIS, thickness=2, parent=self.tag)
        dpg.draw_line((ax, 0), (ax, self.h), color=C_AXIS, thickness=2, parent=self.tag)

        # Origin label
        ox, oy = self.to_screen(0, 0)
        dpg.draw_text((int(ox) + 6, int(oy) - 18), "0, 0", color=C_LABEL, parent=self.tag, size=14)

        # Labels at major ticks in meters
        step_px = major * px_per_world
        min_label_px = 50.0
        label_every = max(1, int(math.ceil(min_label_px / max(1e-6, step_px))))
        # Determine formatting
        if major >= 1:
            decimals = 0
        else:
            decimals = min(6, max(0, int(math.ceil(-math.log10(major)))))
        fmt = "{:,." + str(decimals) + "f}"

        # Label positions
        y_label = ay + 2
        if y_label < 0 or y_label > self.h - 14:
            y_label = self.h - 16
        x_label = ax + 4
        if x_label < 0 or x_label > self.w - 40:
            x_label = 6

        for k in range(kx0, kx1 + 1):
            if k % label_every != 0:
                continue
            xw = k * major
            if abs(xw) <= 1e-9:
                continue
            sx, _ = self.to_screen(xw, 0)
            dpg.draw_text((int(sx) + 3, int(y_label)), fmt.format(xw), color=C_LABEL, parent=self.tag, size=14)
        for k in range(ky0, ky1 + 1):
            if k % label_every != 0:
                continue
            yw = k * major
            if abs(yw) <= 1e-9:
                continue
            _, sy = self.to_screen(0, yw)
            dpg.draw_text((int(x_label), int(sy) - 12), fmt.format(yw), color=C_LABEL, parent=self.tag, size=14)

    # ---------- Shape drawing ----------
    def draw_shape(self, s):
        import pymunk
        if isinstance(s, pymunk.Circle):
            self.draw_circle(s)
        elif isinstance(s, pymunk.Poly):
            self.draw_poly(s)
        elif isinstance(s, pymunk.Segment):
            #print("[Renderer] Drawing segment:", s.a, s.b)
            self.draw_segment(s)

    def draw_circle(self, s: pymunk.Circle):
        p = self.to_screen(s.body.position.x, s.body.position.y)
        dpg.draw_circle(p, s.radius * self.zoom, color=C_CIRC, fill=C_CIRC, parent=self.tag)

    def draw_poly(self, s: pymunk.Poly):
        vs = [self.to_screen(*s.body.local_to_world(v)) for v in s.get_vertices()]
        dpg.draw_polygon(vs, color=C_BOX, fill=C_BOX, parent=self.tag)

    def draw_segment(self, s: pymunk.Segment):
        a_world = s.body.local_to_world(s.a)
        b_world = s.body.local_to_world(s.b)
        a = self.to_screen(*a_world)
        b = self.to_screen(*b_world)
        thickness_screen = max(2, int(s.radius * 2 * self.zoom))
        dpg.draw_line(a, b, color=C_SURF, thickness=thickness_screen, parent=self.tag)

    def draw_velocity(self, s: pymunk.Shape):
        v = s.body.velocity
        if v.length < 1:
            return
        p0 = self.to_screen(s.body.position.x, s.body.position.y)
        p1 = self.to_screen(s.body.position.x + v.x * 0.05, s.body.position.y + v.y * 0.05)
        dpg.draw_arrow(p1, p0, color=C_VEL, thickness=2, size=8, parent=self.tag)

    def draw_vector(self, V, color=C_FORCE):
        p0s = self.to_screen(V.x1, V.y1)
        p1s = self.to_screen(V.x2, V.y2)
        dpg.draw_arrow(p1s, p0s, color=color, thickness=3, size=10, parent=self.tag)
        label = f"{V.label} {V.magnitude:.1f} N @ {V.angle_deg:.0f} deg"
        label_pos = (p1s[0] + 6, p1s[1] - 12)
        dpg.draw_text(label_pos, label, color=color, parent=self.tag, size=14)

    # ---------- Ghost previews ----------
    def ghost_line(self, p0, p1):
        dpg.draw_line(self.to_screen(*p0), self.to_screen(*p1), color=C_GHOST, thickness=2, parent=self.tag)

    def ghost_circle(self, center, r):
        dpg.draw_circle(self.to_screen(*center), r * self.zoom, color=C_GHOST, parent=self.tag)

    @staticmethod
    def draw_ghost_line(p1: tuple[float, float], p2: tuple[float, float]):
        """Draw a temporary, semi-transparent line for previews.
        Note: This uses a global drawlist/tag; not used by core renderer.
        """
        PREVIEW_LINE_TAG = "temp_wall_preview"
        DRAWLIST_TAG = "drawlist"

        # 1. Delete the previous ghost line if it exists
        if dpg.does_item_exist(PREVIEW_LINE_TAG):
            dpg.delete_item(PREVIEW_LINE_TAG)

        # 2. Draw the new semi-transparent ghost line
        # (Color: light cyan with 120 alpha for the 'ghost' effect)
        dpg.draw_line(
            p1,
            p2,
            color=(120, 255, 255, 120),
            thickness=2,
            parent=DRAWLIST_TAG,
            tag=PREVIEW_LINE_TAG
        )

    @staticmethod
    def clear_ghost_line():
        """Delete the ghost line after the final line is drawn."""
        PREVIEW_LINE_TAG = "temp_wall_preview"
        if dpg.does_item_exist(PREVIEW_LINE_TAG):
            dpg.delete_item(PREVIEW_LINE_TAG)

    # ---------- Highlights ----------
    def highlight_shape(self, s):
        import pymunk
        if isinstance(s, pymunk.Circle):
            p = self.to_screen(s.body.position.x, s.body.position.y)
            dpg.draw_circle(p, s.radius * self.zoom, color=C_HILITE, thickness=3, parent=self.tag)
        elif isinstance(s, pymunk.Poly):
            vs = [self.to_screen(*s.body.local_to_world(v)) for v in s.get_vertices()]
            dpg.draw_polygon(vs, color=C_HILITE, thickness=3, parent=self.tag)
        elif isinstance(s, pymunk.Segment):
            a_world = s.body.local_to_world(s.a)
            b_world = s.body.local_to_world(s.b)
            a = self.to_screen(*a_world)
            b = self.to_screen(*b_world)
            t = max(2, int(s.radius * 2 * self.zoom) + 2)
            dpg.draw_line(a, b, color=C_HILITE, thickness=t, parent=self.tag)

    def draw_shape_label(self, s):
        import pymunk
        base = getattr(s, 'name', None)
        if not base:
            if isinstance(s, pymunk.Segment):
                base = 'Surface'
            elif isinstance(s, pymunk.Poly):
                base = 'Box'
            elif isinstance(s, pymunk.Circle):
                base = 'Circle'
            else:
                base = 'Body'

        # Build info string
        info = ''
        try:
            if isinstance(s, pymunk.Segment):
                e = float(getattr(s, 'elasticity', 0.0))
                mu_s = float(getattr(s, 'static_friction', getattr(s, 'friction', 0.0)))
                mu_d = float(getattr(s, 'dynamic_friction', getattr(s, 'friction', 0.0)))
                info = f" e={e:.2f}  μs={mu_s:.2f}  μd={mu_d:.2f}"
            else:
                m = float(getattr(s.body, 'mass', 0.0))
                info = f" m={m:.2f} kg"
        except Exception:
            pass

        text = f"{base}{info}"

        # Compute top-right screen position
        try:
            if isinstance(s, pymunk.Segment):
                a_w = s.body.local_to_world(s.a)
                b_w = s.body.local_to_world(s.b)
                # Decide top-right vs top-left based on slope: if left end is higher than right end,
                # place the label at top-left; else top-right.
                left = a_w if a_w.x <= b_w.x else b_w
                right = b_w if left is a_w else a_w
                if left.y > right.y:
                    wx = min(a_w.x, b_w.x)
                    wy = max(a_w.y, b_w.y)
                    px, py = self.to_screen(wx, wy)
                    px, py = int(px) - 6, int(py) - 6
                else:
                    wx = max(a_w.x, b_w.x)
                    wy = max(a_w.y, b_w.y)
                    px, py = self.to_screen(wx, wy)
                    px, py = int(px) + 6, int(py) - 6
            elif isinstance(s, pymunk.Circle):
                cx, cy = s.body.position.x, s.body.position.y
                px, py = self.to_screen(cx + float(getattr(s, 'radius', 0.0)), cy + float(getattr(s, 'radius', 0.0)))
                px, py = int(px) + 6, int(py) - 6
            elif isinstance(s, pymunk.Poly):
                vs_local = s.get_vertices()
                # find max world x/y across vertices
                maxx, maxy = -1e18, -1e18
                for v in vs_local:
                    w = s.body.local_to_world(v)
                    if w.x > maxx:
                        maxx = w.x
                    if w.y > maxy:
                        maxy = w.y
                px, py = self.to_screen(maxx, maxy)
                px, py = int(px) + 6, int(py) - 6
            else:
                px, py = self.to_screen(s.body.position.x, s.body.position.y)
                px, py = int(px) + 6, int(py) - 6
            # Draw text at computed location
            dpg.draw_text((px, py), text, color=C_LABEL, parent=self.tag, size=14)
        except Exception:
            pass

    def ghost_box(self, center, w, h):
        cx, cy = center
        hw = float(w) * 0.5
        hh = float(h) * 0.5
        pmin = (cx - hw, cy - hh)
        pmax = (cx + hw, cy + hh)
        pmin_s = self.to_screen(*pmin)
        pmax_s = self.to_screen(*pmax)
        # draw as rectangle outline
        try:
            dpg.draw_rectangle(pmin_s, pmax_s, color=C_GHOST, parent=self.tag)
        except Exception:
            # Fallback: draw polygon
            a = self.to_screen(cx - hw, cy - hh)
            b = self.to_screen(cx + hw, cy - hh)
            c = self.to_screen(cx + hw, cy + hh)
            d = self.to_screen(cx - hw, cy + hh)
            dpg.draw_polygon([a, b, c, d], color=C_GHOST, parent=self.tag)

    def draw_bounds(self, limit: float, color=C_BOUNDS):
        """Draw a square bounding box from -limit..+limit in world coords."""
        L = float(limit)
        # corners in world
        p1 = (-L, -L)
        p2 = ( L, -L)
        p3 = ( L,  L)
        p4 = (-L,  L)
        # draw edges
        dpg.draw_line(self.to_screen(*p1), self.to_screen(*p2), color=color, parent=self.tag)
        dpg.draw_line(self.to_screen(*p2), self.to_screen(*p3), color=color, parent=self.tag)
        dpg.draw_line(self.to_screen(*p3), self.to_screen(*p4), color=color, parent=self.tag)
        dpg.draw_line(self.to_screen(*p4), self.to_screen(*p1), color=color, parent=self.tag)

    # ---------- Dotted helpers ----------
    def _draw_dotted_line_screen(self, p0s, p1s, color, dash=6, gap=6):
        x0, y0 = p0s
        x1, y1 = p1s
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx*dx + dy*dy) ** 0.5
        if dist <= 1e-6:
            return
        ux = dx / dist
        uy = dy / dist
        t = 0.0
        while t < dist:
            t_end = min(dist, t + dash)
            sx = x0 + ux * t
            sy = y0 + uy * t
            ex = x0 + ux * t_end
            ey = y0 + uy * t_end
            dpg.draw_line((sx, sy), (ex, ey), color=color, parent=self.tag)
            t += dash + gap

    def draw_dotted_world(self, p0, p1, color=(59, 130, 246, 180), dash=6, gap=6):
        p0s = self.to_screen(*p0)
        p1s = self.to_screen(*p1)
        self._draw_dotted_line_screen(p0s, p1s, color, dash, gap)





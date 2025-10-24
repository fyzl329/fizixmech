# Physics backend

import pymunk
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------- Config ----------
CANVAS_W, CANVAS_H = 1000, 850
MIN_MAGNITUDE = 0.01

# ---------- Vector model ----------
@dataclass
class Vector:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    def _dxdy(self) -> Tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def magnitude(self) -> float:
        dx, dy = self._dxdy()
        return (dx**2 + dy**2)**0.5

    @property
    def angle_deg(self) -> float:
        dx, dy = self._dxdy()
        if abs(dx) + abs(dy) < 1e-12:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    @property
    def Fx(self) -> float:
        dx, _ = self._dxdy()
        return dx

    @property
    def Fy(self) -> float:
        _, dy = self._dxdy()
        return dy


# ---------- Physics simulation ----------
class Physics:
    def __init__(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, -9.81)
        # Use no global damping for more ideal Newtonian behavior by default
        self.space.damping = 1.0
        self.space.sleep_time_threshold = 1.2
        self.space.idle_speed_threshold = 0.2
        self.space.iterations = 40

        self.dynamic: List[pymunk.Shape] = []
        self.static: List[pymunk.Shape] = []
        # Global forces (act on all dynamic bodies each substep)
        self.vectors: List[Vector] = []
        # Local forces: attached to a specific body, applied each substep
        self.local_forces: List["LocalForce"] = []
        self.is_running = True
        self.bounds_limit = 1000.0
        self.last_pruned = 0
        self._accum = 0.0

    # ----- Impulses & helpers -----
    def wake(self, body: pymunk.Body) -> None:
        try:
            body.activate()
        except Exception:
            pass

    def clamp_body_speed(self, body: pymunk.Body, max_speed: float) -> None:
        try:
            v = body.velocity
            ms = float(max_speed)
            if ms <= 0:
                return
            if v.length > ms:
                body.velocity = v.normalized() * ms
        except Exception:
            pass

    def apply_impulse(self, body: pymunk.Body, impulse: tuple[float, float], point_world: tuple[float, float] | None = None) -> None:
        """Apply an impulse to a body at a world point (adds spin if off-center)."""
        if body is None:
            return
        try:
            self.wake(body)
            if point_world is None:
                point_world = (body.position.x, body.position.y)
            body.apply_impulse_at_world_point(impulse, point_world)
        except Exception:
            pass

    # ----- Object creation -----
    def add_circle(self, r: float = 0.5, pos: Tuple[float, float] = (0, 2.0), m: float = 1.0, friction: float = 0.8, elasticity: float = 0.9):
        I = pymunk.moment_for_circle(m, 0, r)
        b = pymunk.Body(m, I)
        b.position = pos
        s = pymunk.Circle(b, r)
        s.elasticity = float(elasticity)
        s.friction = float(friction)
        self.space.add(b, s)
        self.dynamic.append(s)
        return s

    def add_box(self, w: float = 1.0, h: float = 1.0, pos: Tuple[float, float] = (0, 2.0), m: float = 1.0, friction: float = 0.8, elasticity: float = 0.9):
        I = pymunk.moment_for_box(m, (w, h))
        b = pymunk.Body(m, I)
        b.position = pos
        s = pymunk.Poly.create_box(b, (w, h))
        # store logical dimensions to allow correct editing after rotation
        try:
            setattr(s, 'box_w', float(w))
            setattr(s, 'box_h', float(h))
        except Exception:
            pass
        s.elasticity = float(elasticity)
        s.friction = float(friction)
        self.space.add(b, s)
        self.dynamic.append(s)
        return s

    def add_surface(self, p1, p2, radius: float = 0.05, static_mu: float = 0.8, dynamic_mu: float = 0.6,
                    elasticity: float = 0.3):
        radius = max(0.0, float(radius))
        static_mu = max(0.0, float(static_mu))
        dynamic_mu = max(0.0, float(dynamic_mu))
        elasticity = min(1.0, max(0.0, float(elasticity)))
        b = pymunk.Body(body_type=pymunk.Body.STATIC)
        s = pymunk.Segment(b, p1, p2, radius)
        # Keep kinetic (dynamic) friction realistically lower than static
        s.friction = dynamic_mu
        s.elasticity = float(elasticity)
        # Store both as metadata for the editor/UI
        try:
            setattr(s, 'static_friction', static_mu)
            setattr(s, 'dynamic_friction', dynamic_mu)
        except Exception:
            pass
        self.space.add(b, s)
        self.static.append(s)
        return s

    # ----- Picking helpers -----
    def _dist_point_to_seg(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        """Distance from point P to segment AB in world units."""
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            # A and B are the same point
            dx, dy = px - ax, py - ay
            return (dx * dx + dy * dy) ** 0.5
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        cx, cy = ax + t * vx, ay + t * vy
        dx, dy = px - cx, py - cy
        return (dx * dx + dy * dy) ** 0.5

    def pick_surface(self, p, tol: float = 12.0) -> Optional[pymunk.Segment]:
        """Pick nearest static segment within tolerance (world coords)."""
        best, dmin = None, 1e9
        for s in self.static:
            if not isinstance(s, pymunk.Segment):
                continue
            a_w = s.body.local_to_world(s.a)
            b_w = s.body.local_to_world(s.b)
            d = self._dist_point_to_seg(p.x, p.y, a_w.x, a_w.y, b_w.x, b_w.y)
            if d < tol and d < dmin:
                best, dmin = s, d
        return best

    def pick_vector(self, p, tol: float = 12.0) -> Optional[Vector]:
        """Pick nearest global force vector within tolerance (world coords)."""
        best, dmin = None, 1e9
        for V in self.vectors:
            d = self._dist_point_to_seg(p.x, p.y, V.x1, V.y1, V.x2, V.y2)
            if d < tol and d < dmin:
                best, dmin = V, d
        return best

    # ----- Update helpers -----
    def update_circle(self, shape: pymunk.Circle, mass: float, radius: float) -> Optional[pymunk.Circle]:
        pos = shape.body.position
        vel = shape.body.velocity
        ang = shape.body.angle
        ang_vel = shape.body.angular_velocity
        elast = shape.elasticity
        fric = shape.friction
        name = getattr(shape, 'name', None)
        try:
            self.space.remove(shape, shape.body)
        except Exception:
            pass
        I = pymunk.moment_for_circle(mass, 0, radius)
        b = pymunk.Body(mass, I)
        b.position = pos
        b.velocity = vel
        b.angle = ang
        b.angular_velocity = ang_vel
        s = pymunk.Circle(b, radius)
        s.elasticity = elast
        s.friction = fric
        if name is not None:
            try:
                setattr(s, 'name', name)
            except Exception:
                pass
        self.space.add(b, s)
        # replace in dynamic list
        for i, sh in enumerate(self.dynamic):
            if sh is shape:
                self.dynamic[i] = s
                break
        else:
            self.dynamic.append(s)
        return s

    def update_box(self, shape: pymunk.Poly, mass: float, width: float, height: float) -> Optional[pymunk.Poly]:
        pos = shape.body.position
        vel = shape.body.velocity
        ang = shape.body.angle
        ang_vel = shape.body.angular_velocity
        elast = shape.elasticity
        fric = shape.friction
        name = getattr(shape, 'name', None)
        try:
            self.space.remove(shape, shape.body)
        except Exception:
            pass
        I = pymunk.moment_for_box(mass, (width, height))
        b = pymunk.Body(mass, I)
        b.position = pos
        b.velocity = vel
        b.angle = ang
        b.angular_velocity = ang_vel
        s = pymunk.Poly.create_box(b, (width, height))
        try:
            setattr(s, 'box_w', float(width))
            setattr(s, 'box_h', float(height))
        except Exception:
            pass
        s.elasticity = elast
        s.friction = fric
        if name is not None:
            try:
                setattr(s, 'name', name)
            except Exception:
                pass
        self.space.add(b, s)
        for i, sh in enumerate(self.dynamic):
            if sh is shape:
                self.dynamic[i] = s
                break
        else:
            self.dynamic.append(s)
        return s

    def update_surface(self, seg: pymunk.Segment, length: float, angle_deg: float,
                       static_friction: float, dynamic_friction: float,
                       radius: Optional[float] = None) -> pymunk.Segment:
        static_friction = max(0.0, float(static_friction))
        dynamic_friction = max(0.0, float(dynamic_friction))
        # compute center in world, then new endpoints
        a_w = seg.body.local_to_world(seg.a)
        b_w = seg.body.local_to_world(seg.b)
        cx, cy = (a_w.x + b_w.x) * 0.5, (a_w.y + b_w.y) * 0.5
        half = max(0.0, length) * 0.5
        rad = math.radians(angle_deg)
        dx = half * math.cos(rad)
        dy = half * math.sin(rad)
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        # convert to local coords of the segment's body
        p1_local = seg.body.world_to_local(p1)
        p2_local = seg.body.world_to_local(p2)
        # Recreate segment shape (endpoints are read-only in pymunk)
        body = seg.body
        elast = seg.elasticity
        radius = seg.radius if radius is None else max(0.0, float(radius))
        name = getattr(seg, 'name', None)
        try:
            self.space.remove(seg)
        except Exception:
            pass
        s2 = pymunk.Segment(body, p1_local, p2_local, radius)
        s2.friction = dynamic_friction
        s2.elasticity = elast
        self.space.add(s2)
        # replace in static list
        for i, sh in enumerate(self.static):
            if sh is seg:
                self.static[i] = s2
                break
        else:
            self.static.append(s2)
        # optional metadata for separate static/dynamic friction
        try:
            setattr(s2, 'static_friction', static_friction)
            setattr(s2, 'dynamic_friction', dynamic_friction)
            if name is not None:
                setattr(s2, 'name', name)
        except Exception:
            pass
        return s2

    # ----- Forces -----
    def apply_vectors(self) -> None:
        """Apply all global vectors to every dynamic body."""
        if not self.vectors:
            pass
        else:
            for V in self.vectors:
                fx, fy = V.Fx, V.Fy
                if abs(fx) < 1e-6 and abs(fy) < 1e-6:
                    continue
                for s in self.dynamic:
                    try:
                        s.body.activate()
                    except Exception:
                        pass
                    s.body.apply_force_at_world_point((fx, fy), s.body.position)
        # Apply local forces
        if self.local_forces:
            # Build lookup from body id to body
            bodies = {id(s.body): s.body for s in self.dynamic}
            keep: List[LocalForce] = []
            for lf in list(self.local_forces):
                body = bodies.get(lf.body_id)
                if body is None:
                    # drop if body no longer exists
                    continue
                try:
                    body.activate()
                except Exception:
                    pass
                # Resolve direction
                angle_rad = math.radians(lf.angle_deg)
                if lf.mode == 'body':
                    angle_rad = body.angle + angle_rad
                fx = float(lf.magnitude) * math.cos(angle_rad)
                fy = float(lf.magnitude) * math.sin(angle_rad)
                body.apply_force_at_world_point((fx, fy), body.position)
                keep.append(lf)
            self.local_forces = keep

    # ----- Simulation -----
    def step(self, dt: float) -> None:
        """Advance the simulation with a bounded fixed timestep accumulator.

        - Uses a smaller fixed step (`h`) for smoother integration.
        - Caps the number of substeps to avoid spiral-of-death on slow frames.
        - Discards excessive accumulated time to keep the sim responsive.
        """
        if not self.is_running:
            return
        # Clamp very large frame jumps to avoid huge catches
        dt = float(max(0.0, min(float(dt), 0.25)))
        self._accum += dt
        h = 1.0 / 240.0  # smaller step for smoother motion
        max_steps = 24   # allow up to ~0.1s of catch-up per frame
        steps = 0
        while self._accum >= h and steps < max_steps:
            # Global vectors behave like continuous forces; apply each substep
            self.apply_vectors()
            self.space.step(h)
            self._accum -= h
            steps += 1
        # If we still have a lot accumulated (very slow frame), drop the excess
        # to keep the simulation stable and reduce visible jitter.
        if steps >= max_steps and self._accum > h * 2:
            self._accum = 0.0
        # prune bodies that drift outside the bounding box
        self.last_pruned = self._prune_out_of_bounds()

    # ----- Culling -----
    def _prune_out_of_bounds(self) -> int:
        """Remove dynamic shapes whose body position lies outside +/-bounds_limit.
        Returns the number of shapes removed.
        """
        limit = float(self.bounds_limit)
        removed = 0
        keep: List[pymunk.Shape] = []
        for s in list(self.dynamic):
            pos = s.body.position
            if abs(pos.x) > limit or abs(pos.y) > limit:
                try:
                    self.space.remove(s, s.body)
                except Exception:
                    pass
                removed += 1
            else:
                keep.append(s)
        if removed:
            self.dynamic = keep
        return removed

    # ----- Utilities -----
    def pick_body(self, p, r: float = 35) -> Optional[pymunk.Body]:
        """Pick a dynamic body near point p (world coords)."""
        best, dmin = None, 1e9
        for s in self.dynamic:
            d = (s.body.position - p).length
            if isinstance(s, pymunk.Circle):
                radius = s.radius
            else:
                radius = max(s.bb.top - s.bb.bottom, s.bb.right - s.bb.left) / 2
            if d < radius * 1.5 and d < dmin:
                best, dmin = s.body, d
        return best

    def stats(self):
        ke = sum(0.5 * s.body.mass * (s.body.velocity.length ** 2) for s in self.dynamic)
        return len(self.dynamic), len(self.static), ke


# ---------- LOCAL FORCE MODEL ----------
@dataclass
class LocalForce:
    label: str
    body_id: int  # id(body)
    magnitude: float
    angle_deg: float  # direction relative to mode
    mode: str  # 'body' or 'world'

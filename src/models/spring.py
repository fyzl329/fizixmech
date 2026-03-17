"""Spring model for physics simulation."""

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..physics import Physics


@dataclass
class Spring:
    """Represents a spring connecting two bodies or anchors."""
    
    label: str
    a_id: Optional[int] = None
    b_id: Optional[int] = None
    anchor_a: Optional[Tuple[float, float]] = None
    anchor_b: Optional[Tuple[float, float]] = None
    rest_length: float = 1.0
    stiffness: float = 25.0
    damping: float = 0.5
    max_extension_factor: float = 2.0

    def get_bodies(self, physics: "Physics"):
        """Get the two bodies connected by this spring."""
        a = None
        b = None
        try:
            shapes = list(physics.dynamic) + list(physics.static)
        except Exception:
            shapes = []
        for sh in shapes:
            try:
                body = sh.body
            except Exception:
                continue
            if self.a_id is not None and id(body) == self.a_id:
                a = body
            if self.b_id is not None and id(body) == self.b_id:
                b = body
        return a, b

    def get_endpoints(self, physics: "Physics"):
        """Get the world-space endpoints of the spring."""
        a_body, b_body = self.get_bodies(physics)
        if a_body is not None:
            ax, ay = a_body.position.x, a_body.position.y
        else:
            ax, ay = self.anchor_a if self.anchor_a is not None else (0.0, 0.0)
        if b_body is not None:
            bx, by = b_body.position.x, b_body.position.y
        else:
            bx, by = self.anchor_b if self.anchor_b is not None else (0.0, 0.0)
        return (ax, ay), (bx, by), a_body, b_body

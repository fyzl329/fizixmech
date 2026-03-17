"""Vector model for force representation."""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Vector:
    """Represents a force vector with start and end points."""
    
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    def _dxdy(self) -> Tuple[float, float]:
        """Return the delta x and y components."""
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def magnitude(self) -> float:
        """Return the magnitude (length) of the vector."""
        dx, dy = self._dxdy()
        return (dx**2 + dy**2)**0.5

    @property
    def angle_deg(self) -> float:
        """Return the angle in degrees."""
        dx, dy = self._dxdy()
        if abs(dx) + abs(dy) < 1e-12:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    @property
    def Fx(self) -> float:
        """Return the x-component of the force."""
        dx, _ = self._dxdy()
        return dx

    @property
    def Fy(self) -> float:
        """Return the y-component of the force."""
        _, dy = self._dxdy()
        return dy

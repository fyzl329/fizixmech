"""Local force model for body-specific forces."""

from dataclasses import dataclass


@dataclass
class LocalForce:
    """Represents a force applied to a specific body."""
    
    label: str
    body_id: int  # id(body)
    magnitude: float
    angle_deg: float  # direction relative to mode
    mode: str  # 'body' or 'world'

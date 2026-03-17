"""
Preset objects and templates for Fizix Mech.

Quick-add templates for common physics setups.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PresetObject:
    """Represents a single object in a preset."""
    type: str  # 'circle', 'box', 'surface'
    params: Dict[str, Any]
    position: Tuple[float, float]


@dataclass
class Preset:
    """A complete preset setup."""
    name: str
    description: str
    category: str  # 'basic', 'mechanism', 'vehicle', 'experiment'
    objects: List[PresetObject]
    initial_velocity: Tuple[float, float] = (0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESET_BALL_DROP = Preset(
    name="Ball Drop",
    description="Simple ball drop experiment",
    category="basic",
    objects=[
        PresetObject('circle', {'r': 0.3, 'm': 1.0}, (0, 5)),
        PresetObject('surface', {'length': 10, 'angle': 0}, (0, 0)),
    ]
)

PRESET_STACK = Preset(
    name="Block Stack",
    description="Stack of blocks to knock over",
    category="basic",
    objects=[
        PresetObject('box', {'w': 1.0, 'h': 1.0, 'm': 1.0}, (0, 0.5)),
        PresetObject('box', {'w': 1.0, 'h': 1.0, 'm': 1.0}, (0, 1.5)),
        PresetObject('box', {'w': 1.0, 'h': 1.0, 'm': 1.0}, (0, 2.5)),
        PresetObject('surface', {'length': 8, 'angle': 0}, (0, 0)),
    ]
)

PRESET_RAMP = Preset(
    name="Ramp Roll",
    description="Ball rolling down a ramp",
    category="basic",
    objects=[
        PresetObject('circle', {'r': 0.25, 'm': 0.5}, (-3, 3)),
        PresetObject('surface', {'length': 6, 'angle': -30}, (0, 1.5)),
        PresetObject('surface', {'length': 4, 'angle': 0}, (3, 0)),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESET_PENDULUM = Preset(
    name="Pendulum",
    description="Simple pendulum",
    category="mechanism",
    objects=[
        PresetObject('circle', {'r': 0.3, 'm': 2.0}, (0, -3)),
        PresetObject('surface', {'length': 2, 'angle': 0}, (0, 0)),
    ]
)

PRESET_DOMINO = Preset(
    name="Domino Chain",
    description="Chain of dominoes",
    category="mechanism",
    objects=[
        *[PresetObject('box', {'w': 0.2, 'h': 1.0, 'm': 0.5}, (i * 0.8, 0.5)) 
          for i in range(-4, 5)],
        PresetObject('surface', {'length': 12, 'angle': 0}, (0, 0)),
    ]
)

PRESET_SEESAW = Preset(
    name="Seesaw",
    description="Balance beam with weights",
    category="mechanism",
    objects=[
        PresetObject('box', {'w': 4.0, 'h': 0.2, 'm': 5.0}, (0, 1)),
        PresetObject('circle', {'r': 0.3, 'm': 2.0}, (-1.5, 1.5)),
        PresetObject('circle', {'r': 0.3, 'm': 2.0}, (1.5, 1.5)),
        PresetObject('surface', {'length': 1, 'angle': 90}, (0, 0.5)),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESET_CAR = Preset(
    name="Simple Car",
    description="Basic two-wheel vehicle",
    category="vehicle",
    objects=[
        PresetObject('box', {'w': 2.0, 'h': 0.5, 'm': 5.0}, (0, 1.5)),
        PresetObject('circle', {'r': 0.4, 'm': 1.0}, (-0.7, 0.8)),
        PresetObject('circle', {'r': 0.4, 'm': 1.0}, (0.7, 0.8)),
        PresetObject('surface', {'length': 15, 'angle': 0}, (0, 0)),
    ]
)

PRESET_ROCKET = Preset(
    name="Rocket",
    description="Rocket shape (visual only)",
    category="vehicle",
    objects=[
        PresetObject('box', {'w': 0.5, 'h': 2.0, 'm': 3.0}, (0, 3)),
        PresetObject('circle', {'r': 0.15, 'm': 0.2}, (0, 4.2)),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESET_COLLISION = Preset(
    name="Collision Test",
    description="Two balls colliding",
    category="experiment",
    objects=[
        PresetObject('circle', {'r': 0.4, 'm': 1.0, 'elasticity': 0.9}, (-2, 1)),
        PresetObject('circle', {'r': 0.4, 'm': 1.0, 'elasticity': 0.9}, (2, 1)),
        PresetObject('surface', {'length': 10, 'angle': 0}, (0, 0)),
    ]
)

PRESET_PROJECTILE = Preset(
    name="Projectile Motion",
    description="Platform for projectile experiment",
    category="experiment",
    objects=[
        PresetObject('circle', {'r': 0.3, 'm': 0.5}, (-4, 4)),
        PresetObject('surface', {'length': 2, 'angle': 0}, (-4, 3)),
        PresetObject('surface', {'length': 10, 'angle': 0}, (0, 0)),
    ]
)

PRESET_DOUBLE_PENDULUM = Preset(
    name="Double Pendulum",
    description="Chaotic double pendulum (requires springs)",
    category="experiment",
    objects=[
        PresetObject('circle', {'r': 0.3, 'm': 1.0}, (0, -1.5)),
        PresetObject('circle', {'r': 0.3, 'm': 1.0}, (0, -3)),
        PresetObject('surface', {'length': 2, 'angle': 0}, (0, 0)),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# PRESET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ALL_PRESETS: List[Preset] = [
    PRESET_BALL_DROP,
    PRESET_STACK,
    PRESET_RAMP,
    PRESET_PENDULUM,
    PRESET_DOMINO,
    PRESET_SEESAW,
    PRESET_CAR,
    PRESET_ROCKET,
    PRESET_COLLISION,
    PRESET_PROJECTILE,
    PRESET_DOUBLE_PENDULUM,
]

PRESETS_BY_CATEGORY: Dict[str, List[Preset]] = {}
for preset in ALL_PRESETS:
    if preset.category not in PRESETS_BY_CATEGORY:
        PRESETS_BY_CATEGORY[preset.category] = []
    PRESETS_BY_CATEGORY[preset.category].append(preset)


def get_preset_by_name(name: str) -> Preset | None:
    """Get a preset by its name."""
    for preset in ALL_PRESETS:
        if preset.name.lower() == name.lower():
            return preset
    return None


def apply_preset(app, preset: Preset) -> None:
    """Apply a preset to the current simulation."""
    app._push_undo()
    
    for obj in preset.objects:
        if obj.type == 'circle':
            shape = app.physics.add_circle(
                r=obj.params.get('r', 0.5),
                pos=(obj.position[0], obj.position[1]),
                m=obj.params.get('m', 1.0),
                elasticity=obj.params.get('elasticity', 0.9),
            )
        elif obj.type == 'box':
            shape = app.physics.add_box(
                w=obj.params.get('w', 1.0),
                h=obj.params.get('h', 1.0),
                pos=(obj.position[0], obj.position[1]),
                m=obj.params.get('m', 1.0),
                elasticity=obj.params.get('elasticity', 0.9),
            )
        elif obj.type == 'surface':
            length = obj.params.get('length', 4.0)
            angle = obj.params.get('angle', 0)
            # Calculate endpoints from center, length, and angle
            import math
            half = length / 2
            rad = math.radians(angle)
            dx = half * math.cos(rad)
            dy = half * math.sin(rad)
            p1 = (obj.position[0] - dx, obj.position[1] - dy)
            p2 = (obj.position[0] + dx, obj.position[1] + dy)
            shape = app.physics.add_surface(p1, p2)
        
        # Set name
        try:
            setattr(shape, 'name', f"{preset.name}: {obj.type}")
        except Exception:
            pass
    
    app.hier_dirty = True

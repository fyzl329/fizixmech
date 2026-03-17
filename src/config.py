"""
Configuration constants for Fizix Mech.

All magic numbers and configurable values are centralized here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME: str = "Fizix Mech"
APP_VERSION: str = "1.0.2"
APP_TITLE: str = "FZXMCH - Professional Physics Sandbox"

# ─────────────────────────────────────────────────────────────────────────────
# WINDOW & VIEWPORT
# ─────────────────────────────────────────────────────────────────────────────
WIN_WIDTH: int = 1400
WIN_HEIGHT: int = 900
CANVAS_WIDTH: int = 1000
CANVAS_HEIGHT: int = 850
PANEL_WIDTH: int = 360
TOOLBAR_HEIGHT: int = 40
STATUSBAR_HEIGHT: int = 30

# ─────────────────────────────────────────────────────────────────────────────
# ZOOM & CAMERA
# ─────────────────────────────────────────────────────────────────────────────
ZOOM_FACTOR: float = 0.5
ZOOM_MIN: float = 20.0
ZOOM_MAX: float = 400.0
DEFAULT_ZOOM: float = 80.0
ZOOM_HALFLIFE: float = 0.12  # Seconds for zoom to reach 50% of target

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CLICK_THRESHOLD: float = 20.0  # Pixels to distinguish click vs drag
MIN_MAGNITUDE: float = 0.01  # Minimum force magnitude to apply
HOVER_THRESHOLD: float = 5.0  # Pixels for hover detection

# ─────────────────────────────────────────────────────────────────────────────
# SNAPPING DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
SNAP_STEP: float = 1.0  # Grid snap step in meters
SNAP_ENABLED: bool = True
ANGLE_SNAP_ENABLED: bool = False
ANGLE_SNAP_DEG: float = 15.0
MAGNITUDE_SNAP_ENABLED: bool = False
MAGNITUDE_SNAP_STEP: float = 0.25

# Advanced snapping
ENDPOINT_SNAP: bool = True
MIDPOINT_SNAP: bool = True
PERPENDICULAR_SNAP: bool = False
CENTER_SNAP: bool = True
SNAP_RADIUS: float = 10.0  # Screen pixels

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
PHYSICS_SUBSTEP: float = 1.0 / 240.0  # Fixed timestep for physics integration
PHYSICS_MAX_STEPS: int = 24  # Max substeps per frame to prevent spiral of death
PHYSICS_DT_MAX: float = 0.25  # Max delta time to clamp
GRAVITY_DEFAULT: float = -9.81  # m/s²
BOUNDS_LIMIT: float = 1000.0  # World units before culling

# ─────────────────────────────────────────────────────────────────────────────
# BODY DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
BODY_TOOL_TYPE: str = "box"  # 'box' or 'circle'
BODY_DEFAULT_MASS: float = 1.0
BODY_BOX_WIDTH: float = 1.0
BODY_BOX_HEIGHT: float = 1.0
BODY_CIRCLE_RADIUS: float = 0.5
BODY_DEFAULT_FRICTION: float = 0.8
BODY_DEFAULT_ELASTICITY: float = 0.9

# ─────────────────────────────────────────────────────────────────────────────
# SURFACE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_RADIUS: float = 0.05
SURFACE_STATIC_MU: float = 0.8
SURFACE_DYNAMIC_MU: float = 0.6
SURFACE_ELASTICITY: float = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# SPRING DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
SPRING_DEFAULT_K: float = 25.0  # Stiffness
SPRING_DEFAULT_DAMPING: float = 0.6
SPRING_MAX_EXTENSION_FACTOR: float = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# FLING TOOL
# ─────────────────────────────────────────────────────────────────────────────
FLING_SCALE: float = 1.0  # Impulse per world unit drag (N·s/m)
FLING_LIMIT_SPEED: bool = True
FLING_MAX_SPEED: float = 50.0
FLING_SNAP_TO_CENTER: bool = False
FLING_COOLDOWN: float = 0.10  # Seconds between flings

# ─────────────────────────────────────────────────────────────────────────────
# UNDO/REDO
# ─────────────────────────────────────────────────────────────────────────────
UNDO_MAX_STACK_SIZE: int = 100

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
SETTINGS_FILENAME: str = "settings.json"

# ─────────────────────────────────────────────────────────────────────────────
# COLORS (RGBA) - Professional CAD Theme
# ─────────────────────────────────────────────────────────────────────────────
# UI Colors
COLOR_BG_PRIMARY: tuple[int, int, int, int] = (30, 30, 30, 255)
COLOR_BG_SECONDARY: tuple[int, int, int, int] = (45, 45, 48, 255)
COLOR_BG_TERTIARY: tuple[int, int, int, int] = (60, 60, 65, 255)
COLOR_TEXT_PRIMARY: tuple[int, int, int, int] = (255, 255, 255, 255)
COLOR_TEXT_SECONDARY: tuple[int, int, int, int] = (180, 180, 180, 255)
COLOR_TEXT_MUTED: tuple[int, int, int, int] = (120, 120, 120, 255)

# Accent colors
COLOR_ACCENT_PRIMARY: tuple[int, int, int, int] = (59, 130, 246, 255)  # Blue
COLOR_ACCENT_SECONDARY: tuple[int, int, int, int] = (139, 92, 246, 255)  # Purple
COLOR_ACCENT_SUCCESS: tuple[int, int, int, int] = (34, 197, 94, 255)  # Green
COLOR_ACCENT_WARNING: tuple[int, int, int, int] = (251, 191, 36, 255)  # Yellow
COLOR_ACCENT_ERROR: tuple[int, int, int, int] = (239, 68, 68, 255)  # Red

# Canvas colors
COLOR_GRID: tuple[int, int, int, int] = (70, 70, 75, 180)
COLOR_GRID_MINOR: tuple[int, int, int, int] = (50, 50, 55, 100)
COLOR_AXIS: tuple[int, int, int, int] = (100, 100, 105, 255)
COLOR_ORIGIN: tuple[int, int, int, int] = (255, 100, 100, 255)
COLOR_CIRCLE: tuple[int, int, int, int] = (59, 130, 246, 255)
COLOR_BOX: tuple[int, int, int, int] = (139, 92, 246, 255)
COLOR_SURFACE: tuple[int, int, int, int] = (80, 80, 85, 255)
COLOR_VELOCITY: tuple[int, int, int, int] = (239, 68, 68, 255)
COLOR_FORCE: tuple[int, int, int, int] = (59, 130, 246, 255)
COLOR_GHOST: tuple[int, int, int, int] = (100, 100, 105, 150)
COLOR_HIGHLIGHT: tuple[int, int, int, int] = (250, 204, 21, 220)
COLOR_SELECTED: tuple[int, int, int, int] = (251, 191, 36, 255)
COLOR_BOUNDS: tuple[int, int, int, int] = (50, 50, 55, 120)
COLOR_LABEL: tuple[int, int, int, int] = (200, 200, 205, 255)
COLOR_SPRING_REST: tuple[int, int, int, int] = (251, 191, 36, 255)
COLOR_SPRING_STRETCHED: tuple[int, int, int, int] = (239, 68, 68, 255)
COLOR_SPRING_COMPRESSED: tuple[int, int, int, int] = (59, 130, 246, 255)

# UI Elements
COLOR_BORDER: tuple[int, int, int, int] = (55, 55, 60, 255)
COLOR_BORDER_ACTIVE: tuple[int, int, int, int] = (59, 130, 246, 255)
COLOR_HOVER: tuple[int, int, int, int] = (70, 70, 75, 180)
COLOR_SEPARATOR: tuple[int, int, int, int] = (45, 45, 48, 255)

# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
UI_MARGIN: int = 10
UI_WRAP_WIDTH: int = 300
UI_CHILD_HEIGHT_TOOL: int = 260
UI_CHILD_HEIGHT_PROP: int = 300
UI_SPACING: int = 4
UI_BUTTON_WIDTH_SMALL: int = 35
UI_BUTTON_WIDTH_MED: int = 60
UI_BUTTON_WIDTH_LARGE: int = 80

# ─────────────────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────────────────
RENDER_UNIT_SCALE: float = 1.0  # World units per displayed user unit (meters)

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COMMAND_PALETTE_ENABLED: bool = True
COMMAND_TRIGGER_KEY: str = "Ctrl+P"

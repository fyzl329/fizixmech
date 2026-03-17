# Controls Reference

## Mouse Controls

| Action | Input |
|--------|-------|
| **Select object** | Click on object (Select mode) |
| **Place object** | Left-click (Body/Surface/Force mode) |
| **Draw force vector** | Click and drag (Force mode) |
| **Fling object** | Click object, drag, release (Fling mode) |
| **Pan camera** | Hold `Ctrl` + drag |
| **Zoom** | Scroll wheel |
| **Temporary select** | Hold `Shift` (any mode) |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Toggle pause/play |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `Ctrl` + `Z` | Undo |
| `Ctrl` + `Y` | Redo |
| `Delete` | Delete selected object |
| `Ctrl` + `R` | Restart simulation |

## Tool Modes

### Select
- Click objects to select and edit properties
- Hold `Shift` in any mode for temporary select

### Fling
- Click a body to select
- Drag to set impulse direction and magnitude
- Release to apply impulse

### Surface
- Click and drag to draw static surfaces
- Surfaces have friction and elasticity properties

### Force
- **Global mode**: Draw from origin, applies to all bodies
- **Local mode**: Click a body, drag to set force on that body only

### Body
- Place boxes or circles
- Configure mass, size in Tool Properties

### Spring
- Click a body (or empty space for anchor)
- Click another body (or empty space) to complete
- Springs apply Hooke's law forces

## Snapping Options

| Option | Description |
|--------|-------------|
| **Grid Snap** | Snap positions to grid |
| **Angle Snap** | Snap vector angles to increments |
| **Magnitude Snap** | Snap vector lengths to increments |

## UI Panels

### Controls (Left)
- Simulation controls (play, pause, restart)
- Gravity slider
- Mode selection
- Snapping settings
- Edit tools (undo, redo, delete)
- Tool properties
- Selection properties
- Calculator

### Viewport (Center)
- Main simulation canvas
- Hierarchy panel (right side)

### Hierarchy (Right)
- Tree view of all objects
- Bodies, Surfaces, Forces lists
- Click to select objects

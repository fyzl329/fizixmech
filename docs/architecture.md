# Architecture Documentation

## Overview

Fizix Mech is a 2D physics sandbox application built with Python, Dear PyGui, and Pymunk.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Controls   │  │   Viewport   │  │   Hierarchy/Props   │   │
│  │   Panel     │  │   (Canvas)   │  │      Panel          │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  App Class (src/app.py)                                  │  │
│  │  - Input handling (mouse, keyboard)                      │  │
│  │  - Tool management (select, fling, surface, force, etc.) │  │
│  │  - Undo/Redo system                                      │  │
│  │  - Settings persistence                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│     PHYSICS LAYER        │  │    RENDERER LAYER        │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Physics Class      │  │  │  │ Renderer Class     │  │
│  │ - Pymunk Space     │  │  │  │ - Drawlist mgmt    │  │
│  │ - Body management  │  │  │  │ - Coordinate xform │  │
│  │ - Force application│  │  │  │ - Shape rendering  │  │
│  │ - Spring physics   │  │  │  │ - Visual effects   │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
└──────────────────────────┘  └──────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA MODELS                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │   Vector    │  │    Spring    │  │    LocalForce       │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/main.py`
Application entry point. Initializes and runs the main application loop.

### `src/app.py`
Core application logic:
- Event handling (mouse, keyboard)
- Tool state management
- Undo/Redo system
- Settings load/save
- Rendering orchestration

### `src/physics.py`
Physics simulation engine:
- Pymunk space management
- Body creation (circles, boxes)
- Static surfaces
- Force application (global vectors, local forces)
- Spring dynamics
- Integration step

### `src/renderer.py`
Visual rendering:
- Coordinate transformations (world ↔ screen)
- Grid and axis drawing
- Shape rendering (circles, boxes, surfaces)
- Vector and spring visualization
- Selection highlights

### `src/ui.py`
User interface construction:
- Control panels
- Tool properties
- Selection properties
- Hierarchy view
- Calculator widget

### `src/config.py`
Centralized configuration:
- Window dimensions
- Zoom settings
- Physics parameters
- Color definitions
- Tool defaults

### `src/models/`
Data models:
- `vector.py` - Force vector representation
- `spring.py` - Spring connections
- `local_force.py` - Body-specific forces

## Data Flow

### User Interaction → Physics
```
User clicks canvas
    → App.on_mouse_down()
    → Tool-specific handler
    → Physics.add_circle()/add_box()/add_surface()
    → Pymunk space updated
```

### Physics → Rendering
```
Physics.step(dt)
    → App._render()
    → Renderer.draw_shape() for each body
    → Dear PyGui drawlist updated
```

### Undo/Redo Flow
```
User action
    → App._push_undo()
    → Snapshot current state
    → Store in undo stack
    → On undo: restore previous snapshot
```

## Key Design Patterns

### State Pattern
Tools (select, fling, surface, force, body, spring) behave as different states with distinct mouse handling.

### Command Pattern (Implicit)
Undo/Redo uses state snapshots rather than command objects for simplicity.

### Observer Pattern (Implicit)
UI panels update when selection changes via `_update_properties_panel()`.

## Threading Model

Single-threaded with a fixed timestep physics loop:
- Main thread handles UI, rendering, and physics
- Physics uses accumulator pattern for stable integration
- Frame rate uncapped, physics timestep fixed at 1/240s

## Configuration Files

| File | Purpose |
|------|---------|
| `settings.json` | User preferences (auto-generated) |
| `pyproject.toml` | Project metadata and build config |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Development dependencies |

## Extension Points

### Adding New Tools
1. Add tool mode to `App.mode`
2. Implement mouse handlers (`on_mouse_down`, `on_mouse_drag`, `on_mouse_up`)
3. Add tool properties panel in `_update_tool_panel()`
4. Update `build_ui()` radio button list

### Adding New Shape Types
1. Add creation method in `Physics` class
2. Add rendering in `Renderer.draw_shape()`
3. Add property editing in `_update_properties_panel()`
4. Add to hierarchy in `_rebuild_hierarchy()`

### Adding New Force Types
1. Create model in `src/models/`
2. Add application logic in `Physics.apply_*()`
3. Add visualization in `Renderer.draw_*()`
4. Add UI controls in `src/ui.py`

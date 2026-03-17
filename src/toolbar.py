"""
CAD-style toolbar for Fizix Mech.

Provides a professional toolbar with icon-based tools.
"""

import dearpygui.dearpygui as dpg
from typing import Callable, Optional

# Toolbar icons using Unicode symbols (will be replaced with actual icons later)
ICONS = {
    "select": "⬚",      # ◻ ⟡ ◇
    "move": "✢",
    "rotate": "⟳",
    "scale": "⤢",
    "body_box": "□",
    "body_circle": "○",
    "surface": "╱",
    "force": "➤",
    "spring": "〰",
    "fling": "⚡",
    "delete": "🗑",
    "undo": "↶",
    "redo": "↷",
    "play": "▶",
    "pause": "⏸",
    "stop": "⏹",
    "grid": "▦",
    "snap": "📍",
    "settings": "⚙",
    "layers": "📑",
    "measure": "📏",
    "view_top": "⊤",
    "view_front": "⊥",
    "view_iso": "⬓",
    "zoom_in": "🔍",
    "zoom_out": "🔎",
    "zoom_fit": "⤢",
    "help": "?",
}


class ToolbarButton:
    """Represents a toolbar button."""
    
    def __init__(
        self,
        label: str,
        icon: str,
        callback: Callable,
        tooltip: str = "",
        toggle: bool = False,
        group: str = "",
        default: bool = False,
    ):
        self.label = label
        self.icon = icon
        self.callback = callback
        self.tooltip = tooltip
        self.toggle = toggle
        self.group = group
        self.is_active = default
        self.tag: Optional[str] = None


def create_toolbar(parent: str, app) -> None:
    """Create the main CAD-style toolbar."""
    
    with dpg.group(parent=parent, horizontal=True):
        # File/Save toolbar section
        with dpg.group(horizontal=True):
            dpg.add_button(label=f"{ICONS['settings']} Settings", 
                          callback=lambda: _show_settings(app),
                          width=80)
            dpg.add_button(label=f"{ICONS['help']} Help",
                          callback=lambda: _show_help(app),
                          width=60)
        
        dpg.add_separator(direction=dpg.mvX_Axis)
        
        # Simulation controls
        with dpg.group(horizontal=True):
            play_btn = dpg.add_button(
                label=f"{ICONS['play']}",
                callback=lambda: _toggle_play(app),
                width=40
            )
            dpg.add_item_tooltip(play_btn, "Play/Pause (Space)")
            
            stop_btn = dpg.add_button(
                label=f"{ICONS['stop']}",
                callback=lambda: _stop_sim(app),
                width=40
            )
            dpg.add_item_tooltip(stop_btn, "Restart (Ctrl+R)")
        
        dpg.add_separator(direction=dpg.mvX_Axis)
        
        # Edit tools
        with dpg.group(horizontal=True):
            undo_btn = dpg.add_button(
                label=f"{ICONS['undo']}",
                callback=lambda: app.undo(),
                width=40
            )
            dpg.add_item_tooltip(undo_btn, "Undo (Ctrl+Z)")
            
            redo_btn = dpg.add_button(
                label=f"{ICONS['redo']}",
                callback=lambda: app.redo(),
                width=40
            )
            dpg.add_item_tooltip(redo_btn, "Redo (Ctrl+Y)")
            
            del_btn = dpg.add_button(
                label=f"{ICONS['delete']}",
                callback=lambda: app.delete_selected(),
                width=40
            )
            dpg.add_item_tooltip(del_btn, "Delete (Del)")
        
        dpg.add_separator(direction=dpg.mvX_Axis)
        
        # Tool selection (radio group style)
        with dpg.group(horizontal=True):
            _create_tool_button("select", ICONS["select"], app, "Select (V)")
            _create_tool_button("move", ICONS["move"], app, "Move (M)")
            dpg.add_separator(direction=dpg.mvX_Axis)
            _create_tool_button("body_box", ICONS["body_box"], app, "Box (B)")
            _create_tool_button("body_circle", ICONS["body_circle"], app, "Circle (C)")
            _create_tool_button("surface", ICONS["surface"], app, "Surface (S)")
            _create_tool_button("force", ICONS["force"], app, "Force (F)")
            _create_tool_button("spring", ICONS["spring"], app, "Spring (P)")
            _create_tool_button("fling", ICONS["fling"], app, "Fling (I)")
        
        dpg.add_separator(direction=dpg.mvX_Axis)
        
        # View controls
        with dpg.group(horizontal=True):
            grid_btn = dpg.add_button(
                label=f"{ICONS['grid']}",
                callback=lambda: _toggle_grid(app),
                width=40
            )
            dpg.add_item_tooltip(grid_btn, "Toggle Grid (G)")
            
            snap_btn = dpg.add_button(
                label=f"{ICONS['snap']}",
                callback=lambda: _toggle_snap(app),
                width=40
            )
            dpg.add_item_tooltip(snap_btn, "Toggle Snap")
        
        dpg.add_separator(direction=dpg.mvX_Axis)
        
        # View cube shortcuts
        with dpg.group(horizontal=True):
            top_btn = dpg.add_button(
                label=f"{ICONS['view_top']}",
                callback=lambda: _set_view(app, "top"),
                width=35
            )
            dpg.add_item_tooltip(top_btn, "Top View")
            
            front_btn = dpg.add_button(
                label=f"{ICONS['view_front']}",
                callback=lambda: _set_view(app, "front"),
                width=35
            )
            dpg.add_item_tooltip(front_btn, "Front View")
            
            fit_btn = dpg.add_button(
                label=f"{ICONS['zoom_fit']}",
                callback=lambda: _zoom_fit(app),
                width=35
            )
            dpg.add_item_tooltip(fit_btn, "Zoom to Fit")


def _create_tool_button(tool_name: str, icon: str, app, tooltip: str) -> None:
    """Create a tool selection button."""
    btn = dpg.add_button(
        label=icon,
        width=35,
        callback=lambda: _select_tool(app, tool_name)
    )
    dpg.add_item_tooltip(btn, tooltip)


def _select_tool(app, tool_name: str) -> None:
    """Handle tool selection."""
    # Map toolbar tool names to app modes
    tool_map = {
        "select": "select",
        "move": "select",  # Move uses select mode
        "body_box": "body",
        "body_circle": "body",
        "surface": "surface",
        "force": "force",
        "spring": "spring",
        "fling": "fling",
    }
    
    mode = tool_map.get(tool_name, "select")
    app.on_mode_change(mode)
    
    # Set body tool type for body tools
    if tool_name == "body_box":
        app.body_tool_type = "box"
    elif tool_name == "body_circle":
        app.body_tool_type = "circle"


def _toggle_play(app) -> None:
    """Toggle simulation play/pause."""
    app.toggle_simulation()


def _stop_sim(app) -> None:
    """Restore the startup scene snapshot."""
    app.restore_startup_scene()


def _toggle_grid(app) -> None:
    """Toggle grid visibility."""
    app.toggle_grid()


def _toggle_snap(app) -> None:
    """Toggle snap."""
    app.on_snap_toggle(not app.snap)


def _set_view(app, view_name: str) -> None:
    """Set camera to preset view."""
    if view_name == "top":
        app.R.cam = [0.0, 0.0]
        app.R.zoom = 80.0
    elif view_name == "front":
        app.R.cam = [0.0, 0.0]
        app.R.zoom = 80.0
    # Add more views as needed


def _zoom_fit(app) -> None:
    """Zoom to fit all objects."""
    # Calculate bounds of all objects and zoom to fit
    if not app.physics.dynamic:
        app.R.cam = [0.0, 0.0]
        app.R.zoom = 80.0
        return
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for shape in app.physics.dynamic:
        pos = shape.body.position
        if hasattr(shape, 'radius'):
            r = shape.radius
            min_x = min(min_x, pos.x - r)
            max_x = max(max_x, pos.x + r)
            min_y = min(min_y, pos.y - r)
            max_y = max(max_y, pos.y + r)
        else:
            # Box - approximate
            min_x = min(min_x, pos.x - 1)
            max_x = max(max_x, pos.x + 1)
            min_y = min(min_y, pos.y - 1)
            max_y = max(max_y, pos.y + 1)
    
    # Calculate zoom to fit
    if max_x > min_x and max_y > min_y:
        canvas_w = app.R.w
        canvas_h = app.R.h
        zoom_x = canvas_w / (max_x - min_x) / 1.1
        zoom_y = canvas_h / (max_y - min_y) / 1.1
        app.R.zoom = min(zoom_x, zoom_y)
        app.R.cam = [(min_x + max_x) / 2, (min_y + max_y) / 2]


def _show_settings(app) -> None:
    """Show settings panel."""
    app.show_settings()


def _show_help(app) -> None:
    """Show help panel."""
    app.show_help()

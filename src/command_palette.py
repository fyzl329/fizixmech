"""
Command Palette for Fizix Mech.

Provides keyboard-driven command execution for power users.
"""

import dearpygui.dearpygui as dpg
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Command:
    """Represents a command in the palette."""
    name: str
    description: str
    callback: Callable
    shortcut: str = ""
    category: str = "General"
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = self.name.lower().split()


class CommandPalette:
    """Professional command palette (Ctrl+P style)."""
    
    def __init__(self, app):
        self.app = app
        self.commands: Dict[str, Command] = {}
        self.filtered_commands: List[str] = []
        self.selected_index = 0
        self.visible = False
        self.search_query = ""
        self._setup()
    
    def _setup(self) -> None:
        """Setup the command palette UI."""
        # Create the palette window (hidden by default)
        with dpg.window(
            label="Command Palette",
            modal=True,
            no_resize=True,
            no_move=True,
            no_collapse=True,
            no_title_bar=True,
            no_scrollbar=True,
            tag="cmd_palette_win",
            show=False,
            width=500,
            height=400,
        ):
            # Search input
            dpg.add_input_text(
                hint="Type a command...",
                tag="cmd_palette_search",
                width=480,
                callback=self._on_search,
                on_enter=False,
            )
            
            # Command list
            with dpg.child_window(
                tag="cmd_palette_list",
                width=480,
                height=300,
                border=False,
            ):
                pass
            
            # Hints
            dpg.add_text("↑↓ Navigate  •  Enter Select  •  Esc Close", 
                        color=(120, 120, 120, 255))
        
        # Register keyboard handler
        self._register_handlers()
        
        # Register default commands
        self._register_default_commands()
    
    def _register_handlers(self) -> None:
        """Register keyboard handlers."""
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_P, callback=self._toggle)
            dpg.add_key_press_handler(dpg.mvKey_F1, callback=lambda s, a: self.app.show_help())
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=self._on_escape)
            dpg.add_key_press_handler(dpg.mvKey_Return, callback=self._on_enter)
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=lambda s, a: self._on_nav(-1))
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=lambda s, a: self._on_nav(1))
    
    def _register_default_commands(self) -> None:
        """Register default commands."""
        # File commands
        self.register(Command(
            name="File: Save",
            description="Save current simulation",
            callback=lambda: self.app._save_settings(),
            shortcut="Ctrl+S",
            category="File",
            keywords=["save", "file", "settings"]
        ))
        
        self.register(Command(
            name="File: Restore Startup Scene",
            description="Restore the startup snapshot",
            callback=lambda: self.app.restore_startup_scene(),
            shortcut="Ctrl+R",
            category="File",
            keywords=["restore", "startup", "snapshot", "reset", "file"]
        ))
        
        # Edit commands
        self.register(Command(
            name="Edit: Undo",
            description="Undo last action",
            callback=lambda: self.app.undo(),
            shortcut="Ctrl+Z",
            category="Edit",
            keywords=["undo", "edit", "back"]
        ))
        
        self.register(Command(
            name="Edit: Redo",
            description="Redo last undone action",
            callback=lambda: self.app.redo(),
            shortcut="Ctrl+Y",
            category="Edit",
            keywords=["redo", "edit", "forward"]
        ))
        
        self.register(Command(
            name="Edit: Delete",
            description="Delete selected object",
            callback=lambda: self.app.delete_selected(),
            shortcut="Delete",
            category="Edit",
            keywords=["delete", "remove", "edit"]
        ))
        
        # View commands
        self.register(Command(
            name="View: Zoom In",
            description="Zoom in",
            callback=lambda: setattr(self.app, 'zoom_target', min(400, self.app.zoom_target * 1.2)),
            shortcut="+",
            category="View",
            keywords=["zoom", "in", "view"]
        ))
        
        self.register(Command(
            name="View: Zoom Out",
            description="Zoom out",
            callback=lambda: setattr(self.app, 'zoom_target', max(20, self.app.zoom_target / 1.2)),
            shortcut="-",
            category="View",
            keywords=["zoom", "out", "view"]
        ))
        
        self.register(Command(
            name="View: Reset Camera",
            description="Reset camera to origin",
            callback=lambda: self.app.go_to_origin(),
            shortcut="Home",
            category="View",
            keywords=["camera", "reset", "origin", "view"]
        ))
        
        self.register(Command(
            name="View: Toggle Grid",
            description="Toggle grid visibility",
            callback=lambda: self.app.toggle_grid(),
            shortcut="G",
            category="View",
            keywords=["grid", "toggle", "view"]
        ))
        
        # Simulation commands
        self.register(Command(
            name="Sim: Play/Pause",
            description="Toggle simulation",
            callback=lambda: self.app.toggle_simulation(),
            shortcut="Space",
            category="Simulation",
            keywords=["play", "pause", "sim", "simulation"]
        ))
        
        self.register(Command(
            name="Sim: Toggle Snap",
            description="Toggle grid snap",
            callback=lambda: self.app.on_snap_toggle(not self.app.snap),
            shortcut="",
            category="Simulation",
            keywords=["snap", "grid", "sim"]
        ))
        
        # Tool commands
        tools = [
            ("Select", "select", "V"),
            ("Body: Box", "body", "B", ["box", "body", "tool"]),
            ("Body: Circle", "body", "C", ["circle", "body", "tool"]),
            ("Surface", "surface", "S"),
            ("Force", "force", "F"),
            ("Spring", "spring", "P"),
            ("Fling", "fling", "I"),
        ]
        
        for tool in tools:
            name = tool[0]
            mode = tool[1]
            shortcut = tool[2] if len(tool) > 2 else ""
            keywords = tool[3] if len(tool) > 3 else [name.lower(), "tool"]
            
            self.register(Command(
                name=f"Tool: {name}",
                description=f"Select {name} tool",
                callback=lambda m=mode: self._select_tool(m),
                shortcut=shortcut,
                category="Tools",
                keywords=keywords
            ))
        
        # Help commands
        self.register(Command(
            name="Help: Controls",
            description="Show controls reference",
            callback=lambda: self._show_help(),
            shortcut="F1",
            category="Help",
            keywords=["help", "controls", "keyboard"]
        ))
    
    def _select_tool(self, mode: str) -> None:
        """Select a tool and hide palette."""
        self.app.select_tool(mode)
        self.hide()
    
    def _show_help(self) -> None:
        """Show help panel."""
        self.app.show_help()
        self.hide()

    def _on_escape(self, sender=None, app_data=None) -> None:
        if self.visible:
            self.hide()

    def _on_enter(self, sender=None, app_data=None) -> None:
        if self.visible:
            self.execute_selected()

    def _on_nav(self, direction: int) -> None:
        if self.visible:
            self.navigate(direction)
    
    def register(self, command: Command) -> None:
        """Register a command."""
        self.commands[command.name] = command
        self._update_list()
    
    def _toggle(self, sender=None, app_data=None) -> None:
        """Toggle palette visibility."""
        # Check if Ctrl is held
        ctrl_down = self._ctrl_down()
        
        if ctrl_down:
            if self.visible:
                self.hide()
            else:
                self.show()

    def _ctrl_down(self) -> bool:
        for attr in ("mvKey_Control", "mvKey_LControl", "mvKey_RControl"):
            code = getattr(dpg, attr, None)
            try:
                if code is not None and dpg.is_key_down(code):
                    return True
            except Exception:
                pass
        return False
    
    def show(self) -> None:
        """Show the command palette."""
        self.visible = True
        self.search_query = ""
        self.selected_index = 0
        dpg.set_value("cmd_palette_search", "")
        dpg.configure_item("cmd_palette_win", show=True)
        dpg.focus_item("cmd_palette_search")
        self._update_list()
    
    def hide(self) -> None:
        """Hide the command palette."""
        self.visible = False
        dpg.configure_item("cmd_palette_win", show=False)
    
    def _on_search(self, sender, app_data) -> None:
        """Handle search input."""
        self.search_query = app_data.lower()
        self.selected_index = 0
        self._update_list()
    
    def _update_list(self) -> None:
        """Update the command list based on search query."""
        # Filter commands
        if not self.search_query:
            self.filtered_commands = list(self.commands.keys())
        else:
            self.filtered_commands = [
                name for name, cmd in self.commands.items()
                if (self.search_query in name.lower() or
                    self.search_query in cmd.description.lower() or
                    any(self.search_query in kw for kw in cmd.keywords))
            ]
        
        # Rebuild list UI
        dpg.delete_item("cmd_palette_list", children_only=True)
        
        for i, name in enumerate(self.filtered_commands):
            cmd = self.commands[name]
            is_selected = (i == self.selected_index)
            
            with dpg.group(horizontal=True, parent="cmd_palette_list"):
                # Selection indicator
                dpg.add_text("▶" if is_selected else " ", 
                            color=(59, 130, 246, 255) if is_selected else (0, 0, 0, 0))
                
                # Command name
                dpg.add_text(cmd.name, 
                            color=(255, 255, 255, 255) if is_selected else (200, 200, 205, 255))
                
                # Shortcut
                if cmd.shortcut:
                    dpg.add_text(f"  [{cmd.shortcut}]", 
                                color=(120, 120, 120, 255))
            
            # Description
            dpg.add_text(cmd.description, 
                        color=(120, 120, 120, 255), 
                        parent="cmd_palette_list",
                        indent=20)
            
            # Separator
            if i < len(self.filtered_commands) - 1:
                dpg.add_separator(parent="cmd_palette_list")
    
    def navigate(self, direction: int) -> None:
        """Navigate command list."""
        if not self.filtered_commands:
            return
        
        self.selected_index = (self.selected_index + direction) % len(self.filtered_commands)
        self._update_list()
    
    def execute_selected(self) -> None:
        """Execute the currently selected command."""
        if self.filtered_commands:
            name = self.filtered_commands[self.selected_index]
            cmd = self.commands[name]
            cmd.callback()
            self.hide()

"""
Professional status bar for Fizix Mech.

Shows coordinates, simulation info, and quick stats.
"""

import dearpygui.dearpygui as dpg
from typing import Optional


class StatusBar:
    """Professional CAD-style status bar."""
    
    def __init__(self, app):
        self.app = app
        self.tag_prefix = "status_"
        self._create()
    
    def _create(self) -> None:
        """Create the status bar."""
        with dpg.window(
            label="Status Bar",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            no_collapse=True,
            no_background=False,
            menubar=False,
            tag=f"{self.tag_prefix}window",
        ):
            with dpg.group(horizontal=True):
                # Coordinate display
                dpg.add_text("X: ", tag=f"{self.tag_prefix}x_label")
                dpg.add_text("0.000", tag=f"{self.tag_prefix}x_val", color=(100, 200, 255, 255))
                dpg.add_text("  Y: ", tag=f"{self.tag_prefix}y_label")
                dpg.add_text("0.000", tag=f"{self.tag_prefix}y_val", color=(100, 200, 255, 255))
                dpg.add_text("  Z: ", tag=f"{self.tag_prefix}z_label")
                dpg.add_text("0.000", tag=f"{self.tag_prefix}z_val", color=(100, 200, 255, 255))
                
                dpg.add_separator(direction=dpg.mvX_Axis)
                
                # Simulation status
                dpg.add_text("●", tag=f"{self.tag_prefix}sim_indicator", color=(34, 197, 94, 255))
                dpg.add_text("Running", tag=f"{self.tag_prefix}sim_status")
                
                dpg.add_separator(direction=dpg.mvX_Axis)
                
                # Stats
                dpg.add_text("Bodies: ", tag=f"{self.tag_prefix}bodies_label")
                dpg.add_text("0", tag=f"{self.tag_prefix}bodies_val")
                
                dpg.add_text("  |  Surfaces: ", tag=f"{self.tag_prefix}surfaces_label")
                dpg.add_text("0", tag=f"{self.tag_prefix}surfaces_val")
                
                dpg.add_text("  |  KE: ", tag=f"{self.tag_prefix}ke_label")
                dpg.add_text("0.0 J", tag=f"{self.tag_prefix}ke_val")
                
                dpg.add_separator(direction=dpg.mvX_Axis)
                
                # Zoom level
                dpg.add_text("Zoom: ", tag=f"{self.tag_prefix}zoom_label")
                dpg.add_text("100%", tag=f"{self.tag_prefix}zoom_val")
                
                dpg.add_separator(direction=dpg.mvX_Axis)
                
                # Current tool
                dpg.add_text("Tool: ", tag=f"{self.tag_prefix}tool_label")
                dpg.add_text("Select", tag=f"{self.tag_prefix}tool_val", color=(251, 191, 36, 255))
                
                dpg.add_separator(direction=dpg.mvX_Axis)
                
                # Grid snap indicator
                dpg.add_text("GRID", tag=f"{self.tag_prefix}grid_indicator", color=(100, 100, 100, 255))
                dpg.add_text("  SNAP", tag=f"{self.tag_prefix}snap_indicator", color=(100, 100, 100, 255))
        
        # Set initial window position (will be updated in render loop)
        dpg.configure_item(f"{self.tag_prefix}window", pos=(0, 0), width=100, height=30)
    
    def update_coordinates(self, x: float, y: float) -> None:
        """Update coordinate display."""
        try:
            dpg.set_value(f"{self.tag_prefix}x_val", f"{x:7.3f}")
            dpg.set_value(f"{self.tag_prefix}y_val", f"{y:7.3f}")
            dpg.set_value(f"{self.tag_prefix}z_val", "0.000")  # 2D, but show for CAD feel
        except Exception:
            pass
    
    def update_simulation_status(self, running: bool) -> None:
        """Update simulation status indicator."""
        try:
            color = (34, 197, 94, 255) if running else (239, 68, 68, 255)
            dpg.configure_item(f"{self.tag_prefix}sim_indicator", color=color)
            dpg.set_value(f"{self.tag_prefix}sim_status", "Running" if running else "Paused")
        except Exception:
            pass
    
    def update_stats(self, bodies: int, surfaces: int, ke: float) -> None:
        """Update statistics display."""
        try:
            dpg.set_value(f"{self.tag_prefix}bodies_val", str(bodies))
            dpg.set_value(f"{self.tag_prefix}surfaces_val", str(surfaces))
            dpg.set_value(f"{self.tag_prefix}ke_val", f"{ke:.1f} J")
        except Exception:
            pass
    
    def update_zoom(self, zoom: float) -> None:
        """Update zoom level display."""
        try:
            zoom_pct = int(zoom / 80.0 * 100)  # 80 is default = 100%
            dpg.set_value(f"{self.tag_prefix}zoom_val", f"{zoom_pct}%")
        except Exception:
            pass
    
    def update_tool(self, tool_name: str) -> None:
        """Update current tool display."""
        try:
            dpg.set_value(f"{self.tag_prefix}tool_val", tool_name.title())
        except Exception:
            pass
    
    def update_snap_indicators(self, grid: bool, snap: bool) -> None:
        """Update grid and snap indicators."""
        try:
            grid_color = (59, 130, 246, 255) if grid else (100, 100, 100, 255)
            snap_color = (59, 130, 246, 255) if snap else (100, 100, 100, 255)
            dpg.configure_item(f"{self.tag_prefix}grid_indicator", color=grid_color)
            dpg.configure_item(f"{self.tag_prefix}snap_indicator", color=snap_color)
        except Exception:
            pass
    
    def set_position(self, x: int, y: int, width: int) -> None:
        """Update status bar position and size."""
        try:
            dpg.configure_item(f"{self.tag_prefix}window", pos=(x, y), width=width)
        except Exception:
            pass


class CoordinateDisplay:
    """Floating coordinate display near cursor."""
    
    def __init__(self):
        self.active = False
        self.tag_x = "coord_float_x"
        self.tag_y = "coord_float_y"
        self._create()
    
    def _create(self) -> None:
        """Create floating coordinate display."""
        # Hidden by default, shown when needed
        with dpg.window(
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            no_background=True,
            no_inputs=True,
            tag="coord_float_win",
            show=False,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("X:", tag=self.tag_x, color=(100, 200, 255, 255))
                dpg.add_text("Y:", tag=self.tag_y, color=(100, 200, 255, 255))
    
    def update(self, x: float, y: float, screen_x: int, screen_y: int) -> None:
        """Update position and coordinates."""
        try:
            dpg.set_value(self.tag_x, f"X: {x:7.3f}")
            dpg.set_value(self.tag_y, f"Y: {y:7.3f}")
            dpg.configure_item("coord_float_win", pos=(screen_x + 15, screen_y + 15))
            if not self.active:
                dpg.configure_item("coord_float_win", show=True)
                self.active = True
        except Exception:
            pass
    
    def hide(self) -> None:
        """Hide the coordinate display."""
        if self.active:
            dpg.configure_item("coord_float_win", show=False)
            self.active = False

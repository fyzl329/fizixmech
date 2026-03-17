"""
Professional CAD-style themes for Fizix Mech.

Provides dark/light theme options with professional color schemes.
"""

from dataclasses import dataclass
from typing import Tuple

RgbTuple = Tuple[int, int, int]
RgbATuple = Tuple[int, int, int, int]


@dataclass
class ThemeColors:
    """Color scheme for a theme."""
    # UI Colors
    bg_primary: RgbTuple = (30, 30, 30)
    bg_secondary: RgbTuple = (45, 45, 48)
    bg_tertiary: RgbTuple = (60, 60, 65)
    text_primary: RgbTuple = (255, 255, 255)
    text_secondary: RgbTuple = (180, 180, 180)
    text_muted: RgbTuple = (120, 120, 120)
    
    # Accent colors
    accent_primary: RgbTuple = (59, 130, 246)  # Blue
    accent_secondary: RgbTuple = (139, 92, 246)  # Purple
    accent_success: RgbTuple = (34, 197, 94)  # Green
    accent_warning: RgbTuple = (251, 191, 36)  # Yellow
    accent_error: RgbTuple = (239, 68, 68)  # Red
    accent_info: RgbTuple = (59, 130, 246)  # Blue
    
    # Canvas colors
    canvas_bg: RgbTuple = (25, 25, 27)
    canvas_grid_major: RgbATuple = (70, 70, 75, 180)
    canvas_grid_minor: RgbATuple = (50, 50, 55, 100)
    canvas_axes: RgbATuple = (100, 100, 105, 255)
    canvas_origin: RgbATuple = (255, 100, 100, 255)
    
    # Object colors
    obj_circle: RgbATuple = (59, 130, 246, 255)
    obj_box: RgbATuple = (139, 92, 246, 255)
    obj_surface: RgbATuple = (80, 80, 85, 255)
    obj_highlight: RgbATuple = (250, 204, 21, 220)
    obj_selected: RgbATuple = (251, 191, 36, 255)
    
    # Physics visualization
    velocity: RgbATuple = (239, 68, 68, 255)
    force_global: RgbATuple = (59, 130, 246, 255)
    force_local: RgbATuple = (34, 197, 94, 255)
    spring_rest: RgbATuple = (251, 191, 36, 255)
    spring_stretched: RgbATuple = (239, 68, 68, 255)
    spring_compressed: RgbATuple = (59, 130, 246, 255)
    
    # UI Elements
    border: RgbTuple = (55, 55, 60)
    border_active: RgbTuple = (59, 130, 246)
    hover: RgbATuple = (70, 70, 75, 180)
    separator: RgbTuple = (45, 45, 48)


@dataclass
class LightThemeColors(ThemeColors):
    """Light theme color scheme."""
    # UI Colors
    bg_primary: RgbTuple = (245, 245, 245)
    bg_secondary: RgbTuple = (255, 255, 255)
    bg_tertiary: RgbTuple = (240, 240, 242)
    text_primary: RgbTuple = (30, 30, 30)
    text_secondary: RgbTuple = (80, 80, 80)
    text_muted: RgbTuple = (140, 140, 140)
    
    # Canvas colors
    canvas_bg: RgbTuple = (250, 250, 252)
    canvas_grid_major: RgbATuple = (180, 180, 185, 200)
    canvas_grid_minor: RgbATuple = (220, 220, 225, 150)
    canvas_axes: RgbATuple = (80, 80, 85, 255)
    canvas_origin: RgbATuple = (220, 50, 50, 255)
    
    # Object colors
    obj_surface: RgbATuple = (180, 180, 185, 255)
    
    # UI Elements
    border: RgbTuple = (200, 200, 205)
    border_active: RgbTuple = (59, 130, 246)
    hover: RgbATuple = (200, 200, 205, 150)
    separator: RgbTuple = (220, 220, 225)


class ThemeManager:
    """Manages application themes."""
    
    DARK = "dark"
    LIGHT = "light"
    
    def __init__(self):
        self._themes = {
            self.DARK: ThemeColors(),
            self.LIGHT: LightThemeColors(),
        }
        self._current = self.DARK
    
    @property
    def current(self) -> ThemeColors:
        """Get current theme colors."""
        return self._themes[self._current]
    
    @property
    def current_name(self) -> str:
        """Get current theme name."""
        return self._current
    
    def set_theme(self, name: str) -> None:
        """Set the active theme."""
        if name in self._themes:
            self._current = name
    
    def toggle(self) -> str:
        """Toggle between dark and light themes."""
        self._current = self.LIGHT if self._current == self.DARK else self.DARK
        return self._current
    
    def get_theme(self, name: str) -> ThemeColors:
        """Get a specific theme by name."""
        return self._themes.get(name, self._themes[self.DARK])


# Global theme manager instance
theme_manager = ThemeManager()


def get_current_theme() -> ThemeColors:
    """Get the current theme colors."""
    return theme_manager.current


def set_theme(name: str) -> None:
    """Set the global theme."""
    theme_manager.set_theme(name)


def toggle_theme() -> str:
    """Toggle the global theme."""
    return theme_manager.toggle()

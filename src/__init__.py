# Fizix Mech - 2D Physics Sandbox
"""
FZXMCH - A professional physics sandbox built with Dear PyGui and Pymunk.

Professional CAD-style interface with beginner-friendly accessibility.
"""

__version__ = "1.0.2"
__author__ = "Fayazul"

# Core modules
from .app import App
from .physics import Physics
from .renderer import Renderer

# UI Components
from .toolbar import create_toolbar
from .statusbar import StatusBar
from .command_palette import CommandPalette

# Configuration
from . import config
from . import theme

__all__ = [
    "App",
    "Physics", 
    "Renderer",
    "create_toolbar",
    "StatusBar",
    "CommandPalette",
    "config",
    "theme",
]

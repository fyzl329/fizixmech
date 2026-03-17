"""
Welcome/Onboarding overlay for beginners.

Provides a friendly introduction to Fizix Mech.
"""

import dearpygui.dearpygui as dpg


class WelcomeOverlay:
    """Professional welcome overlay for first-time users."""
    
    def __init__(self, app):
        self.app = app
        self.shown = False
        self._create()
    
    def _create(self) -> None:
        """Create the welcome overlay."""
        # Main welcome window
        with dpg.window(
            label="Welcome to Fizix Mech",
            modal=True,
            no_resize=False,
            no_move=True,
            no_collapse=True,
            no_title_bar=False,
            tag="welcome_win",
            show=False,
            width=600,
            height=500,
        ):
            # Header
            dpg.add_text("Welcome to Fizix Mech!", color=(59, 130, 246, 255))
            dpg.add_text("Professional Physics Sandbox", color=(180, 180, 180, 255))
            dpg.add_separator()
            
            # Introduction
            dpg.add_text(
                "Fizix Mech is a CAD-style physics sandbox that combines "
                "professional tools with beginner-friendly accessibility.",
                wrap=560
            )
            dpg.add_spacer(height=10)
            
            # Quick start cards
            with dpg.group(horizontal=True):
                with dpg.child_window(width=180, height=150, border=True):
                    dpg.add_text("🎯 Create Objects", color=(59, 130, 246, 255))
                    dpg.add_text("Click body tools (□ ○) and click on the canvas to place objects.", wrap=170)
                
                with dpg.child_window(width=180, height=150, border=True):
                    dpg.add_text("▶ Simulate", color=(34, 197, 94, 255))
                    dpg.add_text("Press Space or click Play to start the physics simulation.", wrap=170)
                
                with dpg.child_window(width=180, height=150, border=True):
                    dpg.add_text("⌨ Shortcuts", color=(251, 191, 36, 255))
                    dpg.add_text("Press Ctrl+P for command palette. F1 for help.", wrap=170)
            
            dpg.add_spacer(height=15)
            
            # Tips
            with dpg.collapsing_header(label="💡 Pro Tips", default_open=False):
                dpg.add_text("• Use Ctrl+Drag to pan the view", bullet=True)
                dpg.add_text("• Scroll to zoom in/out", bullet=True)
                dpg.add_text("• Hold Shift for temporary select tool", bullet=True)
                dpg.add_text("• Use the command palette (Ctrl+P) for quick access", bullet=True)
                dpg.add_text("• Right-click properties panel to edit object settings", bullet=True)
            
            dpg.add_spacer(height=15)
            dpg.add_separator()
            
            # Actions
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start Creating",
                    callback=lambda: self._close(),
                    width=150
                )
                dpg.add_button(
                    label="Show Controls",
                    callback=lambda: self._show_controls(),
                    width=120
                )
                dpg.add_button(
                    label="Don't Show Again",
                    callback=lambda: self._disable_welcome(),
                    width=140
                )
    
    def show(self) -> None:
        """Show the welcome overlay."""
        dpg.configure_item("welcome_win", show=True)
        self.shown = True
    
    def hide(self) -> None:
        """Hide the welcome overlay."""
        dpg.configure_item("welcome_win", show=False)
        self.shown = False
    
    def _close(self) -> None:
        """Close the welcome overlay."""
        self.app.set_workspace_mode("build")
        self.hide()
    
    def _show_controls(self) -> None:
        """Show controls reference."""
        self.hide()
        self.app.show_help()
    
    def _disable_welcome(self) -> None:
        """Disable welcome on startup."""
        self.app.show_welcome_on_startup = False
        self.app._save_settings()
        self.hide()


class HelpPanel:
    """Contextual help panel."""
    
    def __init__(self, app):
        self.app = app
        self._create()
    
    def _create(self) -> None:
        """Create the help panel."""
        with dpg.window(
            label="Help & Controls",
            no_collapse=True,
            tag="help_win",
            show=False,
            width=500,
            height=600,
        ):
            dpg.add_text("Keyboard Shortcuts", color=(59, 130, 246, 255))
            dpg.add_separator()
            
            # Shortcuts table
            shortcuts = [
                ("Space", "Play/Pause simulation"),
                ("Ctrl+Z", "Undo"),
                ("Ctrl+Y", "Redo"),
                ("Delete", "Delete selected"),
                ("Ctrl+R", "Restart simulation"),
                ("Ctrl+P", "Command palette"),
                ("Ctrl+Drag", "Pan camera"),
                ("Scroll", "Zoom in/out"),
                ("Shift", "Temporary select"),
                ("+ / -", "Zoom in/out"),
            ]
            
            for key, action in shortcuts:
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{key:15}", color=(59, 130, 246, 255))
                    dpg.add_text(action)
            
            dpg.add_spacer(height=15)
            dpg.add_separator()
            
            dpg.add_text("Tools", color=(59, 130, 246, 255))
            dpg.add_separator()
            
            tools = [
                ("Select (V)", "Select and manipulate objects"),
                ("Box (B)", "Create box-shaped bodies"),
                ("Circle (C)", "Create circular bodies"),
                ("Surface (S)", "Draw static surfaces/platforms"),
                ("Force (F)", "Apply forces to objects"),
                ("Spring (P)", "Connect objects with springs"),
                ("Fling (I)", "Apply impulse to throw objects"),
            ]
            
            for tool, desc in tools:
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{tool:15}", color=(139, 92, 246, 255))
                    dpg.add_text(desc)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open Command Palette", callback=lambda: self._open_palette(), width=160)
                dpg.add_button(label="Close", callback=lambda: self.hide(), width=90)

    def show(self) -> None:
        dpg.configure_item("help_win", show=True)

    def hide(self) -> None:
        dpg.configure_item("help_win", show=False)

    def _open_palette(self) -> None:
        self.hide()
        self.app.show_command_palette()

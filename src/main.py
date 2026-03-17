#!/usr/bin/env python3
"""
Fizix Mech (FZXMCH) - 2D Physics Sandbox
Entry point for the application.
"""

from .app import App


def main():
    """Main entry point."""
    app = App()
    app.run()


if __name__ == "__main__":
    main()

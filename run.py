#!/usr/bin/env python3
"""
Fizix Mech (FZXMCH) - 2D Physics Sandbox
Backward-compatible launcher for running from root directory.
"""

import sys
import os

# Add project root to path so src can be imported as a package
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run
from src.app import App


def main():
    """Main entry point."""
    app = App()
    app.run()


if __name__ == "__main__":
    main()

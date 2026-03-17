# FZXMCH - Fizix Mech

> A 2D physics sandbox that lets you **build, break, and simulate** your own mechanical systems.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

Fizix Mech is a physics sandbox designed for students, tinkerers, and anyone curious about mechanical systems. Built with:

- **Pymunk** - Accurate 2D physics simulation (Chipmunk2D wrapper)
- **Dear PyGui** - Fast, immediate-mode GUI framework
- **Python 3.10+** - Clean, modular, open-source code

---

## Features

- **Rigid Body Physics** - Circles and boxes with mass, friction, elasticity
- **Static Surfaces** - Draw platforms and ramps with customizable friction
- **Forces** - Apply global or local forces to bodies
- **Springs** - Connect bodies with Hooke's law springs
- **Fling Tool** - Apply impulses with drag-and-release interaction
- **Snapping** - Grid, angle, and magnitude snapping for precision
- **Undo/Redo** - Full history support
- **Real-time Simulation** - Watch physics in action with pause/play controls

---

## Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/fyzl329/fizixmech.git
cd fizixmech

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application (3 ways)
python run.py           # Option A: Use launcher
python -m src.main      # Option B: Run as module
python src/main.py      # Option C: Direct script
```

### Option 2: Pip Install

```bash
pip install .
fizixmech  # Runs the application
```

### Option 3: Prebuilt Release

Download the latest `.exe` from the [Releases page](https://github.com/fyzl329/fizixmech/releases).

---

## Quick Start

1. **Launch** the application
2. **Select a tool** from the Controls panel (Body, Surface, Force, etc.)
3. **Click on the canvas** to place objects
4. **Press Space** to start/stop the simulation
5. **Experiment!**

---

## Controls

### Mouse

| Action | Input |
|--------|-------|
| Select object | Click (Select mode) |
| Place object | Left-click |
| Draw force | Click + drag (Force mode) |
| Fling object | Click body, drag, release |
| Pan camera | `Ctrl` + drag |
| Zoom | Scroll wheel |
| Temporary select | Hold `Shift` |

### Keyboard

| Key | Action |
|-----|--------|
| `Space` | Pause/Play |
| `+` / `-` | Zoom in/out |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Delete selected |
| `Ctrl+R` | Restart simulation |

See [docs/controls.md](docs/controls.md) for detailed controls reference.

---

## Project Structure

```
fizixmech/
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── app.py                # Application logic
│   ├── physics.py            # Physics engine
│   ├── renderer.py           # Rendering
│   ├── ui.py                 # UI construction
│   ├── config.py             # Configuration constants
│   ├── logger.py             # Logging utility
│   └── models/               # Data models
│       ├── __init__.py
│       ├── vector.py
│       ├── spring.py
│       └── local_force.py
├── assets/                   # Static assets
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── saves/                    # User saves
├── run.py                    # Launcher script
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml            # Project configuration
├── README.md
└── TODO.md                   # Planned improvements
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## Configuration

User settings are stored in `settings.json` (auto-generated on first run):

- Grid snap settings
- Gravity
- Tool defaults (mass, size, friction, etc.)
- Fling tool settings

---

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See [TODO.md](TODO.md) for planned improvements and known issues.

---

## Troubleshooting

### Application won't start

- Ensure Python 3.10+ is installed
- Verify dependencies: `pip install -r requirements.txt`
- Check for error messages in the console

### Poor performance

- Reduce the number of objects
- Disable velocity arrows (if added in future)
- Lower zoom level for fewer rendered elements

### Objects disappear

- Check if they're outside the bounds limit (1000m)
- Objects outside bounds are automatically culled

---

## Credits

**Developer:** Fayazul  
**Built With:** Dear PyGui, Pymunk, Python  
**License:** MIT  

Created with curiosity, the internet, and a lot of caffeine. ☕

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

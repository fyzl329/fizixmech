# Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Python

Download and install Python 3.10 or later from [python.org](https://www.python.org/downloads/).

During installation on Windows:
- ✅ Check "Add Python to PATH"

### Step 2: Download Fizix Mech

**Option A: Git Clone**
```bash
git clone https://github.com/fyzl329/fizixmech.git
cd fizixmech
```

**Option B: Download ZIP**
1. Go to https://github.com/fyzl329/fizixmech
2. Click "Code" → "Download ZIP"
3. Extract to a folder
4. Open terminal in that folder

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run

```bash
python src/main.py
```

---

## Your First Simulation

### 1. Add a Box
1. Select **Body** mode
2. Choose **box** in Tool Properties
3. Click on the canvas

### 2. Add a Circle
1. Keep **Body** mode selected
2. Choose **circle** in Tool Properties
3. Click elsewhere on the canvas

### 3. Add a Platform
1. Select **Surface** mode
2. Click and drag to draw a line below the objects

### 4. Start Simulation
1. Press **Space** or click **Play** button
2. Watch the objects fall and bounce!

### 5. Experiment
- Try **Force** mode to add wind/gravity
- Try **Spring** mode to connect objects
- Try **Fling** mode to throw objects around

---

## Common Issues

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Python not found" error
- Install Python from python.org
- Make sure "Add to PATH" is checked during installation

### Application is slow
- Close other applications
- Reduce the number of objects in the simulation

### Can't select objects
- Make sure you're in **Select** mode
- Or hold **Shift** for temporary select

---

## Next Steps

- Read [docs/controls.md](docs/controls.md) for all controls
- Check [docs/architecture.md](docs/architecture.md) to understand how it works
- See [TODO.md](TODO.md) for upcoming features

---

## Getting Help

1. Check the [README.md](README.md)
2. Review [docs/controls.md](docs/controls.md)
3. Open an issue on [GitHub](https://github.com/fyzl329/fizixmech/issues)

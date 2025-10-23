# FZXMCH

**Version:** 1.0.0  
**Author:** Fayazul  
**Engine:** Dear PyGui + Pymunk  

Fizix Mech is a physics sandbox that lets you *build, break, and simulate* your own little mechanical sims.  
It’s designed for students (me), tinkerers (me), and anyone who ever wanted to see what happens when you throw stuff around at 0.9c in a controlled environment (also me).

---

## An Overview

Fizix Mech uses:
- **Pymunk** for accurate 2D physics simulation
- **Dear PyGui** for a fast, modern interface
- **Python 3.10** for modular, open-source extensibility

Core modules:
- `app.py` – entry point, manages UI and initialization  
- `ui.py` – builds the graphical interface  
- `renderer.py` – draws the world and handles visuals  
- `physics.py` – handles all physics interactions  

---

## Installation

**Option 1: Run from source**

```bash
git clone https://github.com/fyzl329/fizixmech.git
cd fizixmech
pip install -r requirements.txt
python app.py
```

**Option 2: Run the prebuilt release**

Download the latest **`.exe`** from the [Releases page](https://github.com/fyzl329/fizixmech/releases).  
Unzip it and run **`FizixMech.exe`**.

---
## Controls
| Action             | Input Method                     |
|--------------------|----------------------------------|
| **Pan camera**     | Right-click + drag               |
| **Zoom in/out**    | Scroll wheel                     |
| **Place object**   | Left-click                       |
| **Delete object**  | On-screen delete button          |
| **Pause simulation** | On-screen pause/resume button  |
| **Reset simulation** | On-screen reset button         |
| **Change parameters** | UI sliders and input fields   |

---

## Credits

**Developer:** Fayazul  
**Project:** Fizix Mech  
**Built With:** Dear PyGui, Pymunk, and Python 3.10  
**Version:** 1.0.0  

Fizix Mech was designed and developed by Fayazul as a personal physics sandbox project.  
Created with curiosity, the internet, and a lot of caffeine.

---

Licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
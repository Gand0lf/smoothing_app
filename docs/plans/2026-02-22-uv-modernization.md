# UV + Dash Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate from pip/requirements.txt to uv with pyproject.toml, upgrade to Dash 3.x and modern dependencies, remove Heroku deployment.

**Architecture:** Replace requirements.txt with pyproject.toml, update all imports for Dash 3.x consolidated packages, replace deprecated scipy.interpolate.Rbf with RBFInterpolator.

**Tech Stack:** uv, Python 3.12+, Dash 3.x, Flask 3.x, numpy 2.x, scipy 1.15.x

---

### Task 1: Create pyproject.toml and .python-version

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

**Step 1: Create .python-version**
```
3.12
```

**Step 2: Create pyproject.toml**
```toml
[project]
name = "smoothing-app"
version = "0.1.0"
description = "Visualizing and comparing smoothing/interpolation methods"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "dash>=3.0.0",
    "flask>=3.0.0",
    "numpy>=2.0.0",
    "scipy>=1.15.0",
    "pandas>=2.0.0",
    "plotly>=5.0.0",
    "scikit-learn>=1.6.0",
    "pygam>=0.9.0",
]

[project.scripts]
serve = "app:main"
```

**Step 3: Run uv sync**
Run: `uv sync`
Expected: Dependencies installed, `uv.lock` generated

**Step 4: Commit**
```bash
git add pyproject.toml .python-version uv.lock
git commit -m "feat: migrate to uv with pyproject.toml"
```

---

### Task 2: Update app.py imports and remove Heroku code

**Files:**
- Modify: `app.py`

**Step 1: Update imports**
Replace:
```python
import dash
import socket
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
```
With:
```python
import dash
from dash import html, dcc, Input, Output, State
```

**Step 2: Remove socket/IP variables**
Delete lines:
```python
import socket
...
ip = socket.gethostbyname(socket.gethostname())
portid = 8080
```

**Step 3: Update run_server call**
Replace:
```python
if __name__ == '__main__':
    print("hostname:" + str(ip) + str(portid))
    app.title = "Smoothing Study"
    app.run_server(debug=False)
```
With:
```python
if __name__ == '__main__':
    app.title = "Smoothing Study"
    app.run(debug=True)
```

**Step 4: Verify app runs**
Run: `uv run python app.py`
Expected: App starts at http://127.0.0.1:8050 with debug enabled

**Step 5: Commit**
```bash
git add app.py
git commit -m "refactor: update to Dash 3.x imports, remove Heroku code"
```

---

### Task 3: Update interpolation/utilfcts.py

**Files:**
- Modify: `interpolation/utilfcts.py`

**Step 1: Update imports**
Replace:
```python
import dash_core_components as dcc
```
With:
```python
from dash import dcc
```

**Step 2: Replace deprecated Rbf with RBFInterpolator**
Replace:
```python
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
...
inter = Rbf(x_mesh, y_mesh, z_mesh, function="thin_plate", smooth=0)
ZI = inter(XI, YI)
```
With:
```python
from scipy.interpolate import RBFInterpolator
...
rbf = RBFInterpolator(np.column_stack([x_mesh, y_mesh]), z_mesh, kernel="thin_plate", smoothing=0)
ZI = rbf(np.column_stack([XI.flatten(), YI.flatten()])).reshape(XI.shape)
```

**Step 3: Verify app still works**
Run: `uv run python app.py`
Expected: App runs, surfaces render correctly

**Step 4: Commit**
```bash
git add interpolation/utilfcts.py
git commit -m "refactor: update imports, replace deprecated Rbf with RBFInterpolator"
```

---

### Task 4: Remove Heroku artifacts

**Files:**
- Delete: `Procfile`
- Delete: `requirements.txt`

**Step 1: Delete Procfile**
Run: `rm Procfile`

**Step 2: Delete requirements.txt**
Run: `rm requirements.txt`

**Step 3: Commit**
```bash
git add -A
git commit -m "chore: remove Heroku deployment files"
```

---

### Task 5: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Update Build Commands section**
Replace the Running the Application section with:
```markdown
### Running the Application

```bash
# Install dependencies
uv sync

# Development server (with hot reload)
uv run python app.py

# Add a new dependency
uv add <package>
```
```

**Step 2: Update Project Structure**
Remove `Procfile` and `requirements.txt` from structure, add `pyproject.toml` and `uv.lock`.

**Step 3: Remove Deployment section**
Delete the Deployment section entirely.

**Step 4: Commit**
```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md for uv workflow"
```

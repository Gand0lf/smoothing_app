# AGENTS.md

This document provides essential information for AI coding agents working in this repository.

## Project Overview

A Dash/Flask web application for visualizing and comparing smoothing/interpolation methods (Thin Plate Splines, GAM, Linear Regression) on simulated data. Local development only.

## Build/Lint/Test Commands

### Running the Application

```bash
# Install dependencies
uv sync

# Development server (with hot reload)
uv run python app.py

# Add a new dependency
uv add <package>
```

### Testing

No test framework is currently configured. When adding tests:

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_module.py

# Run a single test function
pytest tests/test_module.py::test_function_name

# Run with verbose output
pytest -v tests/test_module.py
```

### Linting and Type Checking

No linting configuration exists. Recommended commands:

```bash
# Flake8 linting
flake8 . --max-line-length=120 --exclude=.git,__pycache__,.gitignore

# Type checking with mypy
mypy . --ignore-missing-imports

# Auto-format with black
black . --line-length=120
```

## Project Structure

```
smoothing_app/
├── app.py                 # Main Dash application entry point
├── pyproject.toml         # Project config and dependencies (uv)
├── .python-version        # Python version pin (3.12)
├── assets/                # Static assets (CSS, favicon, storage)
│   ├── base-styles.css    # Skeleton CSS framework
│   └── custom-styles.css  # Custom application styles
└── interpolation/         # Core computation module
    ├── __init__.py
    ├── utilfcts.py        # Main interpolation functions
    └── martingale.py      # Martingale class for yield curve analysis
```

## Code Style Guidelines

### Imports

```python
# Standard library first
from pathlib import Path

# Third-party libraries second (alphabetically grouped)
import dash
import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State
from pygam import LinearGAM, s, f, te
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import LinearRegression

# Local modules last
from interpolation.utilfcts import runall, createdata
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `createdata()`, `runall()` |
| Variables | snake_case | `z_mesh`, `n_splines` |
| Constants | UPPER_SNAKE_CASE | `VALID_USERNAME_PASSWORD_PAIRS` |
| Classes | PascalCase | `class martingale:` |
| Private/Internal | Leading underscore | `_temp`, `_shapetomesh` |
| Module-level "constants" | UPPER_CASE | `UPS`, `CENTERS`, `EYES` |

### Formatting

- **Line length**: 120 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Blank lines**: 2 blank lines between top-level functions/classes
- **Spaces around operators**: Use consistently (`x = 5`, not `x=5`)
- **Trailing commas**: Use in multi-line collections for cleaner diffs

### Types

Type hints are optional but encouraged for function signatures:

```python
def runall(flags: list, x, y, z, lambd: float = 0.6, nsplin: int = 20):
    ...

def generateNstore(n_clicks: int):
    ...
```

### Error Handling

Use try/except blocks for operations that may fail. Log or print exceptions for debugging:

```python
try:
    data = json.loads(data)
    x, y, z = data
    return runall(values, x, y, z, lambd=lambdaIn, nsplin=n_splines)
except Exception as e:
    print(e)
    x, y, z = list(createdata())
    return runall(values, x, y, z, lambd=lambdaIn, nsplin=n_splines)
```

### Dash Callbacks

Follow the existing pattern for Dash callbacks:

```python
@app.callback(
    Output("component-id", "property"),
    [Input("input-id", "value")],
    [State("state-id", "children")]
)
def callback_function(input_value, state_value):
    ...
```

### Comments

- Remove or update commented-out code rather than leaving it
- Use docstrings for classes and complex functions
- Comment "why" not "what" for non-obvious logic

## Key Patterns

### Dash Application Structure

1. Initialize app: `app = dash.Dash(__name__)`
2. Define layout with `html.Div` containers
3. Register callbacks with decorators
4. Run with `app.run(debug=True)` for development

### NumPy/SciPy Interpolation Pattern

```python
# Create meshgrid for interpolation
XI, YI = np.meshgrid(tix, tiy)

# Fit interpolator (RBFInterpolator replaces deprecated Rbf)
rbf = RBFInterpolator(
    np.column_stack([x_mesh, y_mesh]), 
    z_mesh, 
    kernel="thin_plate", 
    smoothing=0
)

# Predict on grid
ZI = rbf(np.column_stack([XI.flatten(), YI.flatten()])).reshape(XI.shape)
```

### Plotly Graph Pattern

```python
fig = go.Figure(data=[trace1, trace2])
fig.update_layout(
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    title="Title",
    width=900,
    height=600
)
return dcc.Graph(figure=fig, id='plot-id')
```

## Dependencies

Key packages and their purposes:

- **dash**: Web framework for the UI (v3.x)
- **plotly**: Interactive visualizations
- **numpy**: Numerical operations (v2.x)
- **scipy**: Interpolation algorithms (RBFInterpolator)
- **scikit-learn**: Linear regression fitting
- **pygam**: Generalized Additive Models
- **flask**: WSGI framework (Dash dependency)

# AGENTS.md

This document provides essential information for AI coding agents working in this repository.

## Project Overview

A Dash/Flask web application for visualizing and comparing smoothing/interpolation methods (Thin Plate Splines, GAM, Linear Regression) on simulated data. Deployed on Heroku.

## Build/Lint/Test Commands

### Running the Application

```bash
# Development server
python app.py

# Production (via gunicorn - used by Heroku)
gunicorn app:server

# Install dependencies
pip install -r requirements.txt
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
├── Procfile               # Heroku deployment config
├── requirements.txt       # Python dependencies
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
import socket
from pathlib import Path

# Third-party libraries second (alphabetically grouped)
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from pygam import LinearGAM, s, f, te
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
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
2. Expose Flask server: `server = app.server`
3. Define layout with `html.Div` containers
4. Register callbacks with decorators
5. Run with `app.run_server()` for development

### NumPy/SciPy Interpolation Pattern

```python
# Create meshgrid for interpolation
XI, YI = np.meshgrid(tix, tiy)

# Fit interpolator
inter = Rbf(x_mesh, y_mesh, z_mesh, function="thin_plate", smooth=0)

# Predict on grid
ZI = inter(XI, YI)
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

- **dash**: Web framework for the UI
- **plotly**: Interactive visualizations
- **numpy**: Numerical operations
- **scipy**: Interpolation algorithms (Rbf, splines)
- **scikit-learn**: Linear regression fitting
- **pygam**: Generalized Additive Models
- **gunicorn**: WSGI server for production

## Deployment

- Platform: Heroku
- Config: `Procfile` contains `web: gunicorn app:server`
- The `server` variable in `app.py` is the Flask app exposed for gunicorn

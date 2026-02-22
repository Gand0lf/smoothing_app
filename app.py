import dash
from dash import html, dcc, Input, Output, State
from pathlib import Path
import numpy as np
import json

from interpolation.utilfcts import runall, createdata

np.random.seed(seed=8080)
VALID_USERNAME_PASSWORD_PAIRS = {"testuser": "123"}

app = dash.Dash(__name__)
server = app.server
path = Path()
assets = path / "assets"


def drawcontents():
    return html.Div(
        className="row",
        children=[
            html.Div(id="storage", style={"display": "none"}),
            html.Div(
                id="left-column",
                children=[
                    html.Div(
                        className="card",
                        id="first-card",
                        children=[
                            html.Div(
                                className="graph-checkbox",
                                children=["Display Options"],
                            ),
                            html.Span(
                                "Select interpolation methods to visualize",
                                style={
                                    "color": "#a5b1cd",
                                    "font-size": "0.85rem",
                                    "margin-bottom": "8px",
                                    "display": "block",
                                },
                            ),
                            dcc.Checklist(
                                options=[
                                    {"label": "Thin Plate Spline", "value": "tps"},
                                    {"label": "Least Squares Fit", "value": "lsq"},
                                    {"label": "Sample Points", "value": "sample"},
                                    {
                                        "label": "GAM (Generalized Additive)",
                                        "value": "gam",
                                    },
                                ],
                                value=["sample"],
                                id="checklist-options",
                                className="checklist-smoothing",
                            ),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.P(
                                "GAM Parameters",
                                style={
                                    "font-weight": "600",
                                    "margin-bottom": "4px",
                                    "color": "#fff",
                                    "font-size": "1.1rem",
                                },
                            ),
                            html.Span(
                                "Configure the Generalized Additive Model smoothing",
                                style={
                                    "color": "#a5b1cd",
                                    "font-size": "0.85rem",
                                    "margin-bottom": "12px",
                                    "display": "block",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "# Splines:",
                                                id="splines-label",
                                                style={
                                                    "color": "#fff",
                                                    "font-weight": "500",
                                                    "font-size": "0.95rem",
                                                    "cursor": "help",
                                                    "border-bottom": "1px dotted #13c6e9",
                                                },
                                                title="Number of basis splines. More splines = more flexibility but risk of overfitting.",
                                            ),
                                            dcc.Slider(
                                                min=1,
                                                max=40,
                                                step=5,
                                                marks={
                                                    i * 10: str(i * 10)
                                                    for i in range(1, 5)
                                                },
                                                value=20,
                                                updatemode="drag",
                                                id="slider-n",
                                            ),
                                        ],
                                        style={"margin-bottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Lambda (smoothing):",
                                                id="lambda-label",
                                                style={
                                                    "color": "#fff",
                                                    "font-weight": "500",
                                                    "font-size": "0.95rem",
                                                    "cursor": "help",
                                                    "border-bottom": "1px dotted #13c6e9",
                                                },
                                                title="Smoothing penalty. Higher values produce smoother curves (less wiggle).",
                                            ),
                                            dcc.Input(
                                                id="lambdaIn",
                                                type="number",
                                                value=0.6,
                                                min=0,
                                                max=1000,
                                                step=0.05,
                                            ),
                                        ],
                                    ),
                                ],
                                className="slider-smoothing",
                            ),
                        ],
                    ),
                    html.Div(
                        className="card",
                        id="last-card",
                        children=[
                            html.Span(
                                "Data Controls",
                                style={
                                    "color": "#fff",
                                    "font-weight": "600",
                                    "font-size": "1.1rem",
                                    "display": "block",
                                    "margin-bottom": "8px",
                                },
                            ),
                            html.Span(
                                "Generate new simulated yield curve data",
                                style={
                                    "color": "#a5b1cd",
                                    "font-size": "0.85rem",
                                    "margin-bottom": "12px",
                                    "display": "block",
                                },
                            ),
                            html.Button(
                                "Generate New Sample",
                                id="sampler",
                                n_clicks=0,
                                title="Click to generate a new random yield curve dataset",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="conti", className="ten columns"),
        ],
    )


app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="container scalable",
                    children=[
                        dcc.Markdown("# Smoothing & Interpolation Studies"),
                        html.P(
                            "Compare Thin Plate Splines, GAM, and Linear Regression on simulated yield curve data",
                            className="app-subtitle",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(className="container", children=[drawcontents()]),
    ]
)


@app.callback(
    Output("conti", "children"),
    [
        Input("checklist-options", "value"),
        Input("slider-n", "value"),
        Input("lambdaIn", "value"),
        Input("sampler", "n_clicks"),
    ],
    [State("storage", "children")],
)
def updategraph(values, n_splines, lambdaIn, click, data):
    try:
        data = json.loads(data)
        x, y, z = data
        return runall(values, x, y, z, lambd=lambdaIn, nsplin=n_splines)
    except Exception as e:
        print(e)
        x, y, z = list(createdata())
        return runall(values, x, y, z, lambd=lambdaIn, nsplin=n_splines)


@app.callback(Output("storage", "children"), [Input("sampler", "n_clicks")])
def generateNstore(n_clicks):
    x, y, z = list(createdata())
    _temp = json.dumps([x.tolist(), y.tolist(), z.tolist()])
    return _temp


if __name__ == "__main__":
    app.title = "Smoothing Study"
    app.run(debug=True)

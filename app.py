import dash
import socket
import dash_auth
import dash_html_components as html
import dash_core_components as dcc
from pathlib import Path
from dash.dependencies import Input, Output, State
import numpy as np
import json
#import pandas as pd
#import sqlite3

from interpolation.utilfcts import runall, createdata

np.random.seed(seed=8080)
VALID_USERNAME_PASSWORD_PAIRS = {'testuser': '123'}

ip = socket.gethostbyname(socket.gethostname())
portid = 8080

app = dash.Dash(__name__)
server = app.server
path = Path()
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
assets = path / 'assets'


def drawcontents():
    return html.Div(
        className="row",
        children=[
            html.Div(id='storage', style={'display': 'none'}),
            html.Div(
                className="two columns",
                style={"padding-bottom": "5%"},
                children=[
                    html.Div(
                        [
                            html.Div(
                                className="graph-checkbox",
                                children=["Display:"],
                            ),
                            dcc.Checklist(
                                options=[
                                    {"label": "Thin plate", "value": "tps"},
                                    {"label": "Least Squares Fit", "value": "lsq"},
                                    {"label": "Sample", "value": "sample"},
                                    {"label": "GAM", "value": "gam"},
                                ],
                                value=["sample"],
                                id=f"checklist-options",
                                className="checklist-smoothing",
                            ),
                        ],
                        style={"margin-top": "10px"},
                    ),
                    html.Div(
                        [
                            html.P(
                                "# GAM Splines:",
                                style={"font-weight": "bold", "margin-bottom": "0px"},
                                className="plot-display-text",
                            ),
                            dcc.Slider(
                                min=1,
                                max=40,
                                step=5,
                                marks={i * 10: str(i * 10) for i in range(1, 5)},
                                value=20,
                                updatemode="drag",
                                id="slider-n",
                            ),
                            html.P(
                                "lambda:",
                                style={"font-weight": "bold", "margin-bottom": "0px"},
                                className="plot-display-text",
                            ),
                            dcc.Input(
                                id="lambdaIn",
                                type="number",
                                value=0.6,
                                min=0, max=1000, step=0.05,
                                style=dict(width="100px",color="#a5b1cd"),
                            )
                        ],
                        style={"margin-bottom": "40px"},
                        className="slider-smoothing",
                    ),
                    html.Div(
                        [
                            html.Button('Resample', id='sampler', n_clicks=0, style={"width":"100px","color":"#a5b1cd","text-align":"left","padding":"0 10px"},),
                        ],
                    ),
                ],
            ),
            html.Div(id="conti", className="ten columns")
        ],
    )


app.layout = html.Div(
    children=[
        dcc.Markdown("# Smoothness Studies"),
        html.Div(className="container", children=[drawcontents()]),
    ]
)


@app.callback(
    Output("conti", "children"),
    [Input("checklist-options", "value"),Input("slider-n", "value"),Input("lambdaIn", "value"),Input("sampler", "n_clicks")], [State('storage', 'children')],
)
def updategraph(values,n_splines,lambdaIn,click ,data):
    try:
        data = json.loads(data)
        x, y, z = data
        return runall(values, x, y, z,lambd=lambdaIn,nsplin=n_splines)
    except Exception as e:
        print(e)
        x, y, z = list(createdata())
        return runall(values, x, y, z,lambd=lambdaIn,nsplin=n_splines)


@app.callback(
    Output("storage", "children"),
    [Input("sampler", "n_clicks")]
)
def generateNstore(n_clicks):
    x, y, z = list(createdata())
    _temp = json.dumps([x.tolist(), y.tolist(), z.tolist()])
    return _temp


if __name__ == '__main__':
    print("hostname:" + str(ip) + str(portid))
    app.title = "Smoothing Study"
    app.run_server(debug=False)#, host=ip, port=portid)
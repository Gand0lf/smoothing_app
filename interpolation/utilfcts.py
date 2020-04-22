#from typing import List, Any

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from sklearn.linear_model import LinearRegression
#import plotly.express as px
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
#import math
#import cufflinks as cf
from pygam import LinearGAM, s, f, te
import dash_core_components as dcc
np.random.seed(seed=8080)

def createdata():
    x = np.linspace(0.0, 30.0, num=10)
    y = np.linspace(0, 50, num=50)

    def morph(c, m, x):
        _sqrt = np.sqrt(x)
        _new = c + m * _sqrt
        return (_new)

    # trend=list(map(math.sin,y));trend

    trend = 0.25 * np.sin(y)

    container = []
    for i in y:
        _c = 0.25 * np.random.randn(1)
        _m = 0.25 * np.random.rand(1)
        _yc = morph(_c, _m, x)
        container.append(_yc)

    container = np.array(container)
    matched_trend = np.broadcast_to(trend[None].transpose(), (50, 10))

    z = container + matched_trend
    return x, y, z


def runall(flags: list, x, y, z,lambd=0.6,nsplin=20):
    x,y,z=np.array(x),np.array(y),np.array(z)

    _shapetomesh = pd.DataFrame(z, columns=x, index=y)

    _shapetomesh.head()

    __t = _shapetomesh.iterrows()

    storage = []
    dic = _shapetomesh.to_dict()
    for key, value in dic.items():
        for subkey, subvalue in value.items():
            storage.append([key, subkey, subvalue])
    storage = np.array(storage)

    x_mesh, y_mesh, z_mesh = storage.T[0], storage.T[1], storage.T[2]

    tix = np.linspace(0, 30, 10)
    tiy = np.append(y, y[-1] + y[1:11])  # 10 more steps for extrapolation

    XI, YI = np.meshgrid(tix, tiy)
    # fit in sample
    inter = Rbf(x_mesh, y_mesh, z_mesh, function="thin_plate", smooth=0)
    ZI = inter(XI, YI)  # prediction step with
    def sumSQloss(y, yhat):
        return np.sum((y - yhat) ** 2)

    _X = np.array([x_mesh, y_mesh]).T  # in sample
    reg = LinearRegression().fit(_X, z_mesh)
    zHATlin = reg.predict(_X)
    sumSQloss(storage.T[2], zHATlin)
    s_order = 3

    lam = lambd
    s_num = nsplin

    inter = np.array([XI.flatten(), YI.flatten()]).T

    gam = LinearGAM(s(0, n_splines=s_num, spline_order=s_order, lam=lam) + s(1, n_splines=s_num, spline_order=s_order,
                                                                             lam=lam)).fit(_X,z_mesh)  # in sample fitting
    # gam = LinearGAM(te(0,1,n_splines=100,lam=lam)).fit(_X, z_mesh) #in sample fitting
    zHATgam = gam.predict(_X)
    zHatgam_pred = gam.predict(inter).reshape(60, 10)

    loss = sumSQloss(storage.T[2], ZI[:50].T.flatten())

    _tps = go.Surface(x=tix, y=tiy, z=ZI, opacity=0.9,coloraxis="coloraxis",legendgroup="group"),
    _gam = go.Surface(x=tix, y=tiy, z=zHatgam_pred, opacity=0.9, coloraxis="coloraxis",legendgroup="group"),
    _points = go.Scatter3d(x=storage.T[0], y=storage.T[1], z=storage.T[2],
                           legendgroup="group1",
                           mode='markers',
                           marker=dict(
                               symbol="x",
                               size=2,
                               color="black",
                               # color=storage.T[2],                # set color to an array/list of desired values
                               # colorscale='Viridis',   # choose a colorscale
                               opacity=1
                           ),
                           ),
    _linplane = go.Scatter3d(x=storage.T[0], y=storage.T[1], z=zHATlin,
                             legendgroup="group1",
                             mode='markers',
                             marker=dict(
                                 size=2.5,
                                 color="red",
                                 # color=storage.T[2],                # set color to an array/list of desired values
                                 # colorscale='Viridis',   # choose a colorscale
                                 opacity=1
                             )
                             ),
    selection = []
    if "tps" in flags:
        selection.append(_tps[0])
    if "gam" in flags:
        selection.append(_gam[0])
    if "lsq" in flags:
        selection.append(_linplane[0])
    if "sample" in flags:
        selection.append(_points[0])
    fig_i = go.Figure(
        data=selection,
        #layout=dict(scene=dict(camera={'up': UPS[0], 'center': CENTERS[0], 'eye': EYES[0]})),
        #https://plotly.com/python/reference/#layout-scene-camera
    )

    fig_i.update_layout(
        scene=dict(
            xaxis_title='X Maturities',
            yaxis_title='Y Time',
            zaxis_title='Z Yield'),
        title=f"Simulated data",
        width=900,
        height=600,
        margin=dict(r=20, l=10, b=10, t=10),
        scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=2, z=1),
        showlegend=True,
        #coloraxis_showscale=False,
    )
    fig_i.layout.title.y = 0.95
    fig_i.layout.coloraxis.colorscale = "Viridis"
    plot = dcc.Graph(figure=fig_i, style={'width': '100%', 'height': '100%'}, id='plot')
    return plot


UPS = {
    0: dict(x=0, y=0, z=1),
}

CENTERS = {
    0: dict(x=1.25, y=1.25, z=1.25),
}

EYES = {
    0: dict(x=2, y=2, z=2),
}
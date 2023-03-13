# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import cv2
import numpy as np
import pathlib
from typing import List
import dash_bootstrap_components as dbc
import os

app = Dash(__name__)
# rootfolder = "/home/l049e/cluster_checkpoints/dermoscopyall_ce_run1/"
# agg_df_for_app(rootfolder=rootfolder)
colors = {"background": "white", "text": "black"}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
dataframes_paths = pathlib.Path("./dash_data").glob("*")
DFS = [pd.read_csv(x) for x in dataframes_paths if x.is_file()]
dataframes_paths = pathlib.Path("./dash_data").glob("*")
names = [x.stem for x in dataframes_paths if x.is_file()]

df_dict = {}
for i, name in enumerate(names):
    df_dict[name] = DFS[i]
df = DFS[0]

color_map = {
    "TN": "rgb(26,150,65)",
    "FN": "rgb(166,217,106)",
    "FP": "rgb(253,174,97)",
    "TP": "rgb(215,25,28)",
}


def generate_figures(df):
    figures = []
    for cla in df.label.unique():
        fig = go.Figure()
        for cf in ["TP", "TN", "FP", "FN"]:
            if cf == "TP":
                sub_df = df[(df.label == cla) & (df.predicted == cla)]
            if cf == "TN":
                sub_df = df[(df.label != cla) & (df.predicted != cla)]
            if cf == "FP":
                sub_df = df[(df.label != cla) & (df.predicted == cla)]
            if cf == "FN":
                sub_df = df[(df.label == cla) & (df.predicted != cla)]

            fig.add_trace(
                go.Scatter3d(
                    x=sub_df["0"],
                    y=sub_df["1"],
                    z=sub_df["2"],
                    opacity=0.6,
                    mode="markers",
                    name=cf,
                    text=sub_df["filepath"],
                    marker=dict(size=3, color=color_map[cf], symbol="circle"),
                )
            )
        fig.update_layout(
            coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
        )
        fig = go.FigureWidget(fig.data, fig.layout)
        fig.layout.hovermode = "closest"
        fig.update_layout(width=1000, height=1000, template="simple_white")
        figures.append(fig)

        text_array = np.append(fig["data"][0]["text"], fig["data"][1]["text"])
        text_array = np.append(text_array, fig["data"][2]["text"])
        text_array = np.append(text_array, fig["data"][3]["text"])
    return figures, text_array


FIGURES = {}
TEXT_ARRAYS = {}

for id, df in enumerate(DFS):
    fig_ls, text_array = generate_figures(df)
    FIGURES[names[id]] = fig_ls
    TEXT_ARRAYS[names[id]] = text_array


FIGURE = FIGURES[names[0]][0]
TEXT_ARRAY = TEXT_ARRAYS[names[0]][0]
# START = df.filepath[0]
# IMG_PATH = df.loc[df["filepath"] == START]["filepath"].iloc[0]
# start, end = IMG_PATH.split("l049e/")
# filep = "/home/l049e/Data/" + end
# imbgr = cv2.imread(filep)
# im = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
# IMG = go.Figure()
# IMG.add_trace(go.Image(z=im))
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H2(
            children="Select datafile and class to display:",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            [
                html.Div(id="test"),
                html.Div(
                    [
                        html.Div("Select Data for plotting", id="data_dropdown"),
                        dcc.Dropdown(
                            names,
                            names[0],
                            id="dropdown_data",
                        ),
                        html.Div("Select Class for plotting", id="header_dropdown"),
                        dcc.Dropdown(
                            np.sort(df.label.unique()),
                            np.sort(df.label.unique())[0],
                            id="dropdown_class",
                        ),
                    ]
                ),
                dcc.Graph(
                    id="scatter",
                    hoverData={"points": [{"pointNumber": 0}]},
                    figure=FIGURE,
                    style={
                        "display": "inline-block",
                        "width": "800px",
                        "height": "800px",
                    },
                ),
                dcc.Graph(
                    id="img_plot",
                    style={
                        "display": "inline-block",
                        "width": "800px",
                        "height": "800px",
                    },
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("scatter", "figure"),
    Input("dropdown_class", "value"),
    Input("dropdown_data", "value"),
)
def update_encoder(dropdown_class, dropdown_data):
    global TEXT_ARRAY
    TEXT_ARRAY = TEXT_ARRAYS[dropdown_data][dropdown_class]
    figure = FIGURES[dropdown_data][dropdown_class]

    return figure


@app.callback(Output("dropdown_class", "options"), Input("dropdown_data", "value"))
def update_class_dropdown(data_name):
    df = df_dict[data_name]
    return np.sort(df.label.unique())


@app.callback(
    Output("img_plot", "figure"),
    # Output("test", "children"),
    Input("scatter", "hoverData"),
)
def update_on_hover(hoverData):
    """ """
    if hoverData is None:
        raise PreventUpdate
    try:
        imgpath = hoverData["points"][0]["text"]
        _, end = imgpath.split("/l049e")
        if "cluster" in imgpath:
            filep2 = os.path.join("/home/l049e/Data/" + end)
        if "home" in imgpath:
            filep2 = imgpath
        imbgr2 = cv2.imread(filep2)
        im2 = cv2.cvtColor(imbgr2, cv2.COLOR_BGR2RGB)
        fig2 = go.Figure()
        fig2.add_trace(go.Image(z=im2))
        fig2.update_layout(width=1000, height=1000, template="simple_white")
        return fig2
        # return filep2
    except Exception as error:
        print(error)
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True, port=8055)

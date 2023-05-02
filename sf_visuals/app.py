import argparse
import base64
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from sf_visuals.analyser import Analyser


def main():
    app = Dash(__name__)
    colors = {"background": "white", "text": "black"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    analyser = Analyser(path=args.path)

    app.layout = dcc.Tabs(
        parent_className="container",
        className="tab-container",
        children=[
            dcc.Tab(
                label="Latent Space",
                children=[
                    html.Div(
                        id="sidebar",
                        style={"display": "inline-block"},
                        children=[
                            html.H2(
                                children=f"Experiment folder:",
                            ),
                            html.P(f"{args.path}"),
                            html.H3(
                                children="Classes:",
                            ),
                            dcc.Checklist(
                                [
                                    {
                                        "label": html.Div(
                                            [
                                                f"Class {c}: ",
                                                dcc.Input(
                                                    id={
                                                        "type": "class-name",
                                                        "id": f"{c}",
                                                    },
                                                    type="text",
                                                    value=f"{c}",
                                                ),
                                            ],
                                            style={"display": "inline-block"},
                                        ),
                                        "value": c,
                                    }
                                    for c in analyser.classes
                                ],
                                analyser.classes,
                                id="checklist-classes",
                            ),
                            html.H3(
                                children="Datasets:",
                            ),
                            dcc.RadioItems(
                                [{"label": "All", "value": "ALL"}]
                                + [
                                    {
                                        "label": html.Div(
                                            [
                                                f"Testset {c}",
                                            ],
                                            style={"display": "inline-block"},
                                        ),
                                        "value": c,
                                    }
                                    for c in analyser.testsets
                                ],
                                "ALL",
                                id="checklist-testsets",
                            ),
                            html.H3(
                                children="Color By:",
                            ),
                            dcc.RadioItems(
                                [
                                    {"label": "Confidence", "value": "confidence"},
                                    {
                                        "label": "Source/Target",
                                        "value": "source-target",
                                    },
                                ],
                                "confidence",
                                id="checklist-colorby",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="latentspace",
                        className="latentspace",
                        figure=analyser.plot_latentspace(analyser.testsets[1]),
                        responsive=True,
                        clear_on_unhover=True,
                    ),
                    html.Div(
                        className="hover-preview",
                        children=[
                            html.Img(id="curimg", width="512px", height="512px"),
                        ],
                    ),
                    html.Div(children=[], hidden=True, id="dummy"),
                ],
            ),
            dcc.Tab(label="Failures", children=[html.H2("WIP")]),
        ],
    )

    @app.callback(
        Output("dummy", "children"),
        Input({"type": "class-name", "id": ALL}, "value"),
        State({"type": "class-name", "id": ALL}, "id"),
    )
    def update_class_name(value, id):
        classes = {}
        for i, v in zip(id, value):
            classes[int(i["id"])] = v
        analyser.classes = classes
        return None

    @app.callback(
        Output("latentspace", "figure"),
        Input("checklist-testsets", "value"),
        Input("checklist-classes", "value"),
        Input("checklist-colorby", "value"),
    )
    def update_testset(testset, classes, colorby):
        return analyser.plot_latentspace(
            testset, classes2plot=tuple(classes), coloring=colorby
        )

    @app.callback(
        Output("curimg", "src"),
        Input("latentspace", "hoverData"),
        Input("latentspace", "clickData"),
    )
    def update_on_hover(hoverData, clickData):
        if hoverData is None:
            if clickData is None:
                raise PreventUpdate
            hoverData = clickData

        imgpath = hoverData["points"][0]["text"]
        imgpath = imgpath.replace(
            "/dkfz/cluster/gpu/data/OE0612/l049e/", "/home/t974t/Data/levin/"
        )
        with open(imgpath, "rb") as img:
            data = base64.b64encode(img.read()).replace(b"\n", b"").decode("utf-8")
            return f"data:image/jpeg;base64,{data}"

    app.run(host="0.0.0.0", debug=True, port="8055")


if __name__ == "__main__":
    main()
